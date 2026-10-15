"""
order_manager.py - Gestionnaire d'ordres, contrôle des risques et persistance
SQLite pour pocket-ai-trader.

Ce module est le point d'entrée unique vers l'exécution d'un ordre. Il a été
reconstruit pour satisfaire l'interface réellement exposée par le reste du
dépôt, et non une interface supposée :

    tests/test_order_manager_connector.py
        OrderManager(connector=..., max_position_size=...)
        submit_order(asset=, direction=, amount=, payout=,
                     expiration_seconds=) -> dict possédant "status"
        Le connecteur reçoit un objet exposant
        .order_id .asset .direction .amount .payout .expiration_seconds

    po_connector.PocketOptionConnector
        execute_order(order) -> bool
        (True = confirmé, False = non confirmé, exception = échec)

    data_store.DataStore
        save_order(order_record)  -> INSERT OR REPLACE sur order_id
        get_daily_stats(date)     -> {"realized_pnl": float, ...}

PRINCIPE STRUCTURANT : PERSISTANCE AVANT EXÉCUTION
--------------------------------------------------
Aucun ordre ne peut atteindre le connecteur sans qu'une ligne durable
(status "SUBMITTED") n'existe déjà en base. Si cette écriture échoue,
l'ordre n'est pas envoyé et la réponse porte le statut "FAILED".
Le cycle de vie d'un ordre est donc toujours traçable :

    SUBMITTED -> ACCEPTED | FAILED
    REJECTED  (refusé avant toute tentative d'exécution)

RÈGLE DE CONCEPTION : ÉCHEC FERMÉ
---------------------------------
Toute ambiguïté (montant illisible, perte journalière illisible, connecteur
muet) se traduit par un refus, jamais par un envoi.

Les seuils de sécurité sont des constantes de ce module, volontairement non
pilotées par l'environnement, afin que le comportement soit reproductible en
test. Elles reprennent les valeurs par défaut de Config.py
(PO_MIN_PAYOUT_PERCENT=80.0, PO_MIN_EXPIRATION_SECONDS=60).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(
    "pocket_ai_trader.order_manager"
)

if not logger.handlers:

    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)

logger.setLevel(logging.INFO)


# ============================================================
# STATUTS
# ============================================================

STATUS_SUBMITTED = "SUBMITTED"
STATUS_ACCEPTED = "ACCEPTED"
STATUS_REJECTED = "REJECTED"
STATUS_FAILED = "FAILED"


# ============================================================
# VALEURS PAR DÉFAUT DES GARDE-FOUS
# ============================================================

# Taille de position maximale par ordre (unités de compte).
DEFAULT_MAX_POSITION_SIZE = 50.0

# Perte journalière réalisée au-delà de laquelle tout nouvel ordre est refusé.
DEFAULT_MAX_DAILY_LOSS = 100.0

# Fenêtre anti-doublon : deux ordres identiques (même actif, même sens) dans
# cet intervalle sont considérés comme un doublon.
DEFAULT_DUPLICATE_WINDOW_SECONDS = 10

# Payout minimal accepté. En dessous, l'espérance mathématique devient
# structurellement défavorable (voir break_even_win_rate()).
DEFAULT_MIN_PAYOUT_PERCENT = 80.0

# Expiration minimale acceptée, alignée sur le connecteur et Config.py.
DEFAULT_MIN_EXPIRATION_SECONDS = 60

VALID_DIRECTIONS: Tuple[str, ...] = (
    "BUY",
    "SELL",
)


# ============================================================
# OUTILS
# ============================================================

def _to_float(value: Any) -> Optional[float]:
    """Convertit en float, ou retourne None si la valeur est inexploitable."""

    if value is None or isinstance(value, bool):
        return None

    try:
        return float(value)

    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    """Convertit en int, ou retourne None si la valeur est inexploitable."""

    if value is None or isinstance(value, bool):
        return None

    try:
        return int(float(value))

    except (TypeError, ValueError):
        return None


def _normalize_text(value: Any) -> str:
    """Normalise un sens d'ordre : majuscules, sans espaces parasites."""

    if value is None:
        return ""

    return str(value).strip().upper()


def _normalize_asset(value: Any) -> str:
    """
    Normalise un actif SANS toucher à la casse.

    Les symboles Pocket Option sont sensibles à la casse et contiennent un
    suffixe en minuscules (ex. "#TSLA_otc"). Les passer en majuscules produit
    un symbole inexistant côté plateforme.
    """

    if value is None:
        return ""

    return str(value).strip()


def break_even_win_rate(payout_percent: Any) -> float:
    """
    Taux de réussite nécessaire pour ne pas perdre d'argent.

        break_even = 100 / (1 + payout / 100)

    À 80 % de payout, il faut gagner 55,6 % des trades pour être à
    l'équilibre. C'est la raison d'être du seuil minimal de payout.
    """

    payout = _to_float(payout_percent)

    if payout is None or payout <= 0:
        return 100.0

    return 100.0 / (1.0 + payout / 100.0)


# ============================================================
# OBJET ORDRE TRANSMIS AU CONNECTEUR
# ============================================================

@dataclass
class Order:
    """
    Ordre normalisé transmis à PocketOptionConnector.execute_order().

    Les six attributs sont exactement ceux lus par
    po_connector.PocketOptionConnector._validate_order().
    """

    order_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    asset: str = ""

    direction: str = ""

    amount: float = 0.0

    payout: float = 0.0

    expiration_seconds: int = DEFAULT_MIN_EXPIRATION_SECONDS


# ============================================================
# GESTIONNAIRE D'ORDRES
# ============================================================

class OrderManager:
    """
    Applique les règles de risque, trace l'ordre, puis demande l'exécution
    au connecteur injecté.
    """

    def __init__(
        self,
        connector: Any,
        data_store: Any = None,
        max_position_size: Optional[float] = (
            DEFAULT_MAX_POSITION_SIZE
        ),
        max_daily_loss: Optional[float] = (
            DEFAULT_MAX_DAILY_LOSS
        ),
        duplicate_window_seconds: int = (
            DEFAULT_DUPLICATE_WINDOW_SECONDS
        ),
        min_payout_percent: float = (
            DEFAULT_MIN_PAYOUT_PERCENT
        ),
        min_expiration_seconds: int = (
            DEFAULT_MIN_EXPIRATION_SECONDS
        ),
    ):

        if connector is None:
            raise ValueError(
                "connector est obligatoire : "
                "le gestionnaire ne place jamais d'ordre lui-même."
            )

        self.connector = connector
        self.data_store = data_store

        self.max_position_size = (
            None
            if max_position_size is None
            else float(max_position_size)
        )

        self.max_daily_loss = (
            None
            if max_daily_loss is None
            else float(max_daily_loss)
        )

        self.duplicate_window_seconds = int(
            duplicate_window_seconds
        )

        self.min_payout_percent = float(
            min_payout_percent
        )

        self.min_expiration_seconds = int(
            min_expiration_seconds
        )

        self.emergency_stop_triggered: bool = False
        self.emergency_stop_reason: Optional[str] = None

        # Historique mémoire des ordres réellement acceptés. Sert à la
        # détection des doublons lorsque aucun data_store n'est fourni.
        self.order_history: List[Dict[str, Any]] = []

    # =========================================================
    # ARRÊT D'URGENCE
    # =========================================================

    def trigger_emergency_stop(
        self,
        reason: str = "Arrêt manuel d'urgence",
    ) -> None:
        """Bloque immédiatement l'envoi de nouveaux ordres."""

        self.emergency_stop_triggered = True
        self.emergency_stop_reason = str(reason)

        logger.warning(
            "ARRÊT D'URGENCE | %s",
            self.emergency_stop_reason,
        )

    def reset_emergency_stop(self) -> None:
        """Réactive l'envoi d'ordres après vérification manuelle."""

        self.emergency_stop_triggered = False
        self.emergency_stop_reason = None

        logger.warning(
            "Arrêt d'urgence levé : les ordres peuvent à nouveau être soumis."
        )

    # =========================================================
    # DÉTECTION DES DOUBLONS
    # =========================================================

    def is_duplicate(
        self,
        asset: str,
        direction: str,
        now: Optional[datetime] = None,
    ) -> bool:
        """
        Vrai si un ordre identique a déjà été accepté dans la fenêtre
        anti-doublon. L'historique est parcouru du plus récent au plus ancien.
        """

        if now is None:
            now = datetime.now(timezone.utc)

        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        asset = _normalize_asset(asset)
        direction = _normalize_text(direction)

        for record in reversed(self.order_history):

            if not isinstance(record, dict):
                continue

            timestamp = record.get("timestamp")

            if not timestamp:
                continue

            try:
                order_time = datetime.fromisoformat(
                    str(timestamp)
                )

            except (TypeError, ValueError):
                continue

            if order_time.tzinfo is None:
                order_time = order_time.replace(
                    tzinfo=timezone.utc
                )

            if (
                (now - order_time).total_seconds()
                > self.duplicate_window_seconds
            ):
                # L'historique est ajouté dans l'ordre chronologique :
                # les entrées suivantes sont encore plus anciennes.
                break

            if (
                _normalize_asset(record.get("asset")) == asset
                and _normalize_text(
                    record.get("direction")
                ) == direction
            ):
                return True

        return False

    # =========================================================
    # PERTE JOURNALIÈRE
    # =========================================================

    def get_daily_loss(
        self,
        today: Optional[date] = None,
    ) -> float:
        """
        Perte réalisée du jour, lue dans le data_store.

        Sans data_store, la perte est inconnue et vaut 0.0 : il n'existe
        alors aucune source à interroger. Une exception du data_store est
        propagée, et non masquée : validate_order() la transforme en refus.
        """

        if self.data_store is None:
            return 0.0

        if today is None:
            today = datetime.now(
                timezone.utc
            ).date()

        stats = self.data_store.get_daily_stats(today)

        if not isinstance(stats, dict):
            raise ValueError(
                "get_daily_stats n'a pas retourné de dictionnaire."
            )

        pnl = _to_float(stats.get("realized_pnl", 0.0))

        if pnl is None:
            raise ValueError(
                "realized_pnl illisible dans les statistiques du jour."
            )

        return abs(pnl) if pnl < 0 else 0.0

    # =========================================================
    # VALIDATION DES RISQUES
    # =========================================================

    def validate_order(
        self,
        order_data: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """
        Contrôle les règles de sécurité avant toute exécution.

        Retourne (autorisé, motif_de_refus). Aucun ordre n'est envoyé tant
        que cette méthode n'a pas retourné (True, None).
        """

        if not isinstance(order_data, dict):
            return (
                False,
                "REJET : order_data doit être un dictionnaire.",
            )

        # -----------------------------------------------------
        # ARRÊT D'URGENCE
        # -----------------------------------------------------

        if self.emergency_stop_triggered:
            return (
                False,
                "REJET : Arrêt d'urgence actif "
                f"({self.emergency_stop_reason}).",
            )

        # -----------------------------------------------------
        # ACTIF
        # -----------------------------------------------------

        asset = _normalize_asset(order_data.get("asset"))

        if not asset:
            return (
                False,
                "REJET : asset est obligatoire.",
            )

        # -----------------------------------------------------
        # MONTANT
        # -----------------------------------------------------

        amount = _to_float(order_data.get("amount"))

        if amount is None:
            return False, "REJET : amount invalide."

        if amount <= 0:
            return (
                False,
                "REJET : Le montant doit être supérieur à zéro.",
            )

        if (
            self.max_position_size is not None
            and amount > self.max_position_size
        ):
            return (
                False,
                "REJET : Montant supérieur au maximum autorisé "
                f"({self.max_position_size}).",
            )

        # -----------------------------------------------------
        # SENS
        # -----------------------------------------------------

        direction = _normalize_text(
            order_data.get("direction")
        )

        if direction not in VALID_DIRECTIONS:
            return (
                False,
                "REJET : direction invalide "
                f"({order_data.get('direction')!r}) ; "
                "attendu BUY ou SELL.",
            )

        # -----------------------------------------------------
        # EXPIRATION
        # -----------------------------------------------------

        expiration = _to_int(
            order_data.get("expiration_seconds")
        )

        if expiration is None:
            return (
                False,
                "REJET : expiration_seconds invalide.",
            )

        if expiration < self.min_expiration_seconds:
            return (
                False,
                "REJET : expiration minimale autorisée "
                f"({self.min_expiration_seconds} secondes).",
            )

        # -----------------------------------------------------
        # PAYOUT
        # -----------------------------------------------------

        payout = _to_float(order_data.get("payout"))

        if payout is None:
            return False, "REJET : payout invalide ou absent."

        if payout < self.min_payout_percent:
            return (
                False,
                "REJET : payout "
                f"{payout} % inférieur au seuil "
                f"({self.min_payout_percent} %). "
                "Taux de réussite requis à ce payout : "
                f"{break_even_win_rate(payout):.2f} %.",
            )

        # -----------------------------------------------------
        # PERTE JOURNALIÈRE
        # -----------------------------------------------------

        if self.max_daily_loss is not None:

            try:
                current_loss = self.get_daily_loss()

            except Exception as exc:

                logger.exception(
                    "Perte journalière illisible."
                )

                return (
                    False,
                    "REJET : perte journalière illisible "
                    f"({exc}).",
                )

            if current_loss >= self.max_daily_loss:
                return (
                    False,
                    "REJET : Perte journalière maximale atteinte "
                    f"({current_loss} / {self.max_daily_loss}).",
                )

        return True, None

    # =========================================================
    # PERSISTANCE
    # =========================================================

    def _persist(
        self,
        response: Dict[str, Any],
        spec: Dict[str, Any],
    ) -> Optional[Exception]:
        """
        Écrit la tentative d'ordre en base.

        Retourne None si l'écriture a réussi, sinon l'exception rencontrée.
        """

        if self.data_store is None:
            return None

        record = {
            "order_id": response.get("order_id"),
            "asset": spec.get("asset", ""),
            "direction": spec.get("direction", ""),
            "amount": spec.get("amount", 0.0),
            "expiration_seconds": spec.get(
                "expiration_seconds",
                0,
            ),
            "status": response.get("status"),
            "reason": response.get("reason"),
            "timestamp": response.get("timestamp"),
            "connector_response": (
                response.get("connector_response") or {}
            ),
        }

        try:
            self.data_store.save_order(record)

        except Exception as exc:

            logger.exception(
                "Échec de persistance de l'ordre %s.",
                record.get("order_id"),
            )

            return exc

        return None

    def _connector_snapshot(
        self,
        confirmed: bool,
    ) -> Dict[str, Any]:
        """Dernier état publié par le connecteur, sans jamais lever."""

        snapshot: Dict[str, Any] = {
            "confirmed": bool(confirmed),
        }

        last_execution = getattr(
            self.connector,
            "last_execution",
            None,
        )

        if isinstance(last_execution, dict):
            snapshot["last_execution"] = last_execution

        mode = getattr(self.connector, "mode", None)

        if mode is not None:
            snapshot["mode"] = mode

        return snapshot

    # =========================================================
    # SOUMISSION INTERNE
    # =========================================================

    def submit_order(
        self,
        asset: Any,
        direction: Any,
        amount: Any,
        payout: Any = None,
        expiration_seconds: Any = None,
        timeframe: Any = None,
    ) -> Dict[str, Any]:
        """
        Pipeline complet d'un ordre.

            1. Normalisation et horodatage
            2. Détection des doublons
            3. Validation des risques
            4. Persistance préalable (status SUBMITTED)
            5. Exécution via le connecteur
            6. Persistance du résultat (ACCEPTED ou FAILED)

        `timeframe` est accepté comme synonyme de `expiration_seconds`
        pour rester compatible avec les appels historiques.
        """

        timestamp = datetime.now(timezone.utc).isoformat()

        # -----------------------------------------------------
        # 1. NORMALISATION
        # -----------------------------------------------------

        if expiration_seconds is None:
            expiration_seconds = timeframe

        if expiration_seconds is None:
            expiration_seconds = self.min_expiration_seconds

        spec: Dict[str, Any] = {
            "order_id": str(uuid.uuid4()),
            "asset": _normalize_asset(asset),
            "direction": _normalize_text(direction),
            "amount": _to_float(amount),
            "payout": _to_float(payout),
            "expiration_seconds": _to_int(
                expiration_seconds
            ),
            "timestamp": timestamp,
        }

        # -----------------------------------------------------
        # 2. DOUBLONS
        # -----------------------------------------------------

        if self.is_duplicate(
            spec["asset"],
            spec["direction"],
        ):

            return self._reject(
                spec,
                "Ordre en doublon détecté dans "
                "l'intervalle de sécurité.",
            )

        # -----------------------------------------------------
        # 3. VALIDATION DES RISQUES
        # -----------------------------------------------------

        is_valid, error_msg = self.validate_order(spec)

        if not is_valid:
            return self._reject(
                spec,
                error_msg or "REJET : règle de risque non satisfaite.",
            )

        # -----------------------------------------------------
        # 4. PERSISTANCE AVANT EXÉCUTION
        # -----------------------------------------------------

        submitted = {
            "status": STATUS_SUBMITTED,
            "order_id": spec["order_id"],
            "reason": None,
            "timestamp": spec["timestamp"],
            "order": dict(spec),
            "connector_response": {
                "stage": "pre_execution",
                "payout": spec["payout"],
                "expiration_seconds": spec[
                    "expiration_seconds"
                ],
                "mode": getattr(
                    self.connector,
                    "mode",
                    None,
                ),
            },
        }

        persist_error = self._persist(
            submitted,
            spec,
        )

        if persist_error is not None:

            logger.error(
                "Ordre %s non envoyé : persistance préalable impossible.",
                spec["order_id"],
            )

            return {
                "status": STATUS_FAILED,
                "order_id": spec["order_id"],
                "reason": (
                    "Persistance préalable impossible, "
                    f"ordre non envoyé : {persist_error}"
                ),
                "timestamp": spec["timestamp"],
                "order": dict(spec),
                "connector_response": None,
                "executed": False,
            }

        # -----------------------------------------------------
        # 5. EXÉCUTION
        # -----------------------------------------------------

        order = Order(
            order_id=spec["order_id"],
            asset=spec["asset"],
            direction=spec["direction"],
            amount=spec["amount"],
            payout=spec["payout"],
            expiration_seconds=spec[
                "expiration_seconds"
            ],
        )

        try:

            result = self.connector.execute_order(order)

        except Exception as exc:

            logger.exception(
                "Erreur connecteur sur l'ordre %s.",
                spec["order_id"],
            )

            return self._finalize(
                spec,
                STATUS_FAILED,
                f"Erreur connecteur : {exc}",
                self._connector_snapshot(False),
            )

        if not result:

            return self._finalize(
                spec,
                STATUS_FAILED,
                "Le connecteur n'a pas confirmé l'ordre.",
                self._connector_snapshot(False),
            )

        # -----------------------------------------------------
        # 6. RÉSULTAT
        # -----------------------------------------------------

        response = self._finalize(
            spec,
            STATUS_ACCEPTED,
            None,
            self._connector_snapshot(True),
        )

        self.order_history.append(
            {
                "order_id": spec["order_id"],
                "asset": spec["asset"],
                "direction": spec["direction"],
                "amount": spec["amount"],
                "payout": spec["payout"],
                "expiration_seconds": spec[
                    "expiration_seconds"
                ],
                "timestamp": spec["timestamp"],
                "result": response.get(
                    "connector_response"
                ),
            }
        )

        return response

    # =========================================================
    # RÉPONSES
    # =========================================================

    def _reject(
        self,
        spec: Dict[str, Any],
        reason: str,
    ) -> Dict[str, Any]:
        """Refus avant toute tentative d'exécution."""

        logger.warning(
            "Ordre refusé | %s %s | %s",
            spec.get("direction"),
            spec.get("asset"),
            reason,
        )

        response = {
            "status": STATUS_REJECTED,
            "order_id": spec.get("order_id"),
            "reason": reason,
            "timestamp": spec.get("timestamp"),
            "order": dict(spec),
            "connector_response": None,
            "executed": False,
        }

        self._persist(response, spec)

        return response

    def _finalize(
        self,
        spec: Dict[str, Any],
        status: str,
        reason: Optional[str],
        connector_response: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Réponse après une tentative d'exécution réelle."""

        response = {
            "status": status,
            "order_id": spec["order_id"],
            "reason": reason,
            "timestamp": spec["timestamp"],
            "order": dict(spec),
            "connector_response": connector_response,
            "executed": True,
        }

        self._persist(response, spec)

        return response

    # =========================================================
    # INTERFACE PUBLIQUE DICT
    # =========================================================

    def execute_order(
        self,
        order: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Interface officielle du flux Pocket AI Trader.

        Format attendu :

            {
                "asset": "EURUSD",
                "direction": "BUY",
                "amount": 10.0,
                "payout": 85.0,
                "expiration_seconds": 60
            }

        `timeframe` ou `expiration` sont acceptés à la place de
        `expiration_seconds`.
        """

        if not isinstance(order, dict):

            spec = {
                "order_id": str(uuid.uuid4()),
                "asset": "",
                "direction": "",
                "amount": None,
                "payout": None,
                "expiration_seconds": None,
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            return self._reject(
                spec,
                "REJET : order doit être un dictionnaire.",
            )

        asset = order.get("asset")
        direction = order.get("direction")
        amount = order.get("amount")

        missing = [
            name
            for name, value in (
                ("asset", asset),
                ("direction", direction),
                ("amount", amount),
            )
            if value is None or value == ""
        ]

        if missing:

            spec = {
                "order_id": str(uuid.uuid4()),
                "asset": _normalize_asset(asset),
                "direction": _normalize_text(direction),
                "amount": _to_float(amount),
                "payout": _to_float(order.get("payout")),
                "expiration_seconds": _to_int(
                    order.get(
                        "expiration_seconds",
                        order.get(
                            "timeframe",
                            order.get("expiration"),
                        ),
                    )
                ),
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            return self._reject(
                spec,
                "REJET : champs obligatoires manquants "
                f"({', '.join(missing)}).",
            )

        expiration = order.get(
            "expiration_seconds",
            order.get(
                "timeframe",
                order.get("expiration"),
            ),
        )

        return self.submit_order(
            asset=asset,
            direction=direction,
            amount=amount,
            payout=order.get(
                "payout",
                order.get("payout_percent"),
            ),
            expiration_seconds=expiration,
        )

    # =========================================================
    # ÉTAT
    # =========================================================

    def get_status(self) -> Dict[str, Any]:
        """Vue non sensible des garde-fous actifs."""

        try:
            daily_loss: Optional[float] = self.get_daily_loss()

        except Exception:
            daily_loss = None

        return {
            "manager": "OrderManager",
            "emergency_stop_triggered": (
                self.emergency_stop_triggered
            ),
            "emergency_stop_reason": (
                self.emergency_stop_reason
            ),
            "max_position_size": self.max_position_size,
            "max_daily_loss": self.max_daily_loss,
            "min_payout_percent": self.min_payout_percent,
            "min_expiration_seconds": (
                self.min_expiration_seconds
            ),
            "duplicate_window_seconds": (
                self.duplicate_window_seconds
            ),
            "break_even_win_rate_at_min_payout": round(
                break_even_win_rate(
                    self.min_payout_percent
                ),
                2,
            ),
            "daily_loss": daily_loss,
            "accepted_orders_in_memory": len(
                self.order_history
            ),
            "data_store_attached": (
                self.data_store is not None
            ),
            "connector_mode": getattr(
                self.connector,
                "mode",
                None,
            ),
            "persistence_before_execution": True,
        }

    def health_check(self) -> Dict[str, Any]:
        """Alias de get_status(), pour rester aligné sur les autres modules."""

        return self.get_status()
