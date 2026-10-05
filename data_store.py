"""
order_manager.py
Pocket AI Trader

Gestion sécurisée et centralisée des ordres.

Responsabilités
---------------
- Validation complète des ordres
- Normalisation des paramètres
- Contrôle du montant
- Contrôle du payout
- Contrôle de l'expiration
- Protection contre les doublons
- Limite de perte journalière
- Arrêt d'urgence
- Création d'un Order standardisé
- Exécution via connector.execute_order(order)
- Persistance via DataStore
- Compatibilité avec po_connector.py

Architecture
------------
server.py
    ↓
OrderManager
    ↓
PocketOptionConnector
    ↓
Pocket Option

Mode de sécurité
----------------
DEMO par défaut.

Aucun ordre LIVE ne doit être envoyé sans
activation explicite du connecteur.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid


# ============================================================
# CONSTANTES
# ============================================================

DEFAULT_PAYOUT = 85.0

MIN_PAYOUT = 80.0
MAX_PAYOUT = 92.0

MIN_EXPIRATION_SECONDS = 60

DEFAULT_TIMEFRAME = 60

DEFAULT_MAX_POSITION_SIZE = 50.0

DEFAULT_MAX_DAILY_LOSS = 100.0

DEFAULT_DUPLICATE_WINDOW_SECONDS = 10


# ============================================================
# ORDER MODEL
# ============================================================

@dataclass
class Order:
    """
    Modèle standardisé d'un ordre.

    Cet objet est directement transmis à :

        connector.execute_order(order)
    """

    order_id: str
    asset: str
    direction: str
    amount: float
    payout: float
    expiration_seconds: int
    timeframe: int = DEFAULT_TIMEFRAME


# ============================================================
# ORDER MANAGER
# ============================================================

class OrderManager:
    """
    Gestionnaire central des ordres.

    Le connecteur doit exposer :

        execute_order(order) -> bool

    DEMO :
        l'ordre est simulé par po_connector.py.

    LIVE :
        l'exécution reste verrouillée par le connecteur
        tant que les conditions de sécurité ne sont pas
        explicitement activées.
    """

    def __init__(
        self,
        connector,
        data_store=None,
        max_position_size: float = DEFAULT_MAX_POSITION_SIZE,
        max_daily_loss: float = DEFAULT_MAX_DAILY_LOSS,
        duplicate_window_seconds: int = DEFAULT_DUPLICATE_WINDOW_SECONDS,
        min_payout: float = MIN_PAYOUT,
        max_payout: float = MAX_PAYOUT,
        min_expiration_seconds: int = MIN_EXPIRATION_SECONDS,
    ):
        self.connector = connector
        self.data_store = data_store

        # ----------------------------------------------------
        # LIMITES DE SÉCURITÉ
        # ----------------------------------------------------

        self.max_position_size = float(
            max_position_size
        )

        self.max_daily_loss = float(
            max_daily_loss
        )

        # ----------------------------------------------------
        # PROTECTION CONTRE LES DOUBLONS
        # ----------------------------------------------------

        self.duplicate_window_seconds = int(
            duplicate_window_seconds
        )

        # ----------------------------------------------------
        # CONTRAINTES POCKET OPTION
        # ----------------------------------------------------

        self.min_payout = float(
            min_payout
        )

        self.max_payout = float(
            max_payout
        )

        self.min_expiration_seconds = int(
            min_expiration_seconds
        )

        # ----------------------------------------------------
        # SÉCURITÉ
        # ----------------------------------------------------

        self.emergency_stop_triggered = False

        # ----------------------------------------------------
        # HISTORIQUE LOCAL
        # ----------------------------------------------------

        self._order_history: List[
            Dict[str, Any]
        ] = []

    # ========================================================
    # PUBLIC EXECUTION INTERFACE
    # ========================================================

    def execute_order(
        self,
        order,
    ) -> Dict[str, Any]:
        """
        Interface publique principale.

        Cette méthode accepte :

        1. un dictionnaire
        2. un objet Order
        3. un objet compatible possédant les attributs
           nécessaires.

        Exemple :

            order = {
                "asset": "EURUSD",
                "direction": "BUY",
                "amount": 10.0,
                "payout": 0.80,
                "expiration": 60,
            }

            result = manager.execute_order(order)

        La méthode convertit ensuite l'ordre vers la
        pipeline interne submit_order().
        """

        # ----------------------------------------------------
        # CONVERSION DE L'ENTRÉE
        # ----------------------------------------------------

        if isinstance(order, dict):

            raw_order = dict(order)

        else:

            try:
                raw_order = vars(order).copy()

            except TypeError:

                return {
                    "status": "REJECTED",
                    "reason": (
                        "Format d'ordre invalide. "
                        "Un dictionnaire ou un objet "
                        "compatible est requis."
                    ),
                    "order_id": None,
                }

        # ----------------------------------------------------
        # RÉCUPÉRATION DES CHAMPS
        # ----------------------------------------------------

        asset = raw_order.get(
            "asset"
        )

        direction = raw_order.get(
            "direction"
        )

        amount = raw_order.get(
            "amount"
        )

        timeframe = raw_order.get(
            "timeframe",
            raw_order.get(
                "expiration",
                DEFAULT_TIMEFRAME,
            ),
        )

        payout = raw_order.get(
            "payout",
            DEFAULT_PAYOUT,
        )

        expiration = raw_order.get(
            "expiration_seconds",
            raw_order.get(
                "expiration",
                timeframe,
            ),
        )

        # ----------------------------------------------------
        # VALIDATION DES CHAMPS OBLIGATOIRES
        # ----------------------------------------------------

        missing = []

        if asset is None:
            missing.append("asset")

        if direction is None:
            missing.append("direction")

        if amount is None:
            missing.append("amount")

        if missing:

            return {
                "status": "REJECTED",
                "reason": (
                    "Champs obligatoires manquants: "
                    + ", ".join(missing)
                ),
                "order_id": raw_order.get(
                    "order_id"
                ),
                "order": raw_order,
            }

        # ----------------------------------------------------
        # NORMALISATION PAYOUT
        # ----------------------------------------------------
        #
        # Le système accepte :
        #
        #   80
        #   85
        #   92
        #
        # ou :
        #
        #   0.80
        #   0.85
        #   0.92
        #
        # En interne :
        #
        #   80.0
        #   85.0
        #   92.0
        # ----------------------------------------------------

        try:

            payout_value = float(
                payout
            )

        except (
            TypeError,
            ValueError,
        ):

            return {
                "status": "REJECTED",
                "reason": "Payout invalide.",
                "order_id": raw_order.get(
                    "order_id"
                ),
                "order": raw_order,
            }

        if 0 < payout_value <= 1:

            payout_value *= 100.0

        # ----------------------------------------------------
        # NORMALISATION EXPIRATION
        # ----------------------------------------------------

        try:

            expiration_value = int(
                expiration
            )

        except (
            TypeError,
            ValueError,
        ):

            return {
                "status": "REJECTED",
                "reason": "Expiration invalide.",
                "order_id": raw_order.get(
                    "order_id"
                ),
                "order": raw_order,
            }

        # ----------------------------------------------------
        # NORMALISATION TIMEFRAME
        # ----------------------------------------------------

        try:

            timeframe_value = int(
                timeframe
            )

        except (
            TypeError,
            ValueError,
        ):

            timeframe_value = expiration_value

        # ----------------------------------------------------
        # APPEL DE LA PIPELINE PRINCIPALE
        # ----------------------------------------------------

        try:

            return self.submit_order(
                asset=str(
                    asset
                ).strip(),

                direction=str(
                    direction
                ).upper().strip(),

                amount=float(
                    amount
                ),

                timeframe=timeframe_value,

                payout=payout_value,

                expiration_seconds=expiration_value,
            )

        except Exception as exc:

            return {
                "status": "ERROR",
                "reason": str(exc),
                "order_id": raw_order.get(
                    "order_id"
                ),
                "order": raw_order,
            }

    # ========================================================
    # EMERGENCY STOP
    # ========================================================

    def trigger_emergency_stop(
        self,
        reason: str = "Arrêt manuel d'urgence",
    ) -> None:
        """
        Active l'arrêt d'urgence.
        """

        self.emergency_stop_triggered = True

    def reset_emergency_stop(
        self,
    ) -> None:
        """
        Désactive l'arrêt d'urgence.
        """

        self.emergency_stop_triggered = False

    # ========================================================
    # DAILY LOSS
    # ========================================================

    def get_daily_loss(
        self,
    ) -> float:
        """
        Retourne la perte réalisée du jour.

        Si aucun DataStore n'est disponible :
            0.0

        Si le PnL journalier est négatif :
            retourne sa valeur absolue.
        """

        if self.data_store is None:
            return 0.0

        try:

            today = datetime.now(
                timezone.utc
            ).date()

            stats = (
                self.data_store
                .get_daily_stats(today)
            )

            realized_pnl = float(
                stats.get(
                    "realized_pnl",
                    0.0,
                )
            )

            if realized_pnl < 0:
                return abs(
                    realized_pnl
                )

            return 0.0

        except Exception:

            return 0.0

    # ========================================================
    # DUPLICATE PROTECTION
    # ========================================================

    def is_duplicate(
        self,
        asset: str,
        direction: str,
        now: datetime,
    ) -> bool:
        """
        Vérifie si un ordre identique vient
        d'être envoyé récemment.
        """

        direction = str(
            direction
        ).upper()

        for previous_order in reversed(
            self._order_history
        ):

            timestamp = (
                previous_order.get(
                    "timestamp"
                )
            )

            if timestamp is None:
                continue

            time_diff = (
                now - timestamp
            ).total_seconds()

            if time_diff < 0:
                continue

            if (
                time_diff
                > self.duplicate_window_seconds
            ):
                break

            if (
                previous_order.get(
                    "asset"
                ) == asset
                and
                previous_order.get(
                    "direction"
                ) == direction
            ):
                return True

        return False

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate_order(
        self,
        order_data: Dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        """
        Valide complètement un ordre avant exécution.
        """

        # ----------------------------------------------------
        # EMERGENCY STOP
        # ----------------------------------------------------

        if self.emergency_stop_triggered:

            return (
                False,
                "REJECTED: Arrêt d'urgence actif.",
            )

        # ----------------------------------------------------
        # DIRECTION
        # ----------------------------------------------------

        direction = str(
            order_data.get(
                "direction",
                "",
            )
        ).upper()

        if direction not in {
            "BUY",
            "SELL",
        }:

            return (
                False,
                (
                    "REJECTED: Direction invalide: "
                    f"{direction}. "
                    "Valeurs autorisées: BUY ou SELL."
                ),
            )

        # ----------------------------------------------------
        # ASSET
        # ----------------------------------------------------

        asset = str(
            order_data.get(
                "asset",
                "",
            )
        ).strip()

        if not asset:

            return (
                False,
                "REJECTED: Asset manquant.",
            )

        # ----------------------------------------------------
        # AMOUNT
        # ----------------------------------------------------

        try:

            amount = float(
                order_data.get(
                    "amount",
                    0.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            return (
                False,
                "REJECTED: Montant invalide.",
            )

        if amount <= 0:

            return (
                False,
                (
                    "REJECTED: Le montant "
                    "doit être supérieur à zéro."
                ),
            )

        if amount > self.max_position_size:

            return (
                False,
                (
                    f"REJECTED: Montant {amount} "
                    f"supérieur au maximum autorisé "
                    f"({self.max_position_size})."
                ),
            )

        # ----------------------------------------------------
        # PAYOUT
        # ----------------------------------------------------

        try:

            payout = float(
                order_data.get(
                    "payout",
                    0.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            return (
                False,
                "REJECTED: Payout invalide.",
            )

        # Normalisation supplémentaire
        if 0 < payout <= 1:
            payout *= 100.0

        if not (
            self.min_payout
            <= payout
            <= self.max_payout
        ):

            return (
                False,
                (
                    f"REJECTED: Payout {payout}% "
                    f"hors plage autorisée "
                    f"({self.min_payout}%–"
                    f"{self.max_payout}%)."
                ),
            )

        # ----------------------------------------------------
        # EXPIRATION
        # ----------------------------------------------------

        try:

            expiration_seconds = int(
                order_data.get(
                    "expiration_seconds",
                    0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            return (
                False,
                "REJECTED: Expiration invalide.",
            )

        if (
            expiration_seconds
            < self.min_expiration_seconds
        ):

            return (
                False,
                (
                    "REJECTED: Expiration minimale: "
                    f"{self.min_expiration_seconds} "
                    "secondes."
                ),
            )

        # ----------------------------------------------------
        # DAILY LOSS
        # ----------------------------------------------------

        current_loss = (
            self.get_daily_loss()
        )

        if (
            current_loss
            >= self.max_daily_loss
        ):

            return (
                False,
                (
                    "REJECTED: Limite de perte "
                    "journalière atteinte "
                    f"({self.max_daily_loss})."
                ),
            )

        return True, None

    # ========================================================
    # SUBMIT ORDER
    # ========================================================

    def submit_order(
        self,
        asset: str,
        direction: str,
        amount: float,
        timeframe: int = DEFAULT_TIMEFRAME,
        payout: float = DEFAULT_PAYOUT,
        expiration_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Prépare, valide et exécute un ordre.

        Pipeline :

            normalisation
                ↓
            duplicate protection
                ↓
            validation
                ↓
            Order creation
                ↓
            connector.execute_order(order)
                ↓
            historique
                ↓
            DataStore
        """

        now = datetime.now(
            timezone.utc
        )

        # ----------------------------------------------------
        # NORMALISATION
        # ----------------------------------------------------

        asset = str(
            asset
        ).strip()

        direction = str(
            direction
        ).upper().strip()

        amount = float(
            amount
        )

        timeframe = int(
            timeframe
        )

        payout = float(
            payout
        )

        # Payout sous forme 0.80 → 80.0
        if 0 < payout <= 1:
            payout *= 100.0

        # ----------------------------------------------------
        # EXPIRATION
        # ----------------------------------------------------

        if expiration_seconds is None:

            expiration = timeframe

        else:

            expiration = int(
                expiration_seconds
            )

        # ----------------------------------------------------
        # ORDER SPEC
        # ----------------------------------------------------

        order_spec = {
            "asset": asset,
            "direction": direction,
            "amount": amount,
            "timeframe": timeframe,
            "payout": payout,
            "expiration_seconds": expiration,
        }
        
        # ----------------------------------------------------
        # DUPLICATE CHECK
        # ----------------------------------------------------

        if self.is_duplicate(
            asset,
            direction,
            now,
        ):

            return self._rejected_response(
                now=now,
                order_spec=order_spec,
                reason=(
                    "Ordre en doublon détecté "
                    "dans l'intervalle de sécurité."
                ),
            )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        is_valid, error_message = (
            self.validate_order(
                order_spec
            )
        )

        if not is_valid:

            return self._rejected_response(
                now=now,
                order_spec=order_spec,
                reason=error_message,
            )

        # ----------------------------------------------------
        # ORDER ID
        # ----------------------------------------------------

        order_id = str(
            uuid.uuid4()
        )

        # ----------------------------------------------------
        # ORDER OBJECT
        # ----------------------------------------------------

        order = Order(
            order_id=order_id,
            asset=asset,
            direction=direction,
            amount=amount,
            payout=payout,
            expiration_seconds=expiration,
            timeframe=timeframe,
        )

        # ----------------------------------------------------
        # CONNECTOR CHECK
        # ----------------------------------------------------

        if self.connector is None:

            response = {
                "status": "FAILED",
                "reason": (
                    "Aucun connecteur de trading "
                    "n'est configuré."
                ),
                "order_id": order_id,
                "timestamp": now.isoformat(),
                **order_spec,
            }

            self._save_order(
                response
            )

            return response

        # ----------------------------------------------------
        # EXECUTION VIA CONNECTOR
        # ----------------------------------------------------

        try:

            confirmed = (
                self.connector.execute_order(
                    order
                )
            )

        except Exception as exc:

            response = {
                "status": "FAILED",
                "reason": str(exc),
                "order_id": order_id,
                "timestamp": now.isoformat(),
                **order_spec,
            }

            self._save_order(
                response
            )

            return response

        # ----------------------------------------------------
        # NORMALISATION DU RETOUR CONNECTEUR
        # ----------------------------------------------------

        if isinstance(
            confirmed,
            dict,
        ):

            connector_confirmed = bool(
                confirmed.get(
                    "confirmed",
                    confirmed.get(
                        "success",
                        confirmed.get(
                            "executed",
                            False,
                        ),
                    ),
                )
            )

            connector_response = confirmed

        else:

            connector_confirmed = bool(
                confirmed
            )

            connector_response = (
                getattr(
                    self.connector,
                    "last_execution",
                    None,
                )
            )

            if connector_response is None:

                connector_response = {
                    "confirmed": connector_confirmed,
                }

        # ----------------------------------------------------
        # EXECUTION NON CONFIRMÉE
        # ----------------------------------------------------

        if not connector_confirmed:

            response = {
                "status": "FAILED",
                "reason": (
                    "Le connecteur n'a pas confirmé "
                    "l'exécution de l'ordre."
                ),
                "order_id": order_id,
                "timestamp": now.isoformat(),
                "connector_response": (
                    connector_response
                ),
                **order_spec,
            }

            self._save_order(
                response
            )

            return response

        # ----------------------------------------------------
        # HISTORIQUE
        # ----------------------------------------------------

        history_record = {
            "order_id": order_id,
            "asset": asset,
            "direction": direction,
            "amount": amount,
            "timeframe": timeframe,
            "payout": payout,
            "expiration_seconds": expiration,
            "timestamp": now,
        }

        self._order_history.append(
            history_record
        )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        full_response = {
            "status": "ACCEPTED",
            "order_id": order_id,
            "connector_response": (
                connector_response
            ),
            "timestamp": now.isoformat(),
            **order_spec,
        }

        # ----------------------------------------------------
        # PERSISTENCE
        # ----------------------------------------------------

        self._save_order(
            full_response
        )

        return full_response

    # ========================================================
    # REJECTED RESPONSE
    # ========================================================

    def _rejected_response(
        self,
        now: datetime,
        order_spec: Dict[str, Any],
        reason: Optional[str],
    ) -> Dict[str, Any]:
        """
        Construit une réponse standardisée
        pour un ordre rejeté.
        """

        response = {
            "status": "REJECTED",
            "reason": reason,
            "order_id": None,
            "timestamp": now.isoformat(),
            **order_spec,
        }

        self._save_order(
            response
        )

        return response

    # ========================================================
    # DATA STORE
    # ========================================================

    def _save_order(
        self,
        order: Dict[str, Any],
    ) -> None:
        """
        Sauvegarde l'ordre si DataStore disponible.

        Une erreur de persistance ne doit pas provoquer
        une seconde erreur dans le flux principal.
        """

        if self.data_store is None:
            return

        try:

            self.data_store.save_order(
                order
            )

        except Exception:

            pass


# ============================================================
# LOCAL SELF TEST
# ============================================================

if __name__ == "__main__":

    from types import SimpleNamespace

    try:

        from po_connector import (
            PocketOptionConnector,
        )

        connector = (
            PocketOptionConnector(
                mode="demo",
                trading_enabled=False,
                live_enabled=False,
            )
        )

        manager = OrderManager(
            connector=connector
        )

        test_order = {
            "order_id": "TEST-INTEGRATION-001",
            "asset": "EURUSD",
            "direction": "BUY",
            "amount": 10.0,
            "payout": 0.80,
            "expiration": 60,
            "timeframe": 60,
        }

        print(
            "===================================="
        )

        print(
            "POCKET AI TRADER"
        )

        print(
            "ORDER MANAGER SELF TEST"
        )

        print(
            "===================================="
        )

        print(
            "\nConnector status:"
        )

        print(
            connector.get_status()
        )

        print(
            "\nTest order:"
        )

        print(
            test_order
        )

        print(
            "\nExecution:"
        )

        result = manager.execute_order(
            test_order
        )

        print(
            result
        )

        print(
            "\nLast connector execution:"
        )

        print(
            connector.last_execution
        )

    except Exception as exc:

        print(
            "SELF TEST ERROR:"
        )

        print(
            repr(exc)
)
        # -----
