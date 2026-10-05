"""
order_manager.py

Gestion sécurisée des ordres pour Pocket AI Trader.

Responsabilités :
- Validation des ordres
- Contrôle du risque
- Contrôle du payout
- Contrôle de l'expiration
- Détection des doublons
- Arrêt d'urgence
- Exécution via connector.execute_order(order)
- Persistance via DataStore
- Compatible avec po_connector.py
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid


# ============================================================
# ORDER MODEL
# ============================================================

@dataclass
class Order:
    """
    Objet d'ordre transmis au PocketOptionConnector.
    """

    order_id: str
    asset: str
    direction: str
    amount: float
    payout: float
    expiration_seconds: int


# ============================================================
# ORDER MANAGER
# ============================================================

class OrderManager:
    """
    Gestionnaire central des ordres.

    Le connecteur doit exposer :

        execute_order(order) -> bool
    """

    def __init__(
        self,
        connector,
        data_store=None,
        max_position_size: float = 50.0,
        max_daily_loss: float = 100.0,
        duplicate_window_seconds: int = 10,
        min_payout: float = 80.0,
        max_payout: float = 95.0,
        min_expiration_seconds: int = 60,
    ):
        self.connector = connector
        self.data_store = data_store

        # Limites de sécurité
        self.max_position_size = float(max_position_size)
        self.max_daily_loss = float(max_daily_loss)

        # Protection contre les doublons
        self.duplicate_window_seconds = int(
            duplicate_window_seconds
        )

        # Contraintes Pocket Option
        self.min_payout = float(min_payout)
        self.max_payout = float(max_payout)
        self.min_expiration_seconds = int(
            min_expiration_seconds
        )

        # Sécurité
        self.emergency_stop_triggered = False

        # Historique local des ordres
        self._order_history: List[Dict[str, Any]] = []

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

    def reset_emergency_stop(self) -> None:
        """
        Désactive l'arrêt d'urgence.
        """
        self.emergency_stop_triggered = False

    # ========================================================
    # DAILY LOSS
    # ========================================================

    def get_daily_loss(self) -> float:
        """
        Retourne la perte réalisée du jour.

        Si aucun DataStore n'est disponible,
        aucune perte n'est considérée.
        """

        if self.data_store is None:
            return 0.0

        try:
            today = datetime.now(timezone.utc).date()

            stats = self.data_store.get_daily_stats(today)

            realized_pnl = float(
                stats.get("realized_pnl", 0.0)
            )

            # Une perte est représentée par un PnL négatif.
            if realized_pnl < 0:
                return abs(realized_pnl)

            return 0.0

        except Exception:
            # En cas de problème avec le stockage,
            # on ne bloque pas arbitrairement l'ordre ici.
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
        Vérifie si un ordre identique vient d'être envoyé.
        """

        direction = direction.upper()

        for previous_order in reversed(self._order_history):

            timestamp = previous_order.get("timestamp")

            if timestamp is None:
                continue

            time_diff = (
                now - timestamp
            ).total_seconds()

            # Historique plus ancien que la fenêtre :
            # inutile de continuer.
            if time_diff > self.duplicate_window_seconds:
                break

            if (
                previous_order.get("asset") == asset
                and previous_order.get("direction") == direction
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
            order_data.get("direction", "")
        ).upper()

        if direction not in {"BUY", "SELL"}:
            return (
                False,
                f"REJECTED: Direction invalide: {direction}. "
                "Valeurs autorisées: BUY ou SELL.",
            )

        # ----------------------------------------------------
        # ASSET
        # ----------------------------------------------------

        asset = str(
            order_data.get("asset", "")
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
                order_data.get("amount", 0.0)
            )
        except (TypeError, ValueError):
            return (
                False,
                "REJECTED: Montant invalide.",
            )

        if amount <= 0:
            return (
                False,
                "REJECTED: Le montant doit être supérieur à zéro.",
            )

        if amount > self.max_position_size:
            return (
                False,
                (
                    f"REJECTED: Montant {amount} supérieur "
                    f"au maximum autorisé "
                    f"({self.max_position_size})."
                ),
            )

        # ----------------------------------------------------
        # PAYOUT
        # ----------------------------------------------------

        try:
            payout = float(
                order_data.get("payout", 0.0)
            )
        except (TypeError, ValueError):
            return (
                False,
                "REJECTED: Payout invalide.",
            )

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
        except (TypeError, ValueError):
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
                    f"{self.min_expiration_seconds} secondes."
                ),
            )

        # ----------------------------------------------------
        # DAILY LOSS
        # ----------------------------------------------------

        current_loss = self.get_daily_loss()

        if current_loss >= self.max_daily_loss:
            return (
                False,
                (
                    "REJECTED: Limite de perte journalière "
                    f"atteinte ({self.max_daily_loss})."
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
        timeframe: int = 60,
        payout: float = 85.0,
        expiration_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Prépare, valide et exécute un ordre.

        IMPORTANT :
        Le connecteur est appelé avec :

            connector.execute_order(order)
        """

        now = datetime.now(timezone.utc)

        direction = str(
            direction
        ).upper()

        asset = str(asset).strip()

        # ----------------------------------------------------
        # EXPIRATION
        # ----------------------------------------------------

        if expiration_seconds is None:
            expiration = int(timeframe)
        else:
            expiration = int(expiration_seconds)

        # ----------------------------------------------------
        # ORDRE À VALIDER
        # ----------------------------------------------------

        order_spec = {
            "asset": asset,
            "direction": direction,
            "amount": float(amount),
            "timeframe": int(timeframe),
            "payout": float(payout),
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

        is_valid, error_message = self.validate_order(
            order_spec
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
        # CREATION DE L'OBJET ORDER
        # ----------------------------------------------------

        order = Order(
            order_id=order_id,
            asset=asset,
            direction=direction,
            amount=float(amount),
            payout=float(payout),
            expiration_seconds=expiration,
        )

        # ----------------------------------------------------
        # EXECUTION
        # ----------------------------------------------------

        try:

            # Interface officielle attendue par po_connector.py
            confirmed = bool(
                self.connector.execute_order(order)
            )

        except Exception as exc:

            response = {
                "status": "FAILED",
                "reason": str(exc),
                "order_id": order_id,
                "timestamp": now.isoformat(),
                **order_spec,
            }

            self._save_order(response)

            return response

        # ----------------------------------------------------
        # EXECUTION NON CONFIRMÉE
        # ----------------------------------------------------

        if not confirmed:

            response = {
                "status": "FAILED",
                "reason": (
                    "Le connecteur n'a pas confirmé "
                    "l'exécution de l'ordre."
                ),
                "order_id": order_id,
                "timestamp": now.isoformat(),
                **order_spec,
            }

            self._save_order(response)

            return response

        # ----------------------------------------------------
        # HISTORIQUE
        # ----------------------------------------------------

        history_record = {
            "order_id": order_id,
            "asset": asset,
            "direction": direction,
            "amount": float(amount),
            "timeframe": int(timeframe),
            "payout": float(payout),
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
            "connector_response": getattr(
                self.connector,
                "last_execution",
                {
                    "confirmed": True,
                },
            ),
            "timestamp": now.isoformat(),
            **order_spec,
        }

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
        Construit une réponse standardisée pour
        les ordres rejetés.
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
        Sauvegarde l'ordre si un DataStore est disponible.
        """

        if self.data_store is None:
            return

        try:
            self.data_store.save_order(
                order
            )
        except Exception:
            # La sauvegarde ne doit pas provoquer
            # une seconde erreur dans le flux principal.
            pass
