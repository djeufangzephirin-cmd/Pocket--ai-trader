"""
order_manager.py - Gestionnaire d'ordres et de risques pour pocket-ai-trader.
Assure la validation, l'idempotence, le contrôle des limites et le routage des ordres.
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import uuid

class OrderManager:
    def __init__(
        self,
        connector,
        max_position_size: float = 50.0,
        max_daily_loss: float = 100.0,
        duplicate_window_seconds: int = 10,
    ):
        self.connector = connector
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.duplicate_window_seconds = duplicate_window_seconds

        self.emergency_stop_triggered: bool = False
        self._order_history: List[Dict[str, Any]] = []
        self._daily_realized_loss: float = 0.0

    def trigger_emergency_stop(self, reason: str = "Arrêt manuel d'urgence") -> None:
        """Bloque immédiatement l'envoi de nouveaux ordres."""
        self.emergency_stop_triggered = True

    def reset_emergency_stop(self) -> None:
        """Réactive le trading après vérification manuelle."""
        self.emergency_stop_triggered = False

    def is_duplicate(self, asset: str, direction: str, now: datetime) -> bool:
        """
        Vérifie si un ordre identique a déjà été soumis dans la fenêtre temporelle définie.
        """
        for order in reversed(self._order_history):
            time_diff = (now - order["timestamp"]).total_seconds()
            if time_diff > self.duplicate_window_seconds:
                break
            if order["asset"] == asset and order["direction"] == direction:
                return True
        return False

    def validate_order(self, order_data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Contrôle les règles de sécurité avant exécution.
        """
        if self.emergency_stop_triggered:
            return False, "REJET : Arrêt d'urgence actif."

        amount = order_data.get("amount", 0.0)
        if amount <= 0:
            return False, "REJET : Le montant doit être supérieur à zéro."

        if amount > self.max_position_size:
            return False, f"REJET : Montant {amount} supérieur au max autorisé ({self.max_position_size})."

        if self._daily_realized_loss >= self.max_daily_loss:
            return False, f"REJET : Perte journalière maximale atteinte ({self.max_daily_loss})."

        return True, None

    def submit_order(self, asset: str, direction: str, amount: float, timeframe: int = 60) -> Dict[str, Any]:
        """
        Pipeline complet d'exécution d'un ordre :
        1. Horodatage
        2. Vérification anti-doublon (AVANT tout enregistrement)
        3. Contrôle des limites de risque
        4. Exécution via le connecteur
        5. Enregistrement dans l'historique
        """
        now = datetime.now(timezone.utc)

        # 1. Détection des doublons (correctif appliqué ici)
        if self.is_duplicate(asset, direction, now):
            return {
                "status": "REJECTED",
                "reason": "Ordre en doublon détecté dans l'intervalle de sécurité",
                "order_id": None,
                "timestamp": now.isoformat(),
            }

        # 2. Validation des règles de gestion des risques
        order_spec = {"asset": asset, "direction": direction, "amount": amount, "timeframe": timeframe}
        is_valid, error_msg = self.validate_order(order_spec)
        if not is_valid:
            return {
                "status": "REJECTED",
                "reason": error_msg,
                "order_id": None,
                "timestamp": now.isoformat(),
            }

        # 3. Exécution via le connecteur (Mock / Demo / Live)
        order_id = str(uuid.uuid4())
        exec_result = self.connector.place_order(
            order_id=order_id,
            asset=asset,
            direction=direction,
            amount=amount,
            timeframe=timeframe,
        )

        # 4. Enregistrement dans l'historique seulement après validation
        record = {
            "order_id": order_id,
            "asset": asset,
            "direction": direction,
            "amount": amount,
            "timeframe": timeframe,
            "timestamp": now,
            "result": exec_result,
        }
        self._order_history.append(record)

        return {
            "status": "ACCEPTED",
            "order_id": order_id,
            "connector_response": exec_result,
            "timestamp": now.isoformat(),
        }
