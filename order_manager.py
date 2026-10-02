"""
order_manager.py - Gestionnaire d'ordres, des risques et persistance SQLite pour pocket-ai-trader.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

class OrderManager:
    def __init__(
        self,
        connector,
        data_store=None,
        max_position_size: float = 50.0,
        max_daily_loss: float = 100.0,
        duplicate_window_seconds: int = 10,
    ):
        self.connector = connector
        self.data_store = data_store
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.duplicate_window_seconds = duplicate_window_seconds

        self.emergency_stop_triggered: bool = False
        self._order_history: List[Dict[str, Any]] = []

    def trigger_emergency_stop(self, reason: str = "Arrêt manuel d'urgence") -> None:
        """Bloque immédiatement l'envoi de nouveaux ordres."""
        self.emergency_stop_triggered = True

    def reset_emergency_stop(self) -> None:
        """Réactive le trading après vérification manuelle."""
        self.emergency_stop_triggered = False

    def is_duplicate(self, asset: str, direction: str, now: datetime) -> bool:
        """Vérifie si un ordre identique a déjà été soumis récemment."""
        for order in reversed(self._order_history):
            time_diff = (now - order["timestamp"]).total_seconds()
            if time_diff > self.duplicate_window_seconds:
                break
            if order["asset"] == asset and order["direction"] == direction:
                return True
        return False

    def get_daily_loss(self) -> float:
        """Récupère les pertes du jour via le data_store si disponible."""
        if self.data_store is not None:
            today = datetime.now(timezone.utc).date()
            stats = self.data_store.get_daily_stats(today)
            pnl = stats.get("realized_pnl", 0.0)
            return abs(pnl) if pnl < 0 else 0.0
        return 0.0

    def validate_order(self, order_data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Contrôle les règles de sécurité avant exécution."""
        if self.emergency_stop_triggered:
            return False, "REJET : Arrêt d'urgence actif."

        amount = order_data.get("amount", 0.0)
        if amount <= 0:
            return False, "REJET : Le montant doit être supérieur à zéro."

        if amount > self.max_position_size:
            return False, f"REJET : Montant {amount} supérieur au max autorisé ({self.max_position_size})."

        current_loss = self.get_daily_loss()
        if current_loss >= self.max_daily_loss:
            return False, f"REJET : Perte journalière maximale atteinte ({self.max_daily_loss})."

        return True, None

    def submit_order(
        self,
        asset: str,
        direction: str,
        amount: float,
        timeframe: int = 60,
    ) -> Dict[str, Any]:
        """
        Pipeline d'exécution :
        1. Horodatage
        2. Vérification anti-doublon (avant enregistrement)
        3. Contrôle des règles de risque
        4. Exécution connecteur
        5. Persistance SQLite (data_store) et mémoire
        """
        now = datetime.now(timezone.utc)
        order_spec = {"asset": asset, "direction": direction, "amount": amount, "timeframe": timeframe}

        # 1. Vérification des doublons
        if self.is_duplicate(asset, direction, now):
            response = {
                "status": "REJECTED",
                "reason": "Ordre en doublon détecté dans l'intervalle de sécurité",
                "order_id": None,
                "timestamp": now.isoformat(),
                **order_spec,
            }
            if self.data_store:
                self.data_store.save_order(response)
            return response

        # 2. Validation des règles de gestion des risques
        is_valid, error_msg = self.validate_order(order_spec)
        if not is_valid:
            response = {
                "status": "REJECTED",
                "reason": error_msg,
                "order_id": None,
                "timestamp": now.isoformat(),
                **order_spec,
            }
            if self.data_store:
                self.data_store.save_order(response)
            return response

        # 3. Exécution via le connecteur
        order_id = str(uuid.uuid4())
        exec_result = self.connector.place_order(
            order_id=order_id,
            asset=asset,
            direction=direction,
            amount=amount,
            timeframe=timeframe,
        )

        # 4. Enregistrement en mémoire
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

        # 5. Persistance dans la base de données
        full_response = {
            "status": "ACCEPTED",
            "order_id": order_id,
            "connector_response": exec_result,
            "timestamp": now.isoformat(),
            **order_spec,
        }
        if self.data_store:
            self.data_store.save_order(full_response)

        return full_response
