"""
order_manager.py - Gestionnaire d'ordres, des risques et persistance SQLite
pour pocket-ai-trader.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
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
        self.order_history: List[Dict[str, Any]] = []

    # =========================================================
    # ARRÊT D'URGENCE
    # =========================================================

    def trigger_emergency_stop(
        self,
        reason: str = "Arrêt manuel d'urgence"
    ) -> None:
        """Bloque immédiatement l'envoi de nouveaux ordres."""
        print(f"Blocage immédiat : {reason}")
        self.emergency_stop_triggered = True

    def reset_emergency_stop(self) -> None:
        """Réactive le trading après vérification manuelle."""
        self.emergency_stop_triggered = False

    # =========================================================
    # DÉTECTION DES DOUBLONS
    # =========================================================

    def is_duplicate(
        self,
        asset: str,
        direction: str,
        now: datetime,
    ) -> bool:
        """Vérifie si un ordre identique a déjà été soumis récemment."""

        for order in reversed(self.order_history):

            if not isinstance(order, dict):
                continue

            timestamp = order.get("timestamp")

            if not timestamp:
                continue

            try:
                order_time = datetime.fromisoformat(timestamp)
            except (ValueError, TypeError):
                continue

            if (
                (now - order_time).total_seconds()
                > self.duplicate_window_seconds
            ):
                break

            if (
                order.get("asset") == asset
                and order.get("direction") == direction
            ):
                return True

        return False

    # =========================================================
    # PERTE JOURNALIÈRE
    # =========================================================

    def get_daily_loss(self) -> float:
        """Récupère les pertes du jour via le data_store si disponible."""

        if self.data_store is not None:
            try:
                today = datetime.now(timezone.utc).date()
                stats = self.data_store.get_daily_stats(today)

                if isinstance(stats, dict):
                    pnl = stats.get("realized_pnl", 0.0)
                    return abs(float(pnl)) if pnl < 0 else 0.0

            except Exception:
                pass

        return 0.0

    # =========================================================
    # VALIDATION DES RISQUES
    # =========================================================

    def validate_order(
        self,
        order_data: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Contrôle les règles de sécurité avant exécution."""

        if self.emergency_stop_triggered:
            return False, "REJET : Arrêt d'urgence actif."

        amount = order_data.get("amount", 0.0)

        try:
            amount = float(amount)
        except (TypeError, ValueError):
            return False, "REJET : amount invalide."

        if amount <= 0:
            return False, "REJET : Le montant doit être supérieur à zéro."

        if amount > self.max_position_size:
            return (
                False,
                f"REJET : Montant supérieur au max autorisé "
                f"({self.max_position_size})."
            )

        current_loss = self.get_daily_loss()

        if current_loss >= self.max_daily_loss:
            return (
                False,
                "REJET : Perte journalière maximale atteinte."
            )

        return True, None

    # =========================================================
    # SOUMISSION INTERNE
    # =========================================================

    def submit_order(
        self,
        asset: str,
        direction: str,
        amount: float,
        timeframe: int = 60,
    ) -> Dict[str, Any]:
        """
        Pipeline interne d'exécution.

        1. Horodatage
        2. Vérification anti-doublon
        3. Contrôle des règles de risque
        4. Exécution connecteur
        5. Persistance mémoire
        6. Persistance SQLite
        """

        now = datetime.now(timezone.utc)

        asset = str(asset).upper().strip()
        direction = str(direction).upper().strip()

        try:
            amount = float(amount)
            timeframe = int(timeframe)
        except (TypeError, ValueError):
            return {
                "status": "REJECTED",
                "reason": "Paramètres numériques invalides.",
                "order_id": None,
                "timestamp": now.isoformat(),
            }

        order_spec = {
            "asset": asset,
            "direction": direction,
            "amount": amount,
            "timeframe": timeframe,
        }

        # -----------------------------------------------------
        # 1. VÉRIFICATION DES DOUBLONS
        # -----------------------------------------------------

        if self.is_duplicate(asset, direction, now):

            response = {
                "status": "REJECTED",
                "reason": "Ordre en doublon détecté dans l'intervalle de sécurité.",
                "order_id": None,
                "timestamp": now.isoformat(),
                "order_spec": order_spec,
            }

            if self.data_store:
                self.data_store.save_order(response)

            return response

        # -----------------------------------------------------
        # 2. VALIDATION DES RISQUES
        # -----------------------------------------------------

        is_valid, error_msg = self.validate_order(order_spec)

        if not is_valid:

            response = {
                "status": "REJECTED",
                "reason": error_msg,
                "order_id": None,
                "timestamp": now.isoformat(),
                "order_spec": order_spec,
            }

            if self.data_store:
                self.data_store.save_order(response)

            return response

        # -----------------------------------------------------
        # 3. EXÉCUTION VIA LE CONNECTEUR
        # -----------------------------------------------------

        order_id = str(uuid.uuid4())

        try:
            exec_result = self.connector.place_order(
                order_id=order_id,
                asset=asset,
                direction=direction,
                amount=amount,
                timeframe=timeframe,
            )

        except Exception as exc:

            response = {
                "status": "ERROR",
                "reason": f"Erreur connecteur : {exc}",
                "order_id": order_id,
                "timestamp": now.isoformat(),
                "order_spec": order_spec,
            }

            if self.data_store:
                self.data_store.save_order(response)

            return response

        # -----------------------------------------------------
        # 4. ENREGISTREMENT EN MÉMOIRE
        # -----------------------------------------------------

        record = {
            "order_id": order_id,
            "asset": asset,
            "direction": direction,
            "amount": amount,
            "timeframe": timeframe,
            "timestamp": now.isoformat(),
            "result": exec_result,
        }

        self.order_history.append(record)
            def execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Interface publique d'exécution d'un ordre."""
        asset = order.get("asset")
        direction = order.get("direction")
        amount = order.get("amount")
        timeframe = order.get("timeframe", 60)

        if not asset or not direction or amount is None:
            return {
                "status": "REJECTED",
                "reason": "Paramètres d'ordre incomplets.",
                "order_id": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "order": order,
            }

        return self.submit_order(
            asset=asset,
            direction=direction,
            amount=float(amount),
            timeframe=int(timeframe),
        )

        # -----------------------------------------------------
        # 5. PERSISTANCE BASE DE DONNÉES
        # -----------------------------------------------------

        full_response = {
            "status": "ACCEPTED",
            "order_id": order_id,
            "connector_response": exec_result,
            "timestamp": now.isoformat(),
            "order": order_spec,
        }

        if self.data_store:
            self.data_store.save_order(full_response)

        return full_response

    # =========================================================
    # EXECUTE_ORDER
    # =========================================================

    def execute_order(
        self,
        order: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Interface officielle utilisée par le flux Pocket AI Trader.

        Format attendu :

        {
            "asset": "EURUSD",
            "direction": "BUY",
            "amount": 10.0,
            "timeframe": 60
        }

        Cette méthode transforme l'ordre en paramètres compatibles
        avec submit_order().
        """

        # -----------------------------------------------------
        # VÉRIFICATION DU TYPE
        # -----------------------------------------------------

        if not isinstance(order, dict):

            return {
                "status": "REJECTED",
                "reason": "order doit être un dictionnaire.",
                "order_id": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # -----------------------------------------------------
        # RÉCUPÉRATION DES PARAMÈTRES
        # -----------------------------------------------------

        asset = order.get("asset")

        direction = order.get("direction")

        amount = order.get("amount")

        # Accepte timeframe ou expiration
        timeframe = order.get(
            "timeframe",
            order.get("expiration", 60)
        )

        # -----------------------------------------------------
        # VÉRIFICATION DES PARAMÈTRES OBLIGATOIRES
        # -----------------------------------------------------

        if not asset:
            return {
                "status": "REJECTED",
                "reason": "asset est obligatoire.",
                "order_id": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "order": order,
            }

        if not direction:
            return {
                "status": "REJECTED",
                "reason": "direction est obligatoire.",
                "order_id": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "order": order,
            }

        if amount is None:
            return {
                "status": "REJECTED",
                "reason": "amount est obligatoire.",
                "order_id": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "order": order,
            }

        # -----------------------------------------------------
        # NORMALISATION
        # -----------------------------------------------------

        asset = str(asset).upper().strip()
        direction = str(direction).upper().strip()

        try:
            amount = float(amount)
            timeframe = int(timeframe)
        except (TypeError, ValueError):

            return {
                "status": "REJECTED",
                "reason": "amount ou timeframe invalide.",
                "order_id": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "order": order,
            }

        # -----------------------------------------------------
        # CONTRÔLE EXPIRATION MINIMALE
        # -----------------------------------------------------

        if timeframe < 60:

            return {
                "status": "REJECTED",
                "reason": "L'expiration minimale autorisée est de 60 secondes.",
                "order_id": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "order": order,
            }

        # -----------------------------------------------------
        # APPEL DU PIPELINE PRINCIPAL
        # -----------------------------------------------------

        return self.submit_order(
            asset=asset,
            direction=direction,
            amount=amount,
            timeframe=timeframe,
        )
