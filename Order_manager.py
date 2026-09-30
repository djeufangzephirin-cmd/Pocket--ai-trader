"""
order_manager.py
Pocket AI Trader
Version: 1.0.0

Responsabilités:
- Validation des signaux provenant de server.py
- Construction et validation des ordres
- Gestion DEMO / LIVE
- Kill switch
- Contrôle payout
- Contrôle expiration
- Contrôle du risque
- Anti-doublon
- Limitation à un trade simultané
- Cooldown
- Journalisation

IMPORTANT:
- server.py v4.1.0 n'est pas modifié.
- DEMO est le mode par défaut.
- TRADING_ENABLED=false par défaut.
- Aucun ordre LIVE ne peut être envoyé accidentellement.
"""

from __future__ import annotations

import os
import time
import uuid
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from threading import Lock
from typing import Any, Dict, Optional


# ============================================================
# CONFIGURATION
# ============================================================

TRADING_MODE = os.getenv("TRADING_MODE", "demo").strip().lower()

TRADING_ENABLED = (
    os.getenv("TRADING_ENABLED", "false").strip().lower()
    == "true"
)

KILL_SWITCH = (
    os.getenv("TRADING_KILL_SWITCH", "true").strip().lower()
    == "true"
)

MIN_PAYOUT = 80.0
MAX_PAYOUT = 92.0

MIN_EXPIRATION_SECONDS = 60

DEFAULT_RISK_PERCENT = 1.0
MIN_RISK_PERCENT = 0.1
MAX_RISK_PERCENT = 2.0

MAX_SIMULTANEOUS_TRADES = 1

DEFAULT_COOLDOWN_SECONDS = 60

ALLOWED_DIRECTIONS = {"BUY", "SELL"}
ALLOWED_MODES = {"demo", "live"}

LOGGER_NAME = "pocket_ai_trader.order_manager"


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(LOGGER_NAME)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)


# ============================================================
# ENUMS
# ============================================================

class TradingMode(str, Enum):
    DEMO = "demo"
    LIVE = "live"


class OrderStatus(str, Enum):
    REJECTED = "rejected"
    CREATED = "created"
    EXECUTED = "executed"
    CLOSED = "closed"


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class TradeOrder:
    order_id: str
    asset: str
    direction: str
    amount: float
    payout: float
    expiration_seconds: int

    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None

    risk_percent: float = DEFAULT_RISK_PERCENT

    mode: str = TradingMode.DEMO.value
    status: str = OrderStatus.CREATED.value

    created_at: float = 0.0
    executed_at: Optional[float] = None
    closed_at: Optional[float] = None

    result: Optional[str] = None
    profit_loss: Optional[float] = None


# ============================================================
# EXCEPTIONS
# ============================================================

class OrderManagerError(Exception):
    """Erreur générale du gestionnaire d'ordres."""


class OrderValidationError(OrderManagerError):
    """Ordre invalide."""


class TradingDisabledError(OrderManagerError):
    """Trading désactivé."""


class KillSwitchError(OrderManagerError):
    """Kill switch actif."""


class RiskLimitError(OrderManagerError):
    """Limite de risque dépassée."""


class DuplicateTradeError(OrderManagerError):
    """Trade déjà présent ou identique."""


# ============================================================
# ORDER MANAGER
# ============================================================

class OrderManager:
    """
    Gestionnaire central des ordres.

    server.py fournit:
        signal
        confidence
        trade_plan
        risk_management

    order_manager.py décide uniquement si l'ordre respecte
    les règles d'exécution.

    L'exécution physique est déléguée au connector.
    """

    def __init__(
        self,
        connector: Optional[Any] = None,
        mode: Optional[str] = None,
        trading_enabled: Optional[bool] = None,
        kill_switch: Optional[bool] = None,
        cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
    ):
        self.mode = (
            mode.strip().lower()
            if mode is not None
            else TRADING_MODE
        )

        self.trading_enabled = (
            TRADING_ENABLED
            if trading_enabled is None
            else bool(trading_enabled)
        )

        self.kill_switch = (
            KILL_SWITCH
            if kill_switch is None
            else bool(kill_switch)
        )

        self.cooldown_seconds = max(
            0,
            int(cooldown_seconds),
        )

        self.connector = connector

        self._lock = Lock()

        self.active_orders: Dict[str, TradeOrder] = {}

        self.last_trade_timestamp: Optional[float] = None

        self.last_signal_key: Optional[str] = None

        logger.info(
            "OrderManager initialisé | mode=%s | enabled=%s | "
            "kill_switch=%s",
            self.mode,
            self.trading_enabled,
            self.kill_switch,
        )


    # ========================================================
    # CONFIGURATION
    # ========================================================

    def get_status(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "trading_enabled": self.trading_enabled,
            "kill_switch": self.kill_switch,
            "active_orders": len(self.active_orders),
            "max_simultaneous_trades": MAX_SIMULTANEOUS_TRADES,
            "cooldown_seconds": self.cooldown_seconds,
            "min_payout": MIN_PAYOUT,
            "max_payout": MAX_PAYOUT,
            "min_expiration_seconds": MIN_EXPIRATION_SECONDS,
        }


    def enable_trading(self) -> None:
        """
        Active explicitement le moteur.

        Le kill switch reste prioritaire.
        """
        self.trading_enabled = True

        logger.warning(
            "TRADING ENABLED | mode=%s",
            self.mode,
        )


    def disable_trading(self) -> None:
        """
        Désactive immédiatement les nouveaux ordres.
        """
        self.trading_enabled = False

        logger.warning("TRADING DISABLED")


    def activate_kill_switch(self) -> None:
        """
        Arrêt d'urgence.
        """
        self.kill_switch = True

        logger.critical(
            "KILL SWITCH ACTIVATED"
        )


    def deactivate_kill_switch(self) -> None:
        """
        Réactivation manuelle du système.
        """
        self.kill_switch = False

        logger.warning(
            "KILL SWITCH DEACTIVATED"
        )


    # ========================================================
    # VALIDATION CONFIGURATION
    # ========================================================

    def validate_configuration(self) -> None:

        if self.mode not in ALLOWED_MODES:
            raise OrderValidationError(
                f"TRADING_MODE invalide: {self.mode}. "
                f"Valeurs autorisées: demo, live."
            )

        if self.cooldown_seconds < 0:
            raise OrderValidationError(
                "Le cooldown ne peut pas être négatif."
            )

        # Sécurité supplémentaire:
        # LIVE ne peut jamais être activé implicitement.
        if self.mode == TradingMode.LIVE.value:
            logger.warning(
                "MODE LIVE détecté. "
                "Une activation explicite reste nécessaire."
            )


    # ========================================================
    # VALIDATION PAYOUT
    # ========================================================

    @staticmethod
    def validate_payout(payout: float) -> float:

        try:
            payout = float(payout)
        except (TypeError, ValueError):
            raise OrderValidationError(
                "Payout invalide."
            )

        if payout < MIN_PAYOUT:
            raise OrderValidationError(
                f"Payout trop faible: {payout}%. "
                f"Minimum accepté: {MIN_PAYOUT}%."
            )

        if payout > MAX_PAYOUT:
            raise OrderValidationError(
                f"Payout trop élevé: {payout}%. "
                f"Maximum accepté: {MAX_PAYOUT}%."
            )

        return payout


    # ========================================================
    # VALIDATION EXPIRATION
    # ========================================================

    @staticmethod
    def validate_expiration(
        expiration_seconds: int,
    ) -> int:

        try:
            expiration_seconds = int(
                expiration_seconds
            )
        except (TypeError, ValueError):
            raise OrderValidationError(
                "Expiration invalide."
            )

        if expiration_seconds < MIN_EXPIRATION_SECONDS:
            raise OrderValidationError(
                f"Expiration trop courte: "
                f"{expiration_seconds}s. "
                f"Minimum: {MIN_EXPIRATION_SECONDS}s."
            )

        return expiration_seconds


    # ========================================================
    # VALIDATION RISQUE
    # ========================================================

    @staticmethod
    def validate_risk_percent(
        risk_percent: float,
    ) -> float:

        try:
            risk_percent = float(risk_percent)
        except (TypeError, ValueError):
            raise RiskLimitError(
                "Risk percent invalide."
            )

        if not (
            MIN_RISK_PERCENT
            <= risk_percent
            <= MAX_RISK_PERCENT
        ):
            raise RiskLimitError(
                f"Le risque doit être compris entre "
                f"{MIN_RISK_PERCENT}% et "
                f"{MAX_RISK_PERCENT}%."
            )

        return risk_percent


    # ========================================================
    # VALIDATION DIRECTION
    # ========================================================

    @staticmethod
    def validate_direction(
        direction: str,
    ) -> str:

        if not isinstance(direction, str):
            raise OrderValidationError(
                "Direction invalide."
            )

        direction = direction.strip().upper()

        if direction not in ALLOWED_DIRECTIONS:
            raise OrderValidationError(
                f"Direction invalide: {direction}. "
                f"Valeurs: BUY ou SELL."
            )

        return direction


    # ========================================================
    # COOLDOWN
    # ========================================================

    def cooldown_active(self) -> bool:

        if self.last_trade_timestamp is None:
            return False

        elapsed = (
            time.time()
            - self.last_trade_timestamp
        )

        return elapsed < self.cooldown_seconds


    def cooldown_remaining(self) -> float:

        if self.last_trade_timestamp is None:
            return 0.0

        remaining = (
            self.cooldown_seconds
            - (
                time.time()
                - self.last_trade_timestamp
            )
        )

        return max(0.0, remaining)


    # ========================================================
    # SIGNAL KEY / ANTI-DUPLICATION
    # ========================================================

    @staticmethod
    def build_signal_key(
        asset: str,
        direction: str,
        entry_price: Optional[float],
    ) -> str:

        return (
            f"{asset.upper()}|"
            f"{direction.upper()}|"
            f"{entry_price}"
        )


    # ========================================================
    # ACTIVE TRADE LIMIT
    # ========================================================

    def has_active_trade(self) -> bool:
        return len(self.active_orders) >= (
            MAX_SIMULTANEOUS_TRADES
        )


    # ========================================================
    # PRE-TRADE CHECK
    # ========================================================

    def pre_trade_check(
        self,
        direction: str,
        payout: float,
        expiration_seconds: int,
        risk_percent: float,
        asset: str,
        entry_price: Optional[float] = None,
    ) -> None:

        self.validate_configuration()

        direction = self.validate_direction(
            direction
        )

        self.validate_payout(payout)

        self.validate_expiration(
            expiration_seconds
        )

        self.validate_risk_percent(
            risk_percent
        )

        if not asset:
            raise OrderValidationError(
                "Asset obligatoire."
            )

        if self.kill_switch:
            raise KillSwitchError(
                "KILL SWITCH actif. "
                "Aucun nouvel ordre autorisé."
            )

        if not self.trading_enabled:
            raise TradingDisabledError(
                "TRADING_ENABLED=false. "
                "Aucun ordre autorisé."
            )

        if self.has_active_trade():
            raise DuplicateTradeError(
                "Limite de trades simultanés atteinte."
            )

        if self.cooldown_active():
            raise DuplicateTradeError(
                "Cooldown actif. "
                f"Encore "
                f"{self.cooldown_remaining():.1f}s."
            )

        signal_key = self.build_signal_key(
            asset,
            direction,
            entry_price,
        )

        if signal_key == self.last_signal_key:
            raise DuplicateTradeError(
                "Signal identique déjà traité."
            )


    # ========================================================
    # BUILD ORDER
    # ========================================================

    def create_order(
        self,
        asset: str,
        direction: str,
        amount: float,
        payout: float,
        expiration_seconds: int,
        risk_percent: float = DEFAULT_RISK_PERCENT,
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit_1: Optional[float] = None,
        take_profit_2: Optional[float] = None,
    ) -> TradeOrder:

        with self._lock:

            self.pre_trade_check(
                direction=direction,
                payout=payout,
                expiration_seconds=expiration_seconds,
                risk_percent=risk_percent,
                asset=asset,
                entry_price=entry_price,
            )

            try:
                amount = float(amount)
            except (TypeError, ValueError):
                raise OrderValidationError(
                    "Montant invalide."
                )

            if amount <= 0:
                raise OrderValidationError(
                    "Le montant doit être > 0."
                )

            direction = self.validate_direction(
                direction
            )

            payout = self.validate_payout(
                payout
            )

            expiration_seconds = (
                self.validate_expiration(
                    expiration_seconds
                )
            )

            risk_percent = (
                self.validate_risk_percent(
                    risk_percent
                )
            )

            order_id = str(
                uuid.uuid4()
            )

            now = time.time()

            order = TradeOrder(
                order_id=order_id,
                asset=asset.upper(),
                direction=direction,
                amount=amount,
                payout=payout,
                expiration_seconds=expiration_seconds,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                risk_percent=risk_percent,
                mode=self.mode,
                status=OrderStatus.CREATED.value,
                created_at=now,
            )

            self.active_orders[order_id] = order

            self.last_signal_key = (
                self.build_signal_key(
                    asset,
                    direction,
                    entry_price,
                )
            )

            logger.info(
                "Ordre créé | id=%s | %s %s | "
                "amount=%s | payout=%s | expiration=%ss | mode=%s",
                order.order_id,
                order.asset,
                order.direction,
                order.amount,
                order.payout,
                order.expiration_seconds,
                order.mode,
            )

            return order


    # ========================================================
    # EXECUTE ORDER
    # ========================================================

    def execute_order(
        self,
        order: TradeOrder,
    ) -> TradeOrder:

        if order.order_id not in self.active_orders:
            raise OrderManagerError(
                "Ordre inconnu."
            )

        # ----------------------------------------------------
        # DEMO
        # ----------------------------------------------------

        if self.mode == TradingMode.DEMO.value:

            logger.info(
                "DEMO EXECUTION | %s %s | amount=%s",
                order.asset,
                order.direction,
                order.amount,
            )

            order.status = (
                OrderStatus.EXECUTED.value
            )

            order.executed_at = time.time()

            self.last_trade_timestamp = (
                order.executed_at
            )

            return order

        # ----------------------------------------------------
        # LIVE
        # ----------------------------------------------------

        if self.mode == TradingMode.LIVE.value:

            if not self.trading_enabled:
                raise TradingDisabledError(
                    "Trading LIVE désactivé."
                )

            if self.kill_switch:
                raise KillSwitchError(
                    "KILL SWITCH actif."
                )

            if self.connector is None:
                raise OrderManagerError(
                    "Connector LIVE absent. "
                    "Aucun ordre LIVE envoyé."
                )

            result = self._send_to_connector(
                order
            )

            if not result:
                raise OrderManagerError(
                    "Le connector n'a pas confirmé "
                    "l'exécution."
                )

            order.status = (
                OrderStatus.EXECUTED.value
            )

            order.executed_at = time.time()

            self.last_trade_timestamp = (
                order.executed_at
            )

            logger.warning(
                "LIVE ORDER EXECUTED | id=%s",
                order.order_id,
            )

            return order

        raise OrderManagerError(
            f"Mode inconnu: {self.mode}"
        )


    # ========================================================
    # CONNECTOR
    # ========================================================

    def _send_to_connector(
        self,
        order: TradeOrder,
    ) -> bool:

        if self.connector is None:
            return False

        # Interface volontairement stricte.
        #
        # Le connector doit fournir:
        #
        # execute_order(order)
        #
        # et retourner True uniquement si
        # la plateforme confirme l'ordre.

        execute_method = getattr(
            self.connector,
            "execute_order",
            None,
        )

        if not callable(execute_method):
            raise OrderManagerError(
    "Le connector doit exposer "
                "execute_order(order)."
            )

        result = execute_method(order)

        return bool(result)


    # ========================================================
    # COMPLETE / CLOSE ORDER
    # ========================================================

    def close_order(
        self,
        order_id: str,
        result: str,
        profit_loss: float,
    ) -> TradeOrder:

        with self._lock:

            order = self.active_orders.get(
                order_id
            )

            if order is None:
                raise OrderManagerError(
                    "Ordre introuvable."
                )

            order.status = (
                OrderStatus.CLOSED.value
            )

            order.closed_at = time.time()

            order.result = str(result)

            order.profit_loss = float(
                profit_loss
            )

            del self.active_orders[
                order_id
            ]

            logger.info(
                "Ordre clôturé | id=%s | result=%s | P/L=%s",
                order.order_id,
                order.result,
                order.profit_loss,
            )

            return order


    # ========================================================
    # CANCEL ORDER
    # ========================================================

    def cancel_order(
        self,
        order_id: str,
    ) -> TradeOrder:

        with self._lock:

            order = self.active_orders.get(
                order_id
            )

            if order is None:
                raise OrderManagerError(
                    "Ordre introuvable."
                )

            order.status = (
                OrderStatus.REJECTED.value
            )

            del self.active_orders[
                order_id
            ]

            logger.info(
                "Ordre annulé | id=%s",
                order.order_id,
            )

            return order


    # ========================================================
    # ACTIVE ORDERS
    # ========================================================

    def get_active_orders(self):
        return [
            asdict(order)
            for order in self.active_orders.values()
        ]


    # ========================================================
    # RESET
    # ========================================================

    def reset_signal_guard(self) -> None:
        """
        Permet de débloquer le prochain signal
        après une nouvelle période de marché.
        """
        self.last_signal_key = None


# ============================================================
# SINGLETON
# ============================================================

order_manager = OrderManager()


# ============================================================
# SIMPLE PUBLIC API
# ============================================================

def get_order_manager() -> OrderManager:
    return order_manager


def get_trading_status() -> Dict[str, Any]:return order_manager.get_status()


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    manager = OrderManager(
        mode="demo",
        trading_enabled=False,
        kill_switch=True,
    )

    print(
        "Order Manager status:"
    )

    print(
        manager.get_status()
    )

    print(
        "\nSafety configuration OK."
    )

    print(
        "No order can be executed because:"
    )

    print(
        "- TRADING_ENABLED=false"
    )

    print(
        "- KILL_SWITCH=true"
    )
