"""
po_connector.py
Pocket AI Trader

Connector layer for Pocket Option.

IMPORTANT
---------
- This module does NOT modify server.py.
- It is designed to work with order_manager.py.
- Public interface required by OrderManager:

        execute_order(order) -> bool

- DEMO is the safe default.
- LIVE execution is disabled unless:
    1. TRADING_MODE=live
    2. TRADING_ENABLED=true
    3. PO_LIVE_ENABLED=true
    4. a compatible client is explicitly injected
    5. the client reports a successful execution

Do NOT place Pocket Option credentials directly in this file.
Use environment variables or an injected client.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional


# ============================================================
# CONFIGURATION
# ============================================================

TRADING_MODE = os.getenv(
    "TRADING_MODE",
    "demo",
).strip().lower()

TRADING_ENABLED = (
    os.getenv(
        "TRADING_ENABLED",
        "false",
    ).strip().lower()
    == "true"
)

PO_LIVE_ENABLED = (
    os.getenv(
        "PO_LIVE_ENABLED",
        "false",
    ).strip().lower()
    == "true"
)

PO_CONNECT_TIMEOUT = int(
    os.getenv(
        "PO_CONNECT_TIMEOUT",
        "15",
    )
)

PO_EXECUTION_TIMEOUT = int(
    os.getenv(
        "PO_EXECUTION_TIMEOUT",
        "10",
    )
)


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(
    "pocket_ai_trader.po_connector"
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
# EXCEPTIONS
# ============================================================

class PocketOptionConnectorError(Exception):
    """Base connector exception."""


class ConnectorConfigurationError(
    PocketOptionConnectorError
):
    """Invalid connector configuration."""


class ConnectorConnectionError(
    PocketOptionConnectorError
):
    """Connection failure."""


class ConnectorAuthenticationError(
    PocketOptionConnectorError
):
    """Authentication failure."""


class ConnectorExecutionError(
    PocketOptionConnectorError
):
    """Order execution failure."""


class LiveTradingDisabledError(
    PocketOptionConnectorError
):
    """LIVE trading is not explicitly enabled."""


# ============================================================
# CONNECTOR
# ============================================================

class PocketOptionConnector:
    """
    Adapter between OrderManager and a Pocket Option client.

    The actual platform client is injected instead of being
    hard-coded here.

    Expected client capabilities:

        connect()
        disconnect()

    and one execution method:

        execute_order(...)
        buy(...)
        call(...)
        sell(...)
        put(...)
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        mode: Optional[str] = None,
        trading_enabled: Optional[bool] = None,
        live_enabled: Optional[bool] = None,
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

        self.live_enabled = (
            PO_LIVE_ENABLED
            if live_enabled is None
            else bool(live_enabled)
        )

        self.client = client

        self.connected = False
        self.authenticated = False

        self.last_error: Optional[str] = None

        self.last_execution: Optional[
            Dict[str, Any]
        ] = None

        self._validate_configuration()

    # ========================================================
    # CONFIGURATION
    # ========================================================

    def _validate_configuration(self) -> None:

        if self.mode not in {
            "demo",
            "live",
        }:
            raise ConnectorConfigurationError(
                "TRADING_MODE doit être "
                "'demo' ou 'live'."
            )

        if PO_CONNECT_TIMEOUT <= 0:
            raise ConnectorConfigurationError(
                "PO_CONNECT_TIMEOUT invalide."
            )

        if PO_EXECUTION_TIMEOUT <= 0:
            raise ConnectorConfigurationError(
                "PO_EXECUTION_TIMEOUT invalide."
            )

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self) -> Dict[str, Any]:

        return {
            "connector": "PocketOptionConnector",
            "mode": self.mode,
            "trading_enabled": self.trading_enabled,
            "live_enabled": self.live_enabled,
            "connected": self.connected,
            "authenticated": self.authenticated,
            "client_loaded": self.client is not None,
            "last_error": self.last_error,
        }

    # ========================================================
    # CLIENT INJECTION
    # ========================================================

    def set_client(
        self,
        client: Any,
    ) -> None:

        if client is None:
            raise ConnectorConfigurationError(
                "Le client ne peut pas être None."
            )

        self.client = client

        logger.info(
            "Pocket Option client injecté."
        )

    # ========================================================
    # CONNECTION
    # ========================================================

    def connect(self) -> bool:
        """
        Connecte le client injecté.

        Les identifiants ne sont pas collectés ici.
        """

        if self.client is None:

            self.last_error = (
                "Aucun client Pocket Option "
                "n'est configuré."
            )

            logger.warning(
                self.last_error
            )

            return False

        connect_method = getattr(
            self.client,
            "connect",
            None,
        )

        if not callable(connect_method):

            self.last_error = (
                "Le client doit exposer "
                "connect()."
            )

            raise ConnectorConfigurationError(
                self.last_error
            )

        try:

            result = connect_method()

            if result is False:

                self.last_error = (
                    "Le client a refusé "
                    "la connexion."
                )

                self.connected = False
                self.authenticated = False

                return False

            self.connected = True
            self.authenticated = True
            self.last_error = None

            logger.info(
                "Pocket Option connector connecté."
            )

            return True

        except Exception as exc:

            self.connected = False
            self.authenticated = False
            self.last_error = str(exc)

            logger.exception(
                "Erreur de connexion Pocket Option."
            )

            raise ConnectorConnectionError(
                f"Connexion impossible: {exc}"
            ) from exc

    # ========================================================
    # DISCONNECT
    # ========================================================

    def disconnect(self) -> None:

        if self.client is None:
            self.connected = False
            self.authenticated = False
            return

        disconnect_method = getattr(
            self.client,
            "disconnect",
            None,
        )

        if callable(disconnect_method):

            try:
                disconnect_method()

            except Exception as exc:

                logger.warning(
                    "Erreur lors de la déconnexion: %s",
                    exc,
                )

        self.connected = False
        self.authenticated = False

        logger.info(
            "Pocket Option connector déconnecté."
        )

    # ========================================================
    # CONNECTION CHECK
    # ========================================================

    def is_connected(self) -> bool:

        return (
            self.connected
            and self.authenticated
        )

    # ========================================================
    # LIVE SAFETY CHECK
    # ========================================================

    def _check_live_permission(self) -> None:

        if self.mode != "live":
            return

        if not self.trading_enabled:

            raise LiveTradingDisabledError(
                "TRADING_ENABLED=false. "
                "Exécution LIVE interdite."
            )

        if not self.live_enabled:

            raise LiveTradingDisabledError(
                "PO_LIVE_ENABLED=false. "
                "Exécution LIVE interdite."
            )

        if self.client is None:

            raise LiveTradingDisabledError(
                "Aucun client LIVE configuré."
            )

        if not self.is_connected():

            raise ConnectorConnectionError(
                "Le connecteur LIVE n'est "
                "pas connecté/authentifié."
            )

    # ========================================================
    # ORDER VALIDATION
    # ========================================================

    @staticmethod
    def _validate_order(order: Any) -> None:

        if order is None:
            raise ConnectorExecutionError(
                "Order ne peut pas être None."
            )

        required_fields = [
            "order_id",
            "asset",
            "direction",
            "amount",
            "payout",
            "expiration_seconds",
        ]

        missing = []

        for field in required_fields:

            if not hasattr(
                order,
                field,
            ):
                missing.append(field)

        if missing:

            raise ConnectorExecutionError(
                "Champs order manquants: "
                + ", ".join(missing)
            )

        if order.direction not in {
            "BUY",
            "SELL",
        }:

            raise ConnectorExecutionError(
                "Direction invalide: "
                f"{order.direction}"
            )

        if float(order.amount) <= 0:

            raise ConnectorExecutionError(
                "Montant invalide."
            )

        if int(
            order.expiration_seconds
        ) < 60:

            raise ConnectorExecutionError(
                "Expiration minimale: 60 secondes."
            )

    # ========================================================
    # EXECUTE ORDER
    # ========================================================

    def execute_order(
        self,
        order: Any,
    ) -> bool:
        """
        Interface exacte attendue par OrderManager.

        Returns:
            True  -> ordre confirmé
            False -> ordre non confirmé
        """

        self._validate_order(order)

        self.last_error = None

        # ----------------------------------------------------
        # DEMO MODE
        # ----------------------------------------------------

        if self.mode == "demo":

            logger.info(
                "DEMO connector | order=%s | "
                "%s %s | amount=%s | "
                "expiration=%ss",
                order.order_id,
                order.asset,
                order.direction,
                order.amount,
                order.expiration_seconds,
            )

            self.last_execution = {
                "order_id": order.order_id,
                "asset": order.asset,
                "direction": order.direction,
                "amount": float(order.amount),
                "expiration_seconds": int(
                    order.expiration_seconds
                ),
                "mode": "demo",
                "confirmed": True,
                "timestamp": time.time(),
            }

            return True

        # ----------------------------------------------------
        # LIVE MODE
        # ----------------------------------------------------

        self._check_live_permission()

        return self._execute_live(order)

    # ========================================================
    # LIVE EXECUTION
    # ========================================================

    def _execute_live(
        self,
        order: Any,
    ) -> bool:

        client = self.client

        if client is None:
            raise ConnectorExecutionError(
                "Client LIVE absent."
            )

        direction = order.direction.upper()

        # ----------------------------------------------------
        # GENERIC EXECUTION
        # ----------------------------------------------------

        execute_method = getattr(
            client,
            "execute_order",
            None,
        )

        if callable(execute_method):

            try:

                result = execute_method(order)

                confirmed = (
                    self._normalize_execution_result(
                        result
                    )
                )

                if not confirmed:

                    raise ConnectorExecutionError(
                        "Le client n'a pas confirmé "
                        "l'exécution LIVE."
                    )

                self._record_execution(
                    order,
                    result,
                )

                return True

            except ConnectorExecutionError:
                raise

            except Exception as exc:

                self.last_error = str(exc)

                logger.exception(
                    "Erreur exécution LIVE."
                )

                raise ConnectorExecutionError(
                    f"Exécution LIVE échouée: {exc}"
                ) from exc

        # ----------------------------------------------------
        # DIRECTION-SPECIFIC FALLBACK
        # ----------------------------------------------------

        if direction == "BUY":

            method = self._find_method(
                client,
                [
                    "buy",
                    "call",
                ],
            )

        else:

            method = self._find_method(
                client,
                [
                    "sell",
                    "put",
                ],
            )

        if method is None:

            raise ConnectorConfigurationError(
                "Client incompatible: il doit exposer "
                "execute_order(order), buy()/call(), "
                "ou sell()/put()."
            )

        try:

            result = self._call_direction_method(
                method,
                order,
            )

            confirmed = (
                self._normalize_execution_result(
                    result
                )
            )

            if not confirmed:

                raise ConnectorExecutionError(
                    "Le client n'a pas confirmé "
                    "l'ordre LIVE."
                )

            self._record_execution(
                order,
                result,
            )

            return True

        except ConnectorExecutionError:
            raise

        except Exception as exc:

            self.last_error = str(exc)

            logger.exception(
                "Erreur d'exécution LIVE."
            )

            raise ConnectorExecutionError(
                f"Exécution LIVE échouée: {exc}"
            ) from exc

    # ========================================================
    # METHOD DISCOVERY
    # ========================================================

    @staticmethod
    def _find_method(
        client: Any,
        names: list[str],
    ):

        for name in names:

            method = getattr(
                client,
                name,
                None,
            )

            if callable(method):
                return method

        return None

    # ========================================================
    # DIRECTION CALL
    # ========================================================

    @staticmethod
    def _call_direction_method(
        method: Any,
        order: Any,
    ):

        try:

            return method(
                order.asset,
                float(order.amount),
                int(order.expiration_seconds),
            )

        except TypeError:

            return method(
                asset=order.asset,
                amount=float(order.amount),
                expiration=int(
                    order.expiration_seconds
                ),
            )

    # ========================================================
    # RESULT NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_execution_result(
        result: Any,
    ) -> bool:

        if result is True:
            return True

        if result is False:
            return False

        if result is None:
            return False

        if isinstance(
            result,
            dict,
        ):

            for key in (
                "success",
                "confirmed",
                "executed",
                "ok",
            ):

                if key in result:

                    return bool(
                        result[key]
                    )

            for key in (
                "order_id",
                "id",
                "deal_id",
                "ticket",
            ):

                if result.get(key):
                    return True

            return False

        for attr in (
            "success",
            "confirmed",
            "executed",
            "ok",
        ):

            if hasattr(
                result,
                attr,
            ):

                return bool(
                    getattr(
                        result,
                        attr,
                    )
                )

        for attr in (
            "order_id",
            "id",
            "deal_id",
            "ticket",
        ):

            if getattr(
                result,
                attr,
                None,
            ):

                return True

        return False

    # ========================================================
    # RECORD EXECUTION
    # ========================================================

    def _record_execution(
        self,
        order: Any,
        result: Any,
    ) -> None:

        self.last_execution = {
            "order_id": order.order_id,
            "asset": order.asset,
            "direction": order.direction,
            "amount": float(order.amount),
            "expiration_seconds": int(
                order.expiration_seconds
            ),
            "mode": self.mode,
            "confirmed": True,
            "client_result": result,
            "timestamp": time.time(),
        }

        logger.warning(
            "LIVE ORDER CONFIRMED | "
            "order_id=%s | asset=%s | direction=%s | "
            "amount=%s",
            order.order_id,
            order.asset,
            order.direction,
            order.amount,
        )

    # ========================================================
    # ACCOUNT / MARKET HELPERS
    # ========================================================

    def get_balance(
        self,
    ) -> Optional[float]:

        if self.client is None:
            return None

        for name in (
            "get_balance",
            "balance",
        ):

            attr = getattr(
                self.client,
                name,
                None,
            )

            if callable(attr):

                result = attr()

                try:
                    return float(result)

                except (
                    TypeError,
                    ValueError,
                ):
                    return None

            if attr is not None:

                try:
                    return float(attr)

                except (
                    TypeError,
                    ValueError,
                ):
                    return None

        return None

    def get_payout(
        self,
        asset: str,
    ) -> Optional[float]:

        if self.client is None:
            return None

        method = getattr(
            self.client,
            "get_payout",
            None,
        )

        if not callable(method):
            return None

        try:

            result = method(asset)

            return float(result)

        except (
            TypeError,
            ValueError,
        ):

            return None

    # ========================================================
    # HEALTH CHECK
    # ========================================================

    def health_check(
        self,
    ) -> Dict[str, Any]:

        return {
            "status": (
                "ok"
                if self.connected
                else "disconnected"
            ),
            "mode": self.mode,
            "connected": self.connected,
            "authenticated": self.authenticated,
            "client_loaded": (
                self.client is not None
            ),
            "trading_enabled": (
                self.trading_enabled
            ),
            "live_enabled": (
                self.live_enabled
            ),
            "last_error": self.last_error,
        }


# ============================================================
# SINGLETON
# ============================================================

po_connector = PocketOptionConnector()


# ============================================================
# PUBLIC HELPERS
# ============================================================

def get_po_connector() -> PocketOptionConnector:
    return po_connector


def get_connector_status() -> Dict[str, Any]:
    return po_connector.health_check()


# ============================================================
# LOCAL SELF-TEST
# ============================================================

if __name__ == "__main__":

    from dataclasses import dataclass

    @dataclass
    class TestOrder:

        order_id: str = "TEST-001"
        asset: str = "EURUSD"
        direction: str = "BUY"
        amount: float = 10.0
        payout: float = 85.0
        expiration_seconds: int = 60

    connector = PocketOptionConnector(
        mode="demo",
        trading_enabled=False,
        live_enabled=False,
    )

    print(
        "=== PO CONNECTOR SELF TEST ==="
    )

    print(
        connector.get_status()
    )

    order = TestOrder()

    result = connector.execute_order(
        order
    )

    print(
        "DEMO execution:",
        result,
    )

    print(
        "Last execution:",
        connector.last_execution,
    )
