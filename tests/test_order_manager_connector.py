from dataclasses import dataclass

from order_manager import OrderManager
from po_connector import PocketOptionConnector


@dataclass
class DemoOrder:
    order_id: str = "TEST-001"
    asset: str = "EURUSD"
    direction: str = "BUY"
    amount: float = 10.0
    payout: float = 85.0
    expiration_seconds: int = 60


class FakeConnector:
    def __init__(self, result=True):
        self.result = result
        self.calls = []
        self.last_execution = None

    def execute_order(self, order):
        self.calls.append(order)

        if self.result:
            self.last_execution = {
                "order_id": order.order_id,
                "confirmed": True,
            }

        return self.result


def test_pocket_option_connector_demo():
    connector = PocketOptionConnector(
        mode="demo",
        trading_enabled=False,
        live_enabled=False,
    )

    result = connector.execute_order(
        DemoOrder()
    )

    assert result is True
    assert connector.last_execution["confirmed"] is True


def test_order_manager_uses_execute_order():
    connector = FakeConnector()

    manager = OrderManager(
        connector=connector,
        max_position_size=50.0,
    )

    result = manager.submit_order(
        asset="EURUSD",
        direction="BUY",
        amount=10.0,
        payout=85.0,
        expiration_seconds=60,
    )

    assert result["status"] == "ACCEPTED"
    assert len(connector.calls) == 1

    executed_order = connector.calls[0]

    assert executed_order.asset == "EURUSD"
    assert executed_order.direction == "BUY"
    assert executed_order.amount == 10.0
    assert executed_order.payout == 85.0
    assert executed_order.expiration_seconds == 60


def test_order_manager_rejects_low_payout():
    connector = FakeConnector()

    manager = OrderManager(
        connector=connector
    )

    result = manager.submit_order(
        asset="EURUSD",
        direction="BUY",
        amount=10.0,
        payout=79.0,
        expiration_seconds=60,
    )

    assert result["status"] == "REJECTED"
    assert len(connector.calls) == 0


def test_order_manager_rejects_short_expiration():
    connector = FakeConnector()

    manager = OrderManager(
        connector=connector
    )

    result = manager.submit_order(
        asset="EURUSD",
        direction="BUY",
        amount=10.0,
        payout=85.0,
        expiration_seconds=59,
    )

    assert result["status"] == "REJECTED"
    assert len(connector.calls) == 0


def test_order_manager_rejects_invalid_direction():
    connector = FakeConnector()

    manager = OrderManager(
        connector=connector
    )

    result = manager.submit_order(
        asset="EURUSD",
        direction="CALL",
        amount=10.0,
        payout=85.0,
        expiration_seconds=60,
    )

    assert result["status"] == "REJECTED"
    assert len(connector.calls) == 0


def test_order_manager_rejects_excessive_amount():
    connector = FakeConnector()

    manager = OrderManager(
        connector=connector,
        max_position_size=50.0,
    )

    result = manager.submit_order(
        asset="EURUSD",
        direction="BUY",
        amount=51.0,
        payout=85.0,
        expiration_seconds=60,
    )

    assert result["status"] == "REJECTED"
    assert len(connector.calls) == 0


def test_order_manager_rejects_duplicate():
    connector = FakeConnector()

    manager = OrderManager(
        connector=connector
    )

    first = manager.submit_order(
        asset="EURUSD",
        direction="BUY",
        amount=10.0,
        payout=85.0,
        expiration_seconds=60,
    )

    second = manager.submit_order(
        asset="EURUSD",
        direction="BUY",
        amount=10.0,
        payout=85.0,
        expiration_seconds=60,
    )

    assert first["status"] == "ACCEPTED"
    assert second["status"] == "REJECTED"

    assert len(connector.calls) == 1


def test_order_manager_handles_connector_failure():
    connector = FakeConnector(
        result=False
    )

    manager = OrderManager(
        connector=connector
    )

    result = manager.submit_order(
        asset="EURUSD",
        direction="BUY",
        amount=10.0,
        payout=85.0,
        expiration_seconds=60,
    )

    assert result["status"] == "FAILED"
    assert len(connector.calls) == 1
