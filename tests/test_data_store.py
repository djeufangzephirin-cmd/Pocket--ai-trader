from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from data_store import DataStore

@pytest.fixture
def store(tmp_path) -> DataStore:
    db_file = tmp_path / "test_trader.db"
    return DataStore(db_path=str(db_file))

def make_order(order_id="order-1", **overrides):
    order = {
        "order_id": order_id,
        "asset": "EURUSD",
        "direction": "CALL",
        "amount": 10.0,
        "timeframe": 60,
        "status": "ACCEPTED",
        "connector_response": {"ok": True},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    order.update(overrides)
    return order

def test_database_file_is_created(store):
    assert Path(store.db_path).exists()

def test_required_tables_are_created(store):
    with store._get_connection() as conn:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    assert {"orders", "signals", "daily_stats"}.issubset(tables)

def test_save_and_retrieve_order(store):
    store.save_order(make_order())

    orders = store.get_orders()

    assert len(orders) == 1
    assert orders[0]["order_id"] == "order-1"
    assert orders[0]["asset"] == "EURUSD"
    assert orders[0]["status"] == "ACCEPTED"

def test_saving_same_order_id_replaces_existing_order(store):
    store.save_order(make_order(status="ACCEPTED"))
    store.save_order(make_order(status="CLOSED"))

    orders = store.get_orders()

    assert len(orders) == 1
    assert orders[0]["status"] == "CLOSED"

def test_rejected_order_without_id_is_saved(store):
    store.save_order(
        make_order(
            order_id=None,
            status="REJECTED",
            reason="Montant invalide",
        )
    )

    orders = store.get_orders(status="REJECTED")

    assert len(orders) == 1
    assert orders[0]["order_id"].startswith("rejected-")
    assert orders[0]["reason"] == "Montant invalide"

def test_orders_can_be_filtered_by_asset(store):
    store.save_order(make_order(order_id="eur", asset="EURUSD"))
    store.save_order(make_order(order_id="gbp", asset="GBPUSD"))

    orders = store.get_orders(asset="EURUSD")

    assert len(orders) == 1
    assert orders[0]["asset"] == "EURUSD"

def test_orders_can_be_filtered_by_status(store):
    store.save_order(make_order(order_id="accepted", status="ACCEPTED"))
    store.save_order(make_order(order_id="rejected", status="REJECTED"))

    orders = store.get_orders(status="REJECTED")

    assert len(orders) == 1
    assert orders[0]["order_id"] == "rejected"

def test_order_pnl_can_be_updated(store):
    store.save_order(make_order())
    store.update_order_pnl("order-1", 8.5)

    orders = store.get_orders()

    assert orders[0]["pnl"] == pytest.approx(8.5)

def test_get_orders_respects_limit(store):
    for number in range(5):
        store.save_order(make_order(order_id=f"order-{number}"))

    assert len(store.get_orders(limit=2)) == 2

def test_get_orders_rejects_invalid_limit(store):
    with pytest.raises(ValueError):
        store.get_orders(limit=0)

def test_save_signal(store):
    store.save_signal(
        asset="EURUSD",
        signal="BUY",
        confidence=0.82,
        setup_quality="HIGH",
        indicators={"rsi": 65.3, "ema9": 1.085},
    )

    with store._get_connection() as conn:
        rows = conn.execute("SELECT * FROM signals").fetchall()

    assert len(rows) == 1
    assert rows[0]["asset"] == "EURUSD"
    assert rows[0]["confidence"] == pytest.approx(0.82)

def test_daily_stats_accumulate_results(store):
    today = date.today()

    store.record_trade_result(today, pnl=5.0, is_win=True)
    store.record_trade_result(today, pnl=-3.0, is_win=False)
    store.record_trade_result(today, pnl=2.0, is_win=True)

    stats = store.get_daily_stats(today)

    assert stats["total_trades"] == 3
    assert stats["wins"] == 2
    assert stats["losses"] == 1
    assert stats["realized_pnl"] == pytest.approx(4.0)

def test_empty_day_returns_zero_stats(store):
    stats = store.get_daily_stats(date(2000, 1, 1))

    assert stats["total_trades"] == 0
    assert stats["wins"] == 0
    assert stats["losses"] == 0
    assert stats["realized_pnl"] == 0.0
