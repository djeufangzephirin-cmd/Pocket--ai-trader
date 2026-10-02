"""
test_data_store.py - Tests unitaires pour la persistance SQLite de pocket-ai-trader.
Utilise une base temporaire (tmp_path) pour ne jamais toucher à la base réelle.
"""

import pytest
from datetime import datetime, timezone, date
from pathlib import Path

from data_store import DataStore

@pytest.fixture
def store(tmp_path) -> DataStore:
    """Crée une instance DataStore sur une base SQLite temporaire et isolée."""
    db_file = tmp_path / "test_trader.db"
    return DataStore(db_path=str(db_file))

class TestInitialization:
    def test_db_file_created(self, store: DataStore):
        assert Path(store.db_path).exists()

    def test_tables_exist(self, store: DataStore):
        with store._get_connection() as conn:
            tables = {row["name"] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        assert {"orders", "signals", "daily_stats"}.issubset(tables)

class TestOrders:
    def test_save_and_retrieve_order(self, store: DataStore):
        order_record = {
            "order_id": "abc-123",
            "asset": "EURUSD",
            "direction": "CALL",
            "amount": 10.0,
            "timeframe": 60,
            "status": "ACCEPTED",
            "connector_response": {"ok": True},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        store.save_order(order_record)
        orders = store.get_orders()
        assert len(orders) == 1
        assert orders[0]["order_id"] == "abc-123"
        assert orders[0]["asset"] == "EURUSD"
        assert orders[0]["status"] == "ACCEPTED"

    def test_save_order_idempotent_replace(self, store: DataStore):
        """Un même order_id doit écraser l'ancien, pas créer un doublon."""
        base = {
            "order_id": "dup-1", "asset": "EURUSD", "direction": "CALL",
            "amount": 10.0, "timeframe": 60, "status": "ACCEPTED",
            "connector_response": {}, "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        store.save_order(base)
        base["status"] = "CLOSED"
        store.save_order(base)

        orders = store.get_orders()
        assert len(orders) == 1
        assert orders[0]["status"] == "CLOSED"

    def test_rejected_order_without_id(self, store: DataStore):
        rejected = {
            "order_id": None,
            "asset": "GBPUSD",
            "direction": "PUT",
            "amount": 5.0,
            "timeframe": 30,
            "status": "REJECTED",
            "reason": "Montant invalide",
        }
        store.save_order(rejected)
        orders = store.get_orders(status="REJECTED")
        assert len(orders) == 1
        assert orders[0]["reason"] == "Montant invalide"

    def test_filter_by_asset(self, store: DataStore):
        store.save_order({"order_id": "1", "asset": "EURUSD", "direction": "CALL",
                           "amount": 10, "timeframe": 60, "status": "ACCEPTED"})
        store.save_order({"order_id": "2", "asset": "GBPUSD", "direction": "PUT",
                           "amount": 10, "timeframe": 60, "status": "ACCEPTED"})
        result = store.get_orders(asset="EURUSD")
        assert len(result) == 1
        assert result[0]["asset"] == "EURUSD"

    def test_update_order_pnl(self, store: DataStore):
        store.save_order({"order_id": "pnl-1", "asset": "EURUSD", "direction": "CALL",
                           "amount": 10, "timeframe": 60, "status": "ACCEPTED"})
        store.update_order_pnl("pnl-1", 8.5)
        orders = store.get_orders()
        assert orders[0]["pnl"] == 8.5

    def test_limit_parameter(self, store: DataStore):
        for i in range(5):
            store.save_order({"order_id": str(i), "asset": "EURUSD", "direction": "CALL",
                               "amount": 10, "timeframe": 60, "status": "ACCEPTED"})
        result = store.get_orders(limit=2)
        assert len(result) == 2

class TestSignals:
    def test_save_signal(self, store: DataStore):
        store.save_signal(
            asset="EURUSD", signal="BUY", confidence=0.82,
            setup_quality="HIGH", indicators={"rsi": 65.3, "ema9": 1.085},
        )
        with store._get_connection() as conn:
            rows = conn.execute("SELECT * FROM signals").fetchall()
        assert len(rows) == 1
        assert rows[0]["asset"] == "EURUSD"
        assert rows[0]["confidence"] == 0.82

class TestDailyStats:
    def test_first_trade_of_day(self, store: DataStore):
        today = date.today()
        store.record_trade_result(today, pnl=5.0, is_win=True)
        stats = store.get_daily_stats(today)
        assert stats["total_trades"] == 1
        assert stats["wins"] == 1
        assert stats["losses"] == 0
        assert stats["realized_pnl"] == 5.0

    def test_accumulate_multiple_trades(self, store: DataStore):
        today = date.today()
        store.record_trade_result(today, pnl=5.0, is_win=True)
        store.record_trade_result(today, pnl=-3.0, is_win=False)
        store.record_trade_result(today, pnl=2.0, is_win=True)

        stats = store.get_daily_stats(today)
        assert stats["total_trades"] == 3
        assert stats["wins"] == 2
        assert stats["losses"] == 1
        assert abs(stats["realized_pnl"] - 4.0) < 1e-9

    def test_empty_day_returns_zero(self, store: DataStore):
        far_date = date(2000, 1, 1)
        stats = store.get_daily_stats(far_date)
        assert stats["total_trades"] == 0
        assert stats["realized_pnl"] == 0.0
