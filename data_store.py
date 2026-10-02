"""
data_store.py - Persistance SQLite des ordres, signaux et statistiques.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

class DataStore:
    def __init__(self, db_path: str = "pocket_ai_trader.db"):
        self.db_path = Path(db_path)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Ouvre une connexion SQLite temporaire et sécurisée."""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row

        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Crée les tables et les index nécessaires s'ils n'existent pas."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    asset TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    amount REAL NOT NULL,
                    timeframe INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT,
                    result_json TEXT,
                    pnl REAL,
                    timestamp TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence REAL,
                    setup_quality TEXT,
                    indicators_json TEXT,
                    timestamp TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_stats (
                    trade_date TEXT PRIMARY KEY,
                    total_trades INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    realized_pnl REAL DEFAULT 0.0
                )
                """
            )

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_orders_asset ON orders(asset)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_orders_timestamp ON orders(timestamp)"
            )

    def save_order(self, order_record: Dict[str, Any]) -> None:
        """
        Enregistre un ordre accepté ou rejeté.

        Si le même order_id existe déjà, il est mis à jour sans créer de doublon.
        """
        order_id = order_record.get("order_id")

        if not order_id:
            order_id = (
                f"rejected-"
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
            )

        timestamp = order_record.get(
            "timestamp",
            datetime.now(timezone.utc).isoformat(),
        )

        connector_response = order_record.get("connector_response", {})

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO orders (
                    order_id,
                    asset,
                    direction,
                    amount,
                    timeframe,
                    status,
                    reason,
                    result_json,
                    pnl,
                    timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order_id,
                    order_record.get("asset", ""),
                    order_record.get("direction", ""),
                    order_record.get("amount", 0.0),
                    order_record.get("timeframe", 0),
                    order_record.get("status", "UNKNOWN"),
                    order_record.get("reason"),
                    json.dumps(connector_response),
                    order_record.get("pnl"),
                    timestamp,
                ),
            )

    def update_order_pnl(self, order_id: str, pnl: float) -> None:
        """Met à jour le gain ou la perte d'un ordre clôturé."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE orders SET pnl = ? WHERE order_id = ?",
                (pnl, order_id),
            )

    def get_orders(
        self,
        asset: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retourne les ordres récents, avec filtres facultatifs."""
        if limit < 1:
            raise ValueError("limit doit être supérieur ou égal à 1.")

        query = "SELECT * FROM orders WHERE 1 = 1"
        params: List[Any] = []

        if asset is not None:
            query += " AND asset = ?"
            params.append(asset)

        if status is not None:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [dict(row) for row in rows]

    def save_signal(
        self,
        asset: str,
        signal: str,
        confidence: float,
        setup_quality: str,
        indicators: Dict[str, Any],
    ) -> None:
        """Enregistre un signal produit par strategy_engine."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO signals (
                    asset,
                    signal,
                    confidence,
                    setup_quality,
                    indicators_json,
                    timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    asset,
                    signal,
                    confidence,
                    setup_quality,
                    json.dumps(indicators),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def record_trade_result(
        self,
        trade_date: date,
        pnl: float,
        is_win: bool,
    ) -> None:
        """Ajoute le résultat d'un trade aux statistiques du jour."""
        date_str = trade_date.isoformat()
        wins_increment = 1 if is_win else 0
        losses_increment = 0 if is_win else 1

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO daily_stats (
                    trade_date,
                    total_trades,
                    wins,
                    losses,
                    realized_pnl
                )
                VALUES (?, 1, ?, ?, ?)
                ON CONFLICT(trade_date) DO UPDATE SET
                    total_trades = total_trades + 1,
                    wins = wins + excluded.wins,
                    losses = losses + excluded.losses,
                    realized_pnl = realized_pnl + excluded.realized_pnl
                """,
                (date_str, wins_increment, losses_increment, pnl),
            )

    def get_daily_stats(self, trade_date: date) -> Dict[str, Any]:
        """Retourne les statistiques pour une date donnée."""
        date_str = trade_date.isoformat()

        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM daily_stats WHERE trade_date = ?",
                (date_str,),
            ).fetchone()

        if row is None:
            return {
                "trade_date": date_str,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "realized_pnl": 0.0,
            }

        return dict(row)