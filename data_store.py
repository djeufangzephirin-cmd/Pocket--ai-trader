"""
data_store.py
Pocket AI Trader

Couche de persistance SQLite.

Responsabilités
---------------
- Persistance des ordres
- Persistance des signaux
- Statistiques journalières
- Résultats des trades
- Lecture des ordres
- Mise à jour du PnL
- Compatibilité avec OrderManager
- Compatibilité avec server.py

Aucune logique de trading n'est exécutée ici.

Architecture
------------

server.py
    ↓
OrderManager
    ↓
DataStore
    ↓
SQLite

Le DataStore ne décide jamais BUY / SELL.
Il stocke uniquement les données produites
par les autres modules.
"""

from __future__ import annotations

import json
import sqlite3

from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_DB_PATH = "pocket_ai_trader.db"


# ============================================================
# DATA STORE
# ============================================================

class DataStore:
    """
    Gestionnaire central de la persistance SQLite.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
    ):
        self.db_path = Path(
            db_path
        )

        self._init_db()

    # ========================================================
    # DATABASE CONNECTION
    # ========================================================

    @contextmanager
    def _get_connection(self):
        """
        Ouvre une connexion SQLite temporaire.

        La connexion est automatiquement :
        - commitée si tout fonctionne
        - rollbackée en cas d'erreur
        - fermée dans tous les cas
        """

        conn = sqlite3.connect(
            self.db_path,
            timeout=10,
        )

        conn.row_factory = sqlite3.Row

        try:

            yield conn

            conn.commit()

        except Exception:

            conn.rollback()

            raise

        finally:

            conn.close()

    # ========================================================
    # DATABASE INITIALIZATION
    # ========================================================

    def _init_db(
        self,
    ) -> None:
        """
        Crée les tables nécessaires si elles n'existent pas.
        """

        with self._get_connection() as conn:

            # ------------------------------------------------
            # ORDERS
            # ------------------------------------------------

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

            # ------------------------------------------------
            # SIGNALS
            # ------------------------------------------------

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

            # ------------------------------------------------
            # DAILY STATS
            # ------------------------------------------------

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

            # ------------------------------------------------
            # INDEX ORDERS ASSET
            # ------------------------------------------------

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_orders_asset
                ON orders(asset)
                """
            )

            # ------------------------------------------------
            # INDEX ORDERS TIMESTAMP
            # ------------------------------------------------

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_orders_timestamp
                ON orders(timestamp)
                """
            )

            # ------------------------------------------------
            # INDEX ORDERS STATUS
            # ------------------------------------------------

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_orders_status
                ON orders(status)
                """
            )

            # ------------------------------------------------
            # INDEX SIGNALS TIMESTAMP
            # ------------------------------------------------

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_signals_timestamp
                ON signals(timestamp)
                """
            )

    # ========================================================
    # SAVE ORDER
    # ========================================================

    def save_order(
        self,
        order_record: Dict[str, Any],
    ) -> None:
        """
        Enregistre un ordre accepté, rejeté ou échoué.

        Compatible avec OrderManager.

        Exemple :

        {
            "order_id": "...",
            "asset": "EURUSD",
            "direction": "BUY",
            "amount": 10,
            "timeframe": 60,
            "status": "ACCEPTED",
            "reason": None,
            "connector_response": {...},
            "timestamp": "..."
        }
        """

        if not isinstance(
            order_record,
            dict,
        ):
            raise TypeError(
                "order_record doit être un dictionnaire."
            )

        # ----------------------------------------------------
        # ORDER ID
        # ----------------------------------------------------

        order_id = order_record.get(
            "order_id"
        )

        if not order_id:

            order_id = (
                "rejected-"
                + datetime.now(
                    timezone.utc
                ).strftime(
                    "%Y%m%dT%H%M%S%fZ"
                )
            )

        order_id = str(
            order_id
        )

        # ----------------------------------------------------
        # TIMESTAMP
        # ----------------------------------------------------

        timestamp = order_record.get(
            "timestamp"
        )

        if timestamp is None:

            timestamp = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

        timestamp = str(
            timestamp
        )

        # ----------------------------------------------------
        # CONNECTOR RESPONSE
        # ----------------------------------------------------

        connector_response = (
            order_record.get(
                "connector_response",
                {},
            )
        )

        try:

            result_json = json.dumps(
                connector_response,
                default=str,
            )

        except Exception:

            result_json = json.dumps(
                {
                    "raw": str(
                        connector_response
                    )
                }
            )

        # ----------------------------------------------------
        # AMOUNT
        # ----------------------------------------------------

        try:

            amount = float(
                order_record.get(
                    "amount",
                    0.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            amount = 0.0

        # ----------------------------------------------------
        # TIMEFRAME
        # ----------------------------------------------------

        try:

            timeframe = int(
                order_record.get(
                    "timeframe",
                    order_record.get(
                        "expiration_seconds",
                        0,
                    ),
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            timeframe = 0

        # ----------------------------------------------------
        # PNL
        # ----------------------------------------------------

        pnl = order_record.get(
            "pnl"
        )

        if pnl is not None:

            try:

                pnl = float(
                    pnl
                )

            except (
                TypeError,
                ValueError,
            ):

                pnl = None

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

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

                    str(
                        order_record.get(
                            "asset",
                            "",
                        )
                    ),

                    str(
                        order_record.get(
                            "direction",
                            "",
                        )
                    ).upper(),

                    amount,

                    timeframe,

                    str(
                        order_record.get(
                            "status",
                            "UNKNOWN",
                        )
                    ),

                    order_record.get(
                        "reason"
                    ),

                    result_json,

                    pnl,

                    timestamp,
                ),
            )

    # ========================================================
    # UPDATE ORDER PNL
    # ========================================================

    def update_order_pnl(
        self,
        order_id: str,
        pnl: float,
    ) -> None:
        """
        Met à jour le gain/perte d'un ordre clôturé.
        """

        if not order_id:

            raise ValueError(
                "order_id est obligatoire."
            )

        try:

            pnl_value = float(
                pnl
            )

        except (
            TypeError,
            ValueError,
        ):

            raise ValueError(
                "pnl doit être numérique."
            )

        with self._get_connection() as conn:

            conn.execute(
                """
                UPDATE orders

                SET pnl = ?

                WHERE order_id = ?
                """,

                (
                    pnl_value,
                    str(order_id),
                ),
            )

    # ========================================================
    # GET ORDERS
    # ========================================================

    def get_orders(
        self,
        asset: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Retourne les ordres récents.

        Filtres disponibles :
        - asset
        - status
        - limit
        """

        try:

            limit = int(
                limit
            )

        except (
            TypeError,
            ValueError,
        ):

            raise ValueError(
                "limit doit être un entier."
            )

        if limit < 1:

            raise ValueError(
                "limit doit être supérieur ou égal à 1."
            )

        query = """
            SELECT *
            FROM orders
            WHERE 1 = 1
        """

        params: List[Any] = []

        # ----------------------------------------------------
        # ASSET FILTER
        # ----------------------------------------------------

        if asset is not None:

            query += """
                AND asset = ?
            """

            params.append(
                str(asset)
            )

        # ----------------------------------------------------
        # STATUS FILTER
        # ----------------------------------------------------

        if status is not None:

            query += """
                AND status = ?
            """

            params.append(
                str(status)
            )

        # ----------------------------------------------------
        # ORDER / LIMIT
        # ----------------------------------------------------

        query += """
            ORDER BY timestamp DESC
            LIMIT ?
        """

        params.append(
            limit
        )

        with self._get_connection() as conn:

            rows = conn.execute(
                query,
                params,
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    # ========================================================
    # GET ORDER BY ID
    # ========================================================

    def get_order(
        self,
        order_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retourne un ordre précis.
        """

        if not order_id:

            return None

        with self._get_connection() as conn:

            row = conn.execute(
                """
                SELECT *
                FROM orders
                WHERE order_id = ?
                """,

                (
                    str(order_id),
                ),
            ).fetchone()

        if row is None:

            return None

        return dict(
            row
        )

    # ========================================================
    # SAVE SIGNAL
    # ========================================================

    def save_signal(
        self,
        asset: str,
        signal: str,
        confidence: float,
        setup_quality: str,
        indicators: Dict[str, Any],
    ) -> None:
        """
        Enregistre un signal produit par
        le moteur d'analyse.
        """

        if not asset:

            raise ValueError(
                "asset est obligatoire."
            )

        if not signal:

            raise ValueError(
                "signal est obligatoire."
            )

        try:

            confidence_value = float(
                confidence
            )

        except (
            TypeError,
            ValueError,
        ):

            raise ValueError(
                "confidence doit être numérique."
            )

        if not isinstance(
            indicators,
            dict,
        ):

            raise TypeError(
                "indicators doit être un dictionnaire."
            )

        try:

            indicators_json = json.dumps(
                indicators,
                default=str,
            )

        except Exception:

            indicators_json = json.dumps(
                {
                    "raw": str(
                        indicators
                    )
                }
            )

        timestamp = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

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
                    str(asset),

                    str(signal).upper(),

                    confidence_value,

                    str(
                        setup_quality
                    ),

                    indicators_json,

                    timestamp,
                ),
            )

    # ========================================================
    # GET SIGNALS
    # ========================================================

    def get_signals(
        self,
        asset: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Retourne les signaux récents.
        """

        try:

            limit = int(
                limit
            )

        except (
            TypeError,
            ValueError,
        ):

            raise ValueError(
                "limit doit être un entier."
            )

        if limit < 1:

            raise ValueError(
                "limit doit être supérieur ou égal à 1."
            )

        query = """
            SELECT *
            FROM signals
            WHERE 1 = 1
        """

        params: List[Any] = []

        if asset is not None:

            query += """
                AND asset = ?
            """

            params.append(
                str(asset)
            )

        query += """
            ORDER BY timestamp DESC
            LIMIT ?
        """

        params.append(
            limit
        )

        with self._get_connection() as conn:

            rows = conn.execute(
                query,
                params,
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    # ========================================================
    # RECORD TRADE RESULT
    # ========================================================

    def record_trade_result(
        self,
        trade_date: date,
        pnl: float,
        is_win: bool,
    ) -> None:
        """
        Enregistre le résultat d'un trade
        dans les statistiques journalières.
        """

        if not isinstance(
            trade_date,
            date,
        ):

            raise TypeError(
                "trade_date doit être une date."
            )

        try:

            pnl_value = float(
                pnl
            )

        except (
            TypeError,
            ValueError,
        ):

            raise ValueError(
                "pnl doit être numérique."
            )

        date_str = (
            trade_date.isoformat()
        )

        wins_increment = (
            1 if is_win else 0
        )

        losses_increment = (
            0 if is_win else 1
        )

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

                ON CONFLICT(trade_date)
                DO UPDATE SET

                    total_trades =
                        total_trades + 1,

                    wins =
                        wins + excluded.wins,

                    losses =
                        losses + excluded.losses,

                    realized_pnl =
                        realized_pnl
                        + excluded.realized_pnl
                """,

                (
                    date_str,

                    wins_increment,

                    losses_increment,

                    pnl_value,
                ),
            )

    # ========================================================
    # GET DAILY STATS
    # ========================================================

    def get_daily_stats(
        self,
        trade_date: date,
    ) -> Dict[str, Any]:
        """
        Retourne les statistiques d'une journée.
        """

        if not isinstance(
            trade_date,
            date,
        ):

            raise TypeError(
                "trade_date doit être une date."
            )

        date_str = (
            trade_date.isoformat()
        )

        with self._get_connection() as conn:

            row = conn.execute(
                """
                SELECT *
                FROM daily_stats
                WHERE trade_date = ?
                """,

                (
                    date_str,
                ),
            ).fetchone()

        if row is None:

            return {
                "trade_date": date_str,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "realized_pnl": 0.0,
            }

        return dict(
            row
        )

    # ========================================================
    # GET TODAY STATS
    # ========================================================

    def get_today_stats(
        self,
    ) -> Dict[str, Any]:
        """
        Retourne les statistiques du jour UTC.
        """

        today = datetime.now(
            timezone.utc
        ).date()

        return self.get_daily_stats(
            today
        )

    # ========================================================
    # GET TOTAL ORDERS
    # ========================================================

    def get_total_orders(
        self,
    ) -> int:
        """
        Retourne le nombre total d'ordres.
        """

        with self._get_connection() as conn:

            row = conn.execute(
                """
                SELECT COUNT(*) AS total
                FROM orders
                """
            ).fetchone()

        return int(
            row["total"]
        )

    # ========================================================
    # GET DATABASE STATUS
    # ========================================================

    def health_check(
        self,
    ) -> Dict[str, Any]:
        """
        Vérifie que SQLite est accessible.
        """

        try:

            with self._get_connection() as conn:

                conn.execute(
                    "SELECT 1"
                ).fetchone()

            return {
                "status": "ok",
                "database": str(
                    self.db_path
                ),
            }

        except Exception as exc:

            return {
                "status": "error",
                "database": str(
                    self.db_path
                ),
                "error": str(exc),
            }


# ============================================================
# DEFAULT DATA STORE
# ============================================================

data_store = DataStore()


# ============================================================
# PUBLIC HELPER
# ============================================================

def get_data_store() -> DataStore:
    """
    Retourne l'instance globale du DataStore.
    """

    return data_store


# ============================================================
# LOCAL SELF TEST
# ============================================================

if __name__ == "__main__":

    print(
        "========================================"
    )

    print(
        "POCKET AI TRADER"
    )

    print(
        "DATA STORE SELF TEST"
    )

    print(
        "========================================"
    )

    store = DataStore(
        "pocket_ai_trader_test.db"
    )

    print(
        "\nHealth:"
    )

    print(
        store.health_check()
    )

    test_order = {
        "order_id": "DATASTORE-TEST-001",
        "asset": "EURUSD",
        "direction": "BUY",
        "amount": 10.0,
        "timeframe": 60,
        "status": "ACCEPTED",
        "reason": None,
        "connector_response": {
            "mode": "demo",
            "confirmed": True,
        },
        "timestamp": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }

    store.save_order(
        test_order
    )

    print(
        "\nSaved order:"
    )

    print(
        store.get_order(
            "DATASTORE-TEST-001"
        )
    )

    print(
        "\nAll orders:"
    )

    print(
        store.get_orders(
            limit=10
        )
    )

    store.save_signal(
        asset="EURUSD",
        signal="BUY",
        confidence=82.0,
        setup_quality="GOOD",
        indicators={
            "ema9": 1.1350,
            "ema21": 1.1348,
            "rsi14": 61.2,
            "macd": 0.00012,
        },
    )

    print(
        "\nSignals:"
    )

    print(
        store.get_signals(
            limit=10
        )
    )

    store.record_trade_result(
        trade_date=datetime.now(
            timezone.utc
        ).date(),
        pnl=8.5,
        is_win=True,
    )

    print(
        "\nToday stats:"
    )

    print(
        store.get_today_stats()
    )

    print(
        "\nTotal orders:"
    )

    print(
        store.get_total_orders()
    )

    print(
        "\nSELF TEST COMPLETE"
    )
         
