"""
Chaîne complète : OrderManager -> DataStore SQLite réel -> connecteur réel.

Ces tests vérifient le contrat de bout en bout, et notamment la règle
structurante du gestionnaire : la ligne d'ordre existe en base AVANT que le
connecteur ne soit appelé.

Aucun réseau, aucune authentification Pocket Option, aucune exécution réelle :
le connecteur est instancié en mode démo et ne place jamais d'ordre.
"""

from datetime import datetime, timezone

import pytest

from data_store import DataStore
from order_manager import OrderManager
from po_connector import PocketOptionConnector


VALID_ORDER = {
    "asset": "EURUSD",
    "direction": "BUY",
    "amount": 10.0,
    "payout": 85.0,
    "expiration_seconds": 60,
}


# ============================================================
# DOUBLES DE TEST
# ============================================================

class InspectingConnector(PocketOptionConnector):
    """
    Connecteur démo réel qui observe la base juste avant d'exécuter.

    C'est la preuve directe de la persistance avant exécution : au moment de
    l'appel, la ligne d'ordre doit déjà porter le statut SUBMITTED.
    """

    def __init__(self, store, **kwargs):
        super().__init__(**kwargs)
        self.store = store
        self.status_seen_at_execution = None
        self.row_seen_at_execution = None

    def execute_order(self, order):
        rows = self.store.get_orders()

        if rows:
            self.row_seen_at_execution = rows[0]
            self.status_seen_at_execution = rows[0]["status"]

        return super().execute_order(order)


class SilentConnector(PocketOptionConnector):
    """Connecteur démo qui ne confirme jamais l'ordre."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = 0

    def execute_order(self, order):
        self.calls += 1
        return False


class ExplodingConnector(PocketOptionConnector):
    """Connecteur qui échoue au milieu de l'exécution."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = 0

    def execute_order(self, order):
        self.calls += 1
        raise RuntimeError("transport interrompu")


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def store(tmp_path):
    return DataStore(db_path=str(tmp_path / "integration.db"))


@pytest.fixture
def demo_connector():
    return PocketOptionConnector(
        mode="demo",
        trading_enabled=False,
        live_enabled=False,
    )


@pytest.fixture
def manager(store, demo_connector):
    return OrderManager(
        connector=demo_connector,
        data_store=store,
        max_position_size=50.0,
        max_daily_loss=100.0,
    )


def submit(manager, **overrides):
    payload = dict(VALID_ORDER)
    payload.update(overrides)
    return manager.submit_order(**payload)


def today():
    return datetime.now(timezone.utc).date()


# ============================================================
# BOUT EN BOUT EN DÉMO
# ============================================================

def test_demo_order_is_accepted_and_persisted(manager, store):
    result = submit(manager)

    assert result["status"] == "ACCEPTED"

    rows = store.get_orders()

    assert len(rows) == 1
    assert rows[0]["order_id"] == result["order_id"]
    assert rows[0]["asset"] == "EURUSD"
    assert rows[0]["direction"] == "BUY"
    assert rows[0]["amount"] == 10.0
    assert rows[0]["timeframe"] == 60
    assert rows[0]["status"] == "ACCEPTED"


def test_order_row_exists_before_connector_execution(store):
    connector = InspectingConnector(
        store,
        mode="demo",
        trading_enabled=False,
        live_enabled=False,
    )

    manager = OrderManager(
        connector=connector,
        data_store=store,
    )

    result = submit(manager)

    assert result["status"] == "ACCEPTED"
    assert connector.status_seen_at_execution == "SUBMITTED"


def test_submitted_row_carries_payout_and_stage(store):
    connector = InspectingConnector(
        store,
        mode="demo",
        trading_enabled=False,
        live_enabled=False,
    )

    manager = OrderManager(
        connector=connector,
        data_store=store,
    )

    submit(manager)

    row = connector.row_seen_at_execution

    assert row is not None
    assert "pre_execution" in row["result_json"]
    assert "85.0" in row["result_json"]


def test_execution_leaves_a_single_final_row(manager, store):
    result = submit(manager)

    rows = store.get_orders()

    assert len(rows) == 1
    assert rows[0]["status"] == "ACCEPTED"

    stored = store.get_order(result["order_id"])

    assert stored is not None
    assert stored["status"] == "ACCEPTED"


def test_two_distinct_orders_produce_two_rows(manager, store):
    first = submit(manager, asset="EURUSD")
    second = submit(manager, asset="GBPUSD")

    assert first["status"] == "ACCEPTED"
    assert second["status"] == "ACCEPTED"

    rows = store.get_orders()

    assert len(rows) == 2
    assert {row["asset"] for row in rows} == {"EURUSD", "GBPUSD"}


def test_execute_order_dict_wrapper_end_to_end(manager, store):
    result = manager.execute_order(dict(VALID_ORDER))

    assert result["status"] == "ACCEPTED"
    assert store.get_order(result["order_id"]) is not None


def test_orders_are_queryable_by_status(manager, store):
    submit(manager)
    submit(manager, payout=10.0)

    accepted = store.get_orders(status="ACCEPTED")
    rejected = store.get_orders(status="REJECTED")

    assert len(accepted) == 1
    assert len(rejected) == 1


# ============================================================
# ÉCHECS D'EXÉCUTION
# ============================================================

def test_connector_refusal_is_persisted_as_failed(store):
    connector = SilentConnector(
        mode="demo",
        trading_enabled=False,
        live_enabled=False,
    )

    manager = OrderManager(
        connector=connector,
        data_store=store,
    )

    result = submit(manager)

    assert result["status"] == "FAILED"
    assert connector.calls == 1

    rows = store.get_orders()

    assert len(rows) == 1
    assert rows[0]["status"] == "FAILED"
    assert rows[0]["reason"]


def test_connector_exception_is_persisted_as_failed(store):
    connector = ExplodingConnector(
        mode="demo",
        trading_enabled=False,
        live_enabled=False,
    )

    manager = OrderManager(
        connector=connector,
        data_store=store,
    )

    result = submit(manager)

    assert result["status"] == "FAILED"
    assert "transport interrompu" in result["reason"]

    rows = store.get_orders()

    assert len(rows) == 1
    assert rows[0]["status"] == "FAILED"
    assert "transport interrompu" in rows[0]["reason"]


def test_live_mode_without_flags_fails_closed(store):
    """Le mode live sans les trois drapeaux ne doit jamais exécuter."""

    connector = PocketOptionConnector(
        mode="live",
        trading_enabled=False,
        live_enabled=False,
    )

    manager = OrderManager(
        connector=connector,
        data_store=store,
    )

    result = submit(manager)

    assert result["status"] == "FAILED"
    assert "LIVE" in result["reason"] or "live" in result["reason"]
    assert connector.last_execution is None

    rows = store.get_orders()

    assert len(rows) == 1
    assert rows[0]["status"] == "FAILED"


def test_live_mode_without_client_fails_closed(store):
    connector = PocketOptionConnector(
        mode="live",
        trading_enabled=True,
        live_enabled=True,
    )

    manager = OrderManager(
        connector=connector,
        data_store=store,
    )

    result = submit(manager)

    assert result["status"] == "FAILED"
    assert connector.last_execution is None


# ============================================================
# PERTE JOURNALIÈRE SUR BASE RÉELLE
# ============================================================

def test_daily_loss_from_real_database_blocks_new_orders(
    manager,
    store,
    demo_connector,
):
    store.record_trade_result(today(), -150.0, is_win=False)

    result = submit(manager)

    assert result["status"] == "REJECTED"
    assert "perte journali" in result["reason"].lower()
    assert demo_connector.last_execution is None

    rows = store.get_orders()

    assert len(rows) == 1
    assert rows[0]["status"] == "REJECTED"


def test_daily_loss_below_cap_allows_order(manager, store):
    store.record_trade_result(today(), -40.0, is_win=False)

    result = submit(manager)

    assert result["status"] == "ACCEPTED"
    assert len(store.get_orders(status="ACCEPTED")) == 1


def test_status_reports_daily_loss_from_database(manager, store):
    store.record_trade_result(today(), -150.0, is_win=False)

    status = manager.get_status()

    assert status["daily_loss"] == pytest.approx(150.0)
    assert status["data_store_attached"] is True


def test_profitable_day_does_not_reduce_loss_counter(manager, store):
    store.record_trade_result(today(), 200.0, is_win=True)

    status = manager.get_status()

    assert status["daily_loss"] == 0.0


# ============================================================
# ARRÊT D'URGENCE SUR BASE RÉELLE
# ============================================================

def test_emergency_stop_blocks_and_is_audited(manager, store, demo_connector):
    manager.trigger_emergency_stop("Test d'intégration")

    result = submit(manager)

    assert result["status"] == "REJECTED"
    assert "urgence" in result["reason"].lower()
    assert demo_connector.last_execution is None

    rows = store.get_orders()

    assert len(rows) == 1
    assert rows[0]["status"] == "REJECTED"

    manager.reset_emergency_stop()

    assert submit(manager)["status"] == "ACCEPTED"


# ============================================================
# PISTE D'AUDIT
# ============================================================

def test_rejected_order_row_keeps_submitted_parameters(manager, store):
    submit(
        manager,
        asset="TSLA_otc",
        direction="SELL",
        amount=25.0,
        payout=40.0,
    )

    rows = store.get_orders()

    assert len(rows) == 1
    assert rows[0]["asset"] == "TSLA_otc"
    assert rows[0]["direction"] == "SELL"
    assert rows[0]["amount"] == 25.0
    assert rows[0]["status"] == "REJECTED"


def test_pocket_option_symbol_case_is_preserved(manager, store, demo_connector):
    """
    Les symboles Pocket Option sont sensibles à la casse : "#TSLA_otc" ne doit
    pas devenir "#TSLA_OTC" en traversant le gestionnaire.
    """

    result = submit(manager, asset="#TSLA_otc")

    assert result["status"] == "ACCEPTED"
    assert result["order"]["asset"] == "#TSLA_otc"

    rows = store.get_orders()

    assert rows[0]["asset"] == "#TSLA_otc"

    execution = demo_connector.last_execution

    assert execution is not None
    assert execution["asset"] == "#TSLA_otc"


def test_asset_whitespace_is_trimmed(manager, store):
    result = submit(manager, asset="  #AAPL_otc  ")

    assert result["status"] == "ACCEPTED"
    assert result["order"]["asset"] == "#AAPL_otc"


def test_duplicate_detection_is_case_sensitive_on_asset(manager, store):
    """Deux symboles de casse différente sont deux actifs distincts."""

    first = submit(manager, asset="#TSLA_otc")
    second = submit(manager, asset="#TSLA_OTC")

    assert first["status"] == "ACCEPTED"
    assert second["status"] == "ACCEPTED"
    assert len(store.get_orders()) == 2


def test_every_attempt_is_audited(manager, store):
    """Une séquence mixte laisse une ligne par tentative, jamais zéro."""

    submit(manager, asset="EURUSD")
    submit(manager, asset="EURUSD")          # doublon
    submit(manager, asset="GBPUSD", payout=10.0)  # payout refusé
    submit(manager, asset="USDJPY", amount=900.0)  # taille refusée

    rows = store.get_orders(limit=50)

    assert len(rows) == 4
    assert len([r for r in rows if r["status"] == "ACCEPTED"]) == 1
    assert len([r for r in rows if r["status"] == "REJECTED"]) == 3


def test_connector_response_is_recorded_for_accepted_order(manager, store):
    result = submit(manager)

    stored = store.get_order(result["order_id"])

    assert stored is not None
    assert "confirmed" in stored["result_json"]
    assert "true" in stored["result_json"].lower()
