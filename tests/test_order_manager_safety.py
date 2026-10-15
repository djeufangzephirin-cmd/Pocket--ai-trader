"""
Garde-fous du gestionnaire d'ordres.

Objectif : prouver qu'aucun ordre ne peut atteindre le connecteur lorsqu'une
règle de risque n'est pas satisfaite, et qu'une panne de persistance bloque
l'envoi au lieu de le laisser passer.

Aucun réseau, aucune authentification Pocket Option, aucune exécution réelle :
le connecteur est remplacé par un enregistreur local.
"""

from datetime import datetime, timedelta, timezone

import pytest

from order_manager import (
    DEFAULT_MIN_PAYOUT_PERCENT,
    Order,
    OrderManager,
    break_even_win_rate,
)


# ============================================================
# DOUBLES DE TEST
# ============================================================

class RecordingConnector:
    """Connecteur factice qui enregistre chaque ordre reçu."""

    def __init__(self, result=True, raises=None, mode="demo"):
        self.result = result
        self.raises = raises
        self.mode = mode
        self.calls = []
        self.last_execution = None

    def execute_order(self, order):
        self.calls.append(order)

        if self.raises is not None:
            raise self.raises

        if self.result:
            self.last_execution = {
                "order_id": order.order_id,
                "confirmed": True,
                "mode": self.mode,
            }

        return self.result


class FakeStore:
    """DataStore minimal, sans SQLite."""

    def __init__(self, pnl=0.0, raise_on_stats=False, raise_on_save=False):
        self.pnl = pnl
        self.raise_on_stats = raise_on_stats
        self.raise_on_save = raise_on_save
        self.saved = []

    def get_daily_stats(self, trade_date):
        if self.raise_on_stats:
            raise RuntimeError("base illisible")
        return {"realized_pnl": self.pnl}

    def save_order(self, order_record):
        if self.raise_on_save:
            raise RuntimeError("écriture refusée")
        self.saved.append(dict(order_record))


VALID_ORDER = {
    "asset": "EURUSD",
    "direction": "BUY",
    "amount": 10.0,
    "payout": 85.0,
    "expiration_seconds": 60,
}


def submit(manager, **overrides):
    payload = dict(VALID_ORDER)
    payload.update(overrides)
    return manager.submit_order(**payload)


# ============================================================
# CONSTRUCTION
# ============================================================

def test_manager_refuses_to_be_built_without_connector():
    with pytest.raises(ValueError):
        OrderManager(connector=None)


def test_valid_order_is_accepted_and_transmitted_once():
    connector = RecordingConnector()
    manager = OrderManager(connector=connector)

    result = submit(manager)

    assert result["status"] == "ACCEPTED"
    assert result["executed"] is True
    assert len(connector.calls) == 1


def test_connector_receives_a_complete_order_object():
    connector = RecordingConnector()
    manager = OrderManager(connector=connector)

    submit(manager)

    order = connector.calls[0]

    assert isinstance(order, Order)
    assert order.asset == "EURUSD"
    assert order.direction == "BUY"
    assert order.amount == 10.0
    assert order.payout == 85.0
    assert order.expiration_seconds == 60
    assert order.order_id


def test_connector_rejection_is_reported_as_failed():
    connector = RecordingConnector(result=False)
    manager = OrderManager(connector=connector)

    result = submit(manager)

    assert result["status"] == "FAILED"
    assert len(connector.calls) == 1


# ============================================================
# SEUIL DE PAYOUT
# ============================================================

@pytest.mark.parametrize("payout", [79.9, 50.0, 0.0, -10.0, None, "abc"])
def test_low_or_invalid_payout_is_rejected(payout):
    connector = RecordingConnector()
    manager = OrderManager(connector=connector)

    result = submit(manager, payout=payout)

    assert result["status"] == "REJECTED"
    assert connector.calls == []


def test_payout_floor_is_inclusive():
    connector = RecordingConnector()
    manager = OrderManager(connector=connector)

    result = submit(manager, payout=DEFAULT_MIN_PAYOUT_PERCENT)

    assert result["status"] == "ACCEPTED"


def test_payout_rejection_mentions_break_even_rate():
    connector = RecordingConnector()
    manager = OrderManager(connector=connector)

    result = submit(manager, payout=70.0)

    assert result["status"] == "REJECTED"
    assert "70" in result["reason"]
    assert "58.82" in result["reason"]


# ============================================================
# EXPIRATION
# ============================================================

@pytest.mark.parametrize("expiration", [59, 1, 0, -60, "abc", ""])
def test_short_or_invalid_expiration_is_rejected(expiration):
    connector = RecordingConnector()
    manager = OrderManager(connector=connector)

    result = submit(manager, expiration_seconds=expiration)

    assert result["status"] == "REJECTED"
    assert connector.calls == []


def test_absent_expiration_falls_back_to_the_minimum():
    """Une expiration non fournie prend le minimum autorisé, pas plus."""

    connector = RecordingConnector()
    manager = OrderManager(connector=connector)

    result = submit(manager, expiration_seconds=None)

    assert result["status"] == "ACCEPTED"
    assert connector.calls[0].expiration_seconds == 60


def test_expiration_floor_is_inclusive():
    connector = RecordingConnector()
    manager = OrderManager(connector=connector)

    result = submit(manager, expiration_seconds=60)

    assert result["status"] == "ACCEPTED"


def test_timeframe_is_accepted_as_expiration_alias():
    connector = RecordingConnector()
    manager = OrderManager(connector=connector)

    result = manager.submit_order(
        asset="EURUSD",
        direction="BUY",
        amount=10.0,
        payout=85.0,
        timeframe=120,
    )

    assert result["status"] == "ACCEPTED"
    assert connector.calls[0].expiration_seconds == 120


# ============================================================
# SENS
# ============================================================

@pytest.mark.parametrize("direction", ["CALL", "PUT", "LONG", "SHORT", "", None, 1])
def test_invalid_direction_is_rejected(direction):
    connector = RecordingConnector()
    manager = OrderManager(connector=connector)

    result = submit(manager, direction=direction)

    assert result["status"] == "REJECTED"
    assert connector.calls == []


@pytest.mark.parametrize("direction", ["buy", " sell ", "BuY"])
def test_direction_is_normalized_before_execution(direction):
    connector = RecordingConnector()
    manager = OrderManager(connector=connector)

    result = submit(manager, direction=direction)

    assert result["status"] == "ACCEPTED"
    assert connector.calls[0].direction in {"BUY", "SELL"}


def test_sell_is_accepted():
    connector = RecordingConnector()
    manager = OrderManager(connector=connector)

    result = submit(manager, direction="SELL")

    assert result["status"] == "ACCEPTED"
    assert connector.calls[0].direction == "SELL"


# ============================================================
# MONTANT
# ============================================================

@pytest.mark.parametrize("amount", [0, -1, -0.01, None, "abc", True])
def test_invalid_amount_is_rejected(amount):
    connector = RecordingConnector()
    manager = OrderManager(connector=connector)

    result = submit(manager, amount=amount)

    assert result["status"] == "REJECTED"
    assert connector.calls == []


def test_position_size_cap_is_inclusive():
    connector = RecordingConnector()
    manager = OrderManager(connector=connector, max_position_size=50.0)

    assert submit(manager, amount=50.0)["status"] == "ACCEPTED"


def test_position_size_cap_rejects_above_limit():
    connector = RecordingConnector()
    manager = OrderManager(connector=connector, max_position_size=50.0)

    result = submit(manager, amount=50.01)

    assert result["status"] == "REJECTED"
    assert connector.calls == []


def test_position_size_cap_can_be_disabled_explicitly():
    connector = RecordingConnector()
    manager = OrderManager(connector=connector, max_position_size=None)

    result = submit(manager, amount=10_000.0)

    assert result["status"] == "ACCEPTED"


# ============================================================
# ARRÊT D'URGENCE
# ============================================================

def test_emergency_stop_blocks_every_order():
    connector = RecordingConnector()
    manager = OrderManager(connector=connector)

    manager.trigger_emergency_stop("Anomalie détectée")

    result = submit(manager)

    assert result["status"] == "REJECTED"
    assert "urgence" in result["reason"].lower()
    assert connector.calls == []


def test_emergency_stop_reason_is_kept_in_status():
    connector = RecordingConnector()
    manager = OrderManager(connector=connector)

    manager.trigger_emergency_stop("Perte anormale")

    status = manager.get_status()

    assert status["emergency_stop_triggered"] is True
    assert status["emergency_stop_reason"] == "Perte anormale"


def test_reset_emergency_stop_restores_execution():
    connector = RecordingConnector()
    manager = OrderManager(connector=connector)

    manager.trigger_emergency_stop()
    assert submit(manager)["status"] == "REJECTED"

    manager.reset_emergency_stop()

    assert submit(manager)["status"] == "ACCEPTED"
    assert len(connector.calls) == 1


# ============================================================
# DOUBLONS
# ============================================================

def test_identical_order_within_window_is_rejected():
    connector = RecordingConnector()
    manager = OrderManager(connector=connector)

    first = submit(manager)
    second = submit(manager)

    assert first["status"] == "ACCEPTED"
    assert second["status"] == "REJECTED"
    assert "doublon" in second["reason"].lower()
    assert len(connector.calls) == 1


def test_duplicate_detection_ignores_opposite_direction():
    connector = RecordingConnector()
    manager = OrderManager(connector=connector)

    submit(manager, direction="BUY")
    result = submit(manager, direction="SELL")

    assert result["status"] == "ACCEPTED"
    assert len(connector.calls) == 2


def test_duplicate_detection_ignores_other_asset():
    connector = RecordingConnector()
    manager = OrderManager(connector=connector)

    submit(manager, asset="EURUSD")
    result = submit(manager, asset="GBPUSD")

    assert result["status"] == "ACCEPTED"
    assert len(connector.calls) == 2


def test_order_is_allowed_once_the_window_has_elapsed():
    connector = RecordingConnector()
    manager = OrderManager(
        connector=connector,
        duplicate_window_seconds=10,
    )

    submit(manager)

    # Vieillit artificiellement l'ordre accepté de 60 secondes.
    manager.order_history[0]["timestamp"] = (
        datetime.now(timezone.utc) - timedelta(seconds=60)
    ).isoformat()

    result = submit(manager)

    assert result["status"] == "ACCEPTED"
    assert len(connector.calls) == 2


def test_rejected_order_does_not_block_a_later_valid_order():
    connector = RecordingConnector()
    manager = OrderManager(connector=connector)

    assert submit(manager, payout=10.0)["status"] == "REJECTED"
    assert submit(manager)["status"] == "ACCEPTED"
    assert len(connector.calls) == 1


# ============================================================
# PERTE JOURNALIÈRE
# ============================================================

def test_daily_loss_below_cap_allows_execution():
    store = FakeStore(pnl=-40.0)
    connector = RecordingConnector()
    manager = OrderManager(
        connector=connector,
        data_store=store,
        max_daily_loss=100.0,
    )

    assert submit(manager)["status"] == "ACCEPTED"


def test_daily_loss_at_cap_blocks_execution():
    store = FakeStore(pnl=-100.0)
    connector = RecordingConnector()
    manager = OrderManager(
        connector=connector,
        data_store=store,
        max_daily_loss=100.0,
    )

    result = submit(manager)

    assert result["status"] == "REJECTED"
    assert "perte journali" in result["reason"].lower()
    assert connector.calls == []


def test_daily_loss_above_cap_blocks_execution():
    store = FakeStore(pnl=-250.0)
    connector = RecordingConnector()
    manager = OrderManager(
        connector=connector,
        data_store=store,
        max_daily_loss=100.0,
    )

    assert submit(manager)["status"] == "REJECTED"
    assert connector.calls == []


def test_unreadable_daily_loss_fails_closed():
    store = FakeStore(raise_on_stats=True)
    connector = RecordingConnector()
    manager = OrderManager(
        connector=connector,
        data_store=store,
    )

    result = submit(manager)

    assert result["status"] == "REJECTED"
    assert "illisible" in result["reason"].lower()
    assert connector.calls == []


# ============================================================
# PERSISTANCE AVANT EXÉCUTION
# ============================================================

def test_rejection_is_persisted_without_execution():
    store = FakeStore()
    connector = RecordingConnector()
    manager = OrderManager(
        connector=connector,
        data_store=store,
    )

    submit(manager, payout=10.0)

    assert connector.calls == []
    assert len(store.saved) == 1
    assert store.saved[0]["status"] == "REJECTED"


def test_persistence_failure_prevents_execution():
    store = FakeStore(raise_on_save=True)
    connector = RecordingConnector()
    manager = OrderManager(
        connector=connector,
        data_store=store,
    )

    result = submit(manager)

    assert result["status"] == "FAILED"
    assert result["executed"] is False
    assert "non envoyé" in result["reason"]
    assert connector.calls == []


def test_accepted_order_is_persisted_as_submitted_then_accepted():
    """
    L'ordre est écrit AVANT l'exécution (SUBMITTED), puis réécrit avec le
    résultat (ACCEPTED). Les deux écritures portent le même order_id, donc
    la ligne finale reste unique en base.
    """

    store = FakeStore()
    connector = RecordingConnector()
    manager = OrderManager(
        connector=connector,
        data_store=store,
    )

    result = submit(manager)

    assert result["status"] == "ACCEPTED"
    assert [row["status"] for row in store.saved] == [
        "SUBMITTED",
        "ACCEPTED",
    ]
    assert {row["order_id"] for row in store.saved} == {
        result["order_id"]
    }


def test_failed_execution_is_persisted_as_failed():
    store = FakeStore()
    connector = RecordingConnector(result=False)
    manager = OrderManager(
        connector=connector,
        data_store=store,
    )

    result = submit(manager)

    assert result["status"] == "FAILED"
    assert store.saved[-1]["status"] == "FAILED"


# ============================================================
# EXCEPTIONS DU CONNECTEUR
# ============================================================

def test_connector_exception_becomes_failed_response():
    connector = RecordingConnector(
        raises=RuntimeError("socket fermée")
    )
    manager = OrderManager(connector=connector)

    result = submit(manager)

    assert result["status"] == "FAILED"
    assert "socket fermée" in result["reason"]
    assert result["executed"] is True


def test_connector_exception_is_never_swallowed_silently():
    connector = RecordingConnector(
        raises=RuntimeError("client absent")
    )
    store = FakeStore()
    manager = OrderManager(
        connector=connector,
        data_store=store,
    )

    submit(manager)

    assert len(connector.calls) == 1
    assert store.saved[-1]["status"] == "FAILED"


# ============================================================
# BALAYAGE : AUCUN ENVOI SUR REFUS
# ============================================================

@pytest.mark.parametrize(
    "overrides",
    [
        {"asset": "", "amount": 10.0},
        {"direction": "CALL"},
        {"amount": 0},
        {"amount": 999},
        {"payout": 0},
        {"payout": 79},
        {"expiration_seconds": 30},
        {"expiration_seconds": "soixante"},
        {"amount": "dix"},
    ],
)
def test_no_order_reaches_connector_on_any_rejection(overrides):
    connector = RecordingConnector()
    manager = OrderManager(
        connector=connector,
        max_position_size=100.0,
    )

    result = submit(manager, **overrides)

    assert result["status"] == "REJECTED"
    assert connector.calls == []


# ============================================================
# INTERFACE DICT
# ============================================================

def test_execute_order_wrapper_accepts_valid_dict():
    connector = RecordingConnector()
    manager = OrderManager(connector=connector)

    result = manager.execute_order(dict(VALID_ORDER))

    assert result["status"] == "ACCEPTED"
    assert len(connector.calls) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"asset": "EURUSD"},
        {"asset": "EURUSD", "direction": "BUY"},
        {"direction": "BUY", "amount": 10.0},
        {"asset": "EURUSD", "amount": 10.0},
    ],
)
def test_execute_order_wrapper_rejects_incomplete_dict(payload):
    connector = RecordingConnector()
    manager = OrderManager(connector=connector)

    result = manager.execute_order(payload)

    assert result["status"] == "REJECTED"
    assert connector.calls == []


def test_execute_order_wrapper_rejects_non_dict():
    connector = RecordingConnector()
    manager = OrderManager(connector=connector)

    result = manager.execute_order(["EURUSD", "BUY", 10.0])

    assert result["status"] == "REJECTED"
    assert connector.calls == []


# ============================================================
# SEUIL DE RENTABILITÉ
# ============================================================

@pytest.mark.parametrize(
    "payout,expected",
    [
        (0, 100.0),
        (50, 66.666666),
        (80, 55.555555),
        (92, 52.083333),
        (100, 50.0),
    ],
)
def test_break_even_win_rate(payout, expected):
    assert break_even_win_rate(payout) == pytest.approx(
        expected,
        abs=1e-5,
    )


def test_break_even_win_rate_handles_invalid_input():
    assert break_even_win_rate(None) == 100.0
    assert break_even_win_rate("abc") == 100.0


# ============================================================
# ÉTAT
# ============================================================

def test_get_status_exposes_guardrails():
    connector = RecordingConnector()
    manager = OrderManager(
        connector=connector,
        max_position_size=25.0,
        max_daily_loss=200.0,
    )

    status = manager.get_status()

    assert status["max_position_size"] == 25.0
    assert status["max_daily_loss"] == 200.0
    assert status["min_payout_percent"] == DEFAULT_MIN_PAYOUT_PERCENT
    assert status["min_expiration_seconds"] == 60
    assert status["emergency_stop_triggered"] is False
    assert status["data_store_attached"] is False
    assert status["persistence_before_execution"] is True


def test_get_status_counts_only_accepted_orders():
    connector = RecordingConnector()
    manager = OrderManager(connector=connector)

    submit(manager, payout=10.0)
    submit(manager)

    assert manager.get_status()["accepted_orders_in_memory"] == 1


def test_health_check_matches_get_status():
    manager = OrderManager(connector=RecordingConnector())

    assert manager.health_check() == manager.get_status()
