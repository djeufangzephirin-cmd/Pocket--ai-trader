from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from dotenv import load_dotenv


load_dotenv()


def get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
        "oui",
    }


def get_int(
    name: str,
    default: int,
    minimum: int | None = None,
) -> int:
    value = os.getenv(name)

    if value is None:
        result = default
    else:
        try:
            result = int(value)
        except ValueError as exc:
            raise ValueError(
                f"{name} doit être un entier valide."
            ) from exc

    if minimum is not None and result < minimum:
        raise ValueError(
            f"{name} doit être supérieur ou égal à {minimum}."
        )

    return result


def get_float(
    name: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = os.getenv(name)

    if value is None:
        result = default
    else:
        try:
            result = float(value)
        except ValueError as exc:
            raise ValueError(
                f"{name} doit être un nombre valide."
            ) from exc

    if minimum is not None and result < minimum:
        raise ValueError(
            f"{name} doit être supérieur ou égal à {minimum}."
        )

    if maximum is not None and result > maximum:
        raise ValueError(
            f"{name} doit être inférieur ou égal à {maximum}."
        )

    return result


def get_decimal(
    name: str,
    default: str,
    minimum: Decimal | None = None,
) -> Decimal:
    value = os.getenv(name, default)

    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(
            f"{name} doit être un nombre décimal valide."
        ) from exc

    if minimum is not None and result < minimum:
        raise ValueError(
            f"{name} doit être supérieur ou égal à {minimum}."
        )

    return result


# ============================================================
# APPLICATION
# ============================================================

APP_NAME = "Pocket AI Trader"
APP_VERSION = "3.0.0"

TIMEFRAME = os.getenv(
    "TIMEFRAME",
    "1m",
).strip().lower()

TRADING_PAIRS = tuple(
    pair.strip().upper()
    for pair in os.getenv(
        "TRADING_PAIRS",
        "EURUSD,GBPUSD,USDJPY",
    ).split(",")
    if pair.strip()
)


# ============================================================
# STRATEGY
# ============================================================

MIN_CANDLES = get_int(
    "MIN_CANDLES",
    25,
    minimum=25,
)

MIN_CONFIDENCE = get_float(
    "MIN_CONFIDENCE",
    75.0,
    minimum=0.0,
    maximum=100.0,
)

EMA_FAST_PERIOD = get_int(
    "EMA_FAST_PERIOD",
    9,
    minimum=2,
)

EMA_SLOW_PERIOD = get_int(
    "EMA_SLOW_PERIOD",
    21,
    minimum=3,
)

RSI_PERIOD = get_int(
    "RSI_PERIOD",
    14,
    minimum=2,
)

MACD_FAST_PERIOD = get_int(
    "MACD_FAST_PERIOD",
    12,
    minimum=2,
)

MACD_SLOW_PERIOD = get_int(
    "MACD_SLOW_PERIOD",
    26,
    minimum=3,
)

MACD_SIGNAL_PERIOD = get_int(
    "MACD_SIGNAL_PERIOD",
    9,
    minimum=2,
)

BOLLINGER_PERIOD = get_int(
    "BOLLINGER_PERIOD",
    20,
    minimum=2,
)

BOLLINGER_STD_DEV = get_float(
    "BOLLINGER_STD_DEV",
    2.0,
    minimum=0.1,
)


# ============================================================
# RISK MANAGEMENT
# ============================================================

INITIAL_BALANCE = get_decimal(
    "INITIAL_BALANCE",
    "100.00",
    minimum=Decimal("0"),
)

MAX_TRADE_RISK_PERCENT = get_float(
    "MAX_TRADE_RISK_PERCENT",
    1.0,
    minimum=0.01,
    maximum=100.0,
)

DAILY_STOP_LOSS_PERCENT = get_float(
    "DAILY_STOP_LOSS_PERCENT",
    5.0,
    minimum=0.01,
    maximum=100.0,
)

MAX_CONSECUTIVE_LOSSES = get_int(
    "MAX_CONSECUTIVE_LOSSES",
    3,
    minimum=1,
)


# ============================================================
# EXECUTION
# ============================================================

AUTO_EXECUTE = get_bool(
    "AUTO_EXECUTE",
    False,
)

ORDER_DURATION_SECONDS = get_int(
    "ORDER_DURATION_SECONDS",
    60,
    minimum=1,
)


# ============================================================
# POCKET OPTION
# ============================================================

PO_BASE_URL = os.getenv(
    "PO_BASE_URL",
    "https://pocketoption.com",
).strip()

PO_EMAIL = os.getenv(
    "PO_EMAIL",
    "",
).strip()

PO_PASSWORD = os.getenv(
    "PO_PASSWORD",
    "",
)

PO_SESSION = os.getenv(
    "PO_SESSION",
    "",
).strip()

PO_PROFILE_DIR = os.getenv(
    "PO_PROFILE_DIR",
    "./browser_profile",
).strip()

PO_WEBSOCKET_URL = os.getenv(
    "PO_WEBSOCKET_URL",
    "",
).strip()


# ============================================================
# PLAYWRIGHT
# ============================================================

PLAYWRIGHT_HEADLESS = get_bool(
    "PLAYWRIGHT_HEADLESS",
    True,
)

PLAYWRIGHT_TIMEOUT_MS = get_int(
    "PLAYWRIGHT_TIMEOUT_MS",
    30000,
    minimum=1000,
)


# ============================================================
# LOGGING
# ============================================================

LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO",
).strip().upper()

LOG_FILE = os.getenv(
    "LOG_FILE",
    "pocket_ai_trader.log",
).strip()


# ============================================================
# VALIDATION
# ============================================================

if not TRADING_PAIRS:
    raise ValueError(
        "TRADING_PAIRS doit contenir au moins une paire."
    )

if EMA_FAST_PERIOD >= EMA_SLOW_PERIOD:
    raise ValueError(
        "EMA_FAST_PERIOD doit être inférieur à EMA_SLOW_PERIOD."
    )

if MACD_FAST_PERIOD >= MACD_SLOW_PERIOD:
    raise ValueError(
        "MACD_FAST_PERIOD doit être inférieur à MACD_SLOW_PERIOD."
    )

if not PO_EMAIL and not PO_SESSION:
    print(
        "AVERTISSEMENT : aucune session Pocket Option "
        "ni adresse e-mail configurée. "
        "L'analyse locale reste disponible."
    )


# ============================================================
# CONFIGURATION STRUCTURÉE
# ============================================================

@dataclass(frozen=True)
class StrategyConfig:
    min_candles: int
    min_confidence: float
    ema_fast_period: int
    ema_slow_period: int
    rsi_period: int
    macd_fast_period: int
    macd_slow_period: int
    macd_signal_period: int
    bollinger_period: int
    bollinger_std_dev: float


@dataclass(frozen=True)
class RiskConfig:
    initial_balance: Decimal
    max_trade_risk_percent: float
    daily_stop_loss_percent: float
    max_consecutive_losses: int


@dataclass(frozen=True)
class ExecutionConfig:
    auto_execute: bool
    order_duration_seconds: int


@dataclass(frozen=True)
class PocketOptionConfig:
    base_url: str
    email: str
    password: str
    session: str
    profile_dir: str
    websocket_url: str
    playwright_headless: bool
    playwright_timeout_ms: int


STRATEGY_CONFIG = StrategyConfig(
    min_candles=MIN_CANDLES,
    min_confidence=MIN_CONFIDENCE,
    ema_fast_period=EMA_FAST_PERIOD,
    ema_slow_period=EMA_SLOW_PERIOD,
    rsi_period=RSI_PERIOD,
    macd_fast_period=MACD_FAST_PERIOD,
    macd_slow_period=MACD_SLOW_PERIOD,
    macd_signal_period=MACD_SIGNAL_PERIOD,
    bollinger_period=BOLLINGER_PERIOD,
    bollinger_std_dev=BOLLINGER_STD_DEV,
)


RISK_CONFIG = RiskConfig(
    initial_balance=INITIAL_BALANCE,
    max_trade_risk_percent=MAX_TRADE_RISK_PERCENT,
    daily_stop_loss_percent=DAILY_STOP_LOSS_PERCENT,
    max_consecutive_losses=MAX_CONSECUTIVE_LOSSES,
)


EXECUTION_CONFIG = ExecutionConfig(
    auto_execute=AUTO_EXECUTE,
    order_duration_seconds=ORDER_DURATION_SECONDS,
)


POCKET_OPTION_CONFIG = PocketOptionConfig(
    base_url=PO_BASE_URL,
    email=PO_EMAIL,
    password=PO_PASSWORD,
    session=PO_SESSION,
    profile_dir=PO_PROFILE_DIR,
    websocket_url=PO_WEBSOCKET_URL,
    playwright_headless=PLAYWRIGHT_HEADLESS,
    playwright_timeout_ms=PLAYWRIGHT_TIMEOUT_MS,
)
