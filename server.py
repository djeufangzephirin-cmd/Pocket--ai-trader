import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# POCKET AI TRADER - ENGINE v4.1.0
# ============================================================

APP_VERSION = "4.1.0"

ASSET = "EURUSD"
TIMEFRAME = "15m"

DEFAULT_CANDLES = 100
MIN_CANDLES = 30
MAX_CANDLES = 500

TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "")

# ============================================================
# SCORING CONFIGURATION
# ============================================================

EMA_SEPARATION_THRESHOLD = 0.0005

# ============================================================
# RISK MANAGEMENT CONFIGURATION
# ============================================================

DEFAULT_ACCOUNT_BALANCE = 1000.0
DEFAULT_RISK_PERCENT = 1.0

MIN_RISK_PERCENT = 0.1
MAX_RISK_PERCENT = 2.0

PIP_SIZE_EURUSD = 0.0001
PIP_VALUE_PER_STANDARD_LOT = 10.0

# ============================================================
# TRADE PLAN CONFIGURATION
# ============================================================

ATR_STOP_MULTIPLIER = 1.5
MIN_RR_TP1 = 1.0
MIN_RR_TP2 = 1.5


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Pocket AI Trader",
    description="Moteur d'analyse technique Forex v4.1.0",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class AnalyzeRequest(BaseModel):
    asset: str = ASSET
    timeframe: str = TIMEFRAME

    candles: int = Field(
        default=DEFAULT_CANDLES,
        ge=MIN_CANDLES,
        le=MAX_CANDLES,
    )

    account_balance: float = Field(
        default=DEFAULT_ACCOUNT_BALANCE,
        gt=0,
    )

    risk_percent: float = Field(
        default=DEFAULT_RISK_PERCENT,
    )


class TradePlanRequest(BaseModel):
    signal: str
    price: float = Field(gt=0)
    atr: float = Field(gt=0)


class RiskRequest(BaseModel):
    account_balance: float = Field(
        default=DEFAULT_ACCOUNT_BALANCE,
        gt=0,
    )

    risk_percent: float = Field(
        default=DEFAULT_RISK_PERCENT,
    )

    entry: float = Field(gt=0)
    stop_loss: float = Field(gt=0)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def normalize_asset(asset: str) -> str:
    asset = asset.strip().upper()

    aliases = {
        "EURUSD": "EUR/USD",
        "EUR/USD": "EUR/USD",
    }

    return aliases.get(asset, asset)


def normalize_interval(timeframe: str) -> str:
    value = str(
        timeframe or TIMEFRAME
    ).strip().lower()

    mapping = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "45m": "45min",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "1d": "1day",
    }

    return mapping.get(value, value)


# ============================================================
# RISK VALIDATION
# ============================================================

def validate_risk_percent(
    risk_percent: float,
) -> float:

    risk_percent = safe_float(risk_percent)

    if (
        risk_percent < MIN_RISK_PERCENT
        or risk_percent > MAX_RISK_PERCENT
    ):
        raise ValueError(
            "Le risque doit être compris entre "
            f"{MIN_RISK_PERCENT}% et "
            f"{MAX_RISK_PERCENT}%."
        )

    return risk_percent


# ============================================================
# TWELVE DATA
# ============================================================

def fetch_candles(asset: str, interval: str, candles: int):
    if not TWELVE_DATA_API_KEY:
        raise ValueError("TWELVE_DATA_API_KEY n'est pas configurée.")

    symbol = normalize_asset(asset)

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": candles,
        "apikey": TWELVE_DATA_API_KEY,
        "format": "JSON",
    }

    response = requests.get(
        TWELVE_DATA_URL,
        params=params,
        timeout=20,
    )

    response.raise_for_status()

    data = response.json()

    if "values" not in data:
        raise ValueError(
            f"Réponse Twelve Data invalide: {data}"
        )

    return data["values"]


# ============================================================
# INDICATORS
# ============================================================

def calculate_ema(
    series: pd.Series,
    period: int,
) -> pd.Series:

    return series.ewm(
        span=period,
        adjust=False,
    ).mean()


def calculate_rsi(
    series: pd.Series,
    period: int = 14,
) -> pd.Series:

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan,
    )

    rsi = 100 - (
        100 / (1 + rs)
    )

    return (
        rsi
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(50.0)
    )


def calculate_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:

    previous_close = df["close"].shift(1)

    tr1 = (
        df["high"] - df["low"]
    )

    tr2 = (
        df["high"] - previous_close
    ).abs()

    tr3 = (
        df["low"] - previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()


def calculate_macd(
    series: pd.Series,
) -> Dict[str, pd.Series]:

    ema12 = calculate_ema(
        series,
        12,
    )

    ema26 = calculate_ema(
        series,
        26,
    )

    macd = ema12 - ema26

    signal = calculate_ema(
        macd,
        9,
    )

    histogram = macd - signal

    return {
        "macd": macd,
        "signal": signal,
        "histogram": histogram,
    }


def calculate_indicators(
    df: pd.DataFrame,
) -> Dict[str, float]:

    close = df["close"]

    ema9 = calculate_ema(
        close,
        9,
    )

    ema21 = calculate_ema(
        close,
        21,
    )

    rsi14 = calculate_rsi(
        close,
        14,
    )

    momentum5 = (
        close.pct_change(
            periods=5
        ) * 100
    )

    atr14 = calculate_atr(
        df,
        14,
    )

    macd_data = calculate_macd(
        close
    )

    return {
        "ema9": safe_float(
            ema9.iloc[-1]
        ),
        "ema21": safe_float(
            ema21.iloc[-1]
        ),
        "rsi14": safe_float(
            rsi14.iloc[-1],
            50.0,
        ),
        "momentum_5": safe_float(
            momentum5.iloc[-1]
        ),
        "atr14": safe_float(
            atr14.iloc[-1]
        ),
        "macd": safe_float(
            macd_data["macd"].iloc[-1]
        ),
        "macd_signal": safe_float(
            macd_data["signal"].iloc[-1]
        ),
        "macd_histogram": safe_float(
            macd_data["histogram"].iloc[-1]
        ),
    }


# ============================================================
# SCORING ENGINE v4.1.0
# ============================================================

def calculate_scores(
    indicators: Dict[str, float],
) -> Dict[str, Any]:

    ema9 = indicators["ema9"]
    ema21 = indicators["ema21"]
    rsi = indicators["rsi14"]
    momentum = indicators["momentum_5"]
    macd = indicators["macd"]
    macd_signal = indicators["macd_signal"]

    score = 0

    component_scores = {
        "ema": 0,
        "rsi": 0,
        "momentum": 0,
        "macd": 0,
    }

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    ema_difference = ema9 - ema21

    if ema_difference > 0:

        if (
            abs(ema_difference)
            >= EMA_SEPARATION_THRESHOLD
        ):
            component_scores["ema"] = 2
        else:
            component_scores["ema"] = 1

    elif ema_difference < 0:

        if (
            abs(ema_difference)
            >= EMA_SEPARATION_THRESHOLD
        ):
            component_scores["ema"] = -2
        else:
            component_scores["ema"] = -1

    score += component_scores["ema"]

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------
    #
    # Règle validée :
    # 55 <= RSI < 70 : +1
    # 30 < RSI <= 45 : -1
    #
    # RSI > 70 = surachat/caution
    # RSI < 30 = survente/caution
    # Pas de +1/-1 automatique.
    # --------------------------------------------------------

    if 55 <= rsi < 70:

        component_scores["rsi"] = 1

    elif 30 < rsi <= 45:

        component_scores["rsi"] = -1

    else:

        component_scores["rsi"] = 0

    score += component_scores["rsi"]

    # --------------------------------------------------------
    # MOMENTUM 5
    # --------------------------------------------------------

    if abs(momentum) < 0.03:

        component_scores["momentum"] = 0

    elif (
        0.03 <= momentum < 0.15
    ):

        component_scores["momentum"] = 1

    elif (
        -0.15 < momentum <= -0.03
    ):

        component_scores["momentum"] = -1

    elif momentum >= 0.15:

        component_scores["momentum"] = 2

    elif momentum <= -0.15:

        component_scores["momentum"] = -2

    score += component_scores["momentum"]

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    if macd > macd_signal:

        component_scores["macd"] = 1

    elif macd < macd_signal:

        component_scores["macd"] = -1

    score += component_scores["macd"]

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    if score >= 3:

        trend = "BULLISH"

    elif score <= -3:

        trend = "BEARISH"

    else:

        trend = "NEUTRAL"

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    if score >= 4:

        signal = "BUY"

    elif score <= -4:

        signal = "SELL"

    else:

        signal = "WAIT"

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence = min(
        95,
        max(
            50,
            50 + abs(score) * 8,
        ),
    )

    # --------------------------------------------------------
    # SETUP QUALITY
    # --------------------------------------------------------

    if abs(score) >= 5:

        setup_quality = "STRONG"

    elif abs(score) >= 4:

        setup_quality = "GOOD"

    elif abs(score) >= 2:

        setup_quality = "MODERATE"

    else:

        setup_quality = "WEAK"

    return {
        "score": score,
        "trend": trend,
        "signal": signal,
        "confidence": confidence,
        "setup_quality": setup_quality,
        "component_scores": component_scores,
    }


# ============================================================
# TRADE PLAN
# ============================================================

def build_trade_plan(
    signal: str,
    price: float,
    atr: float,
) -> Dict[str, Any]:

    signal = str(
        signal
    ).strip().upper()

    price = safe_float(price)
    atr = safe_float(atr)

    if signal not in {"BUY", "SELL"}:
        raise ValueError(
            "Le signal doit être BUY ou SELL."
        )

    if price <= 0:
        raise ValueError(
            "Le prix doit être supérieur à zéro."
        )

    if atr <= 0:
        raise ValueError(
            "L'ATR doit être supérieur à zéro."
        )

    risk_distance = (
        atr * ATR_STOP_MULTIPLIER
    )

    if signal == "BUY":

        stop_loss = (
            price - risk_distance
        )

        take_profit_1 = (
            price
            + risk_distance * MIN_RR_TP1
        )

        take_profit_2 = (
            price
            + risk_distance * MIN_RR_TP2
        )

    else:

        stop_loss = (
            price + risk_distance
        )

        take_profit_1 = (
            price
            - risk_distance * MIN_RR_TP1
        )

        take_profit_2 = (
            price
            - risk_distance * MIN_RR_TP2
        )

    rr_tp1 = (
        abs(take_profit_1 - price)
        / risk_distance
    )

    rr_tp2 = (
        abs(take_profit_2 - price)
        / risk_distance
    )

    trade_plan = {
        "direction": signal,
        "entry": round(price, 8),
        "stop_loss": round(
            stop_loss,
            8,
        ),
        "take_profit_1": round(
            take_profit_1,
            8,
        ),
        "take_profit_2": round(
            take_profit_2,
            8,
        ),
        "risk_distance": round(
            risk_distance,
            8,
        ),
        "rr_tp1": round(
            rr_tp1,
            2,
        ),
        "rr_tp2": round(
            rr_tp2,
            2,
        ),
    }

    return trade_plan


# ============================================================
# TRADE PLAN VALIDATION
# ============================================================

def validate_trade_plan(
    trade_plan: Dict[str, Any],
) -> bool:

    direction = trade_plan.get(
        "direction"
    )

    entry = safe_float(
        trade_plan.get("entry")
    )

    stop_loss = safe_float(
        trade_plan.get("stop_loss")
    )

    tp1 = safe_float(
        trade_plan.get("take_profit_1")
    )

    tp2 = safe_float(
        trade_plan.get("take_profit_2")
    )

    rr_tp1 = safe_float(
        trade_plan.get("rr_tp1")
    )

    rr_tp2 = safe_float(
        trade_plan.get("rr_tp2")
    )

    if direction not in {"BUY", "SELL"}:
        return False

    if (
        entry <= 0
        or stop_loss <= 0
        or tp1 <= 0
        or tp2 <= 0
    ):
        return False

    if direction == "BUY":

        if not (
            stop_loss < entry
            < tp1
            < tp2
        ):
            return False

    elif direction == "SELL":

        if not (
            stop_loss > entry
            > tp1
            > tp2
        ):
            return False

    if rr_tp1 < MIN_RR_TP1:
        return False

    if rr_tp2 < MIN_RR_TP2:
        return False

    return True


# ============================================================
# POSITION SIZE / RISK MANAGEMENT
# ============================================================

def calculate_position_size(
    account_balance: float,
    risk_percent: float,
    entry: float,
    stop_loss: float,
) -> Dict[str, float]:

    account_balance = safe_float(
        account_balance
    )

    risk_percent = validate_risk_percent(
        risk_percent
    )

    entry = safe_float(entry)
    stop_loss = safe_float(stop_loss)

    if account_balance <= 0:
        raise ValueError(
            "Le capital doit être supérieur à zéro."
        )

    if entry <= 0:
        raise ValueError(
            "Le prix d'entrée doit être supérieur à zéro."
        )

    if stop_loss <= 0:
        raise ValueError(
            "Le Stop Loss doit être supérieur à zéro."
        )

    stop_distance = abs(
        entry - stop_loss
    )

    if stop_distance <= 0:
        raise ValueError(
            "La distance du Stop Loss doit être supérieure à zéro."
        )

    risk_amount = (
        account_balance
        * risk_percent
        / 100
    )

    stop_loss_pips = (
        stop_distance
        / PIP_SIZE_EURUSD
    )

    position_size_lots = (
        risk_amount
        / (
            stop_loss_pips
            * PIP_VALUE_PER_STANDARD_LOT
        )
    )

    max_loss = (
        position_size_lots
        * stop_loss_pips
        * PIP_VALUE_PER_STANDARD_LOT
    )

    return {
        "account_balance": round(
            account_balance,
            2,
        ),
        "risk_percent": round(
            risk_percent,
            2,
        ),
        "risk_amount": round(
            risk_amount,
            2,
        ),
        "stop_loss_pips": round(
            stop_loss_pips,
            2,
        ),
        "pip_value_per_lot": round(
            PIP_VALUE_PER_STANDARD_LOT,
            2,
        ),
        "position_size_lots": round(
            position_size_lots,
            4,
        ),
        "max_loss": round(
            max_loss,
            2,
        ),
    }


# ============================================================
# RISK VALIDATION
# ============================================================

def validate_risk_management(
    risk_management: Dict[str, Any],
    account_balance: float,
    risk_percent: float,
) -> bool:

    try:

        expected_risk_amount = (
            account_balance
            *risk_percent
            / 100
        )

        actual_risk_amount = safe_float(
            risk_management.get(
                "risk_amount"
            )
        )

        max_loss = safe_float(
            risk_management.get(
                "max_loss"
            )
        )

        if expected_risk_amount <= 0:
            return False

        tolerance = max(
            0.01,
            expected_risk_amount * 0.01,
        )

        if abs(
            actual_risk_amount
            - expected_risk_amount
        ) > tolerance:
            return False

        if abs(
            max_loss
            - expected_risk_amount
        ) > tolerance:
            return False

        return True

    except Exception:
        return False


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "success",
        "app": "Pocket AI Trader",
        "engine_version": APP_VERSION,
        "asset": ASSET,
        "timeframe": TIMEFRAME,
        "message": "API opérationnelle",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "engine_version": APP_VERSION,
        "twelve_data_configured": bool(
            TWELVE_DATA_API_KEY
        ),
    }


# ============================================================
# VERSION
# ============================================================

@app.get("/version")
def version():

    return {
        "engine_version": APP_VERSION,
        "app": "Pocket AI Trader",
        "asset": ASSET,
        "timeframe": TIMEFRAME,
    }


# ============================================================
# CANDLES
# ============================================================

@app.get("/candles")
def get_candles(
    asset: str = ASSET,
    timeframe: str = TIMEFRAME,
    candles: int = DEFAULT_CANDLES,
):

    df = fetch_candles(
        asset=asset,
        timeframe=timeframe,
        candles=candles,
    )

    result = []

    for _, row in df.iterrows():

        result.append(
            {
                "datetime": row[
                    "datetime"
                ].isoformat(),
                "open": safe_float(
                    row["open"]
                ),
                "high": safe_float(
                    row["high"]
                ),
                "low": safe_float(
                    row["low"]
                ),
                "close": safe_float(
                    row["close"]
                ),
            }
        )

    return {
        "status": "success",
        "engine_version": APP_VERSION,
        "asset": normalize_asset(asset),
        "timeframe": timeframe,
        "count": len(result),
        "candles": result,
    }


# ============================================================
# ANALYZE
# ============================================================

@app.post("/analyze")
def analyze(
    request: AnalyzeRequest,
):

    # Validation du risque avant analyse
    try:

        risk_percent = validate_risk_percent(
            request.risk_percent
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    asset = normalize_asset(
        request.asset
    )

    df = fetch_candles(
        asset=asset,
        timeframe=request.timeframe,
        candles=request.candles,
    )

    indicators = calculate_indicators(
        df
    )

    scoring = calculate_scores(
        indicators
    )

    price = safe_float(
        df["close"].iloc[-1]
    )

    signal = scoring["signal"]

    trade_plan = None
    risk_management = None

    trade_plan_valid = False
    risk_valid = False

    # ========================================================
    # TRADE PLAN UNIQUEMENT POUR BUY / SELL
    # ========================================================

    if signal in {"BUY", "SELL"}:

        try:

            trade_plan = build_trade_plan(
                signal=signal,
                price=price,
                atr=indicators["atr14"],
            )

            trade_plan_valid = (
                validate_trade_plan(
                    trade_plan
                )
            )

            if not trade_plan_valid:

                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Trade Plan invalide."
                    ),
                )

            # =================================================
            # RISK MANAGEMENT
            # =================================================

            risk_management = (calculate_position_size(
                    account_balance=(
                        request.account_balance
                    ),
                    risk_percent=risk_percent,
                    entry=trade_plan["entry"],
                    stop_loss=trade_plan[
                        "stop_loss"
                    ],
                )
            )

            risk_valid = (
                validate_risk_management(
                    risk_management,
                    request.account_balance,
                    risk_percent,
                )
            )

            if not risk_valid:

                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Gestion du risque invalide."
                    ),
                )

        except ValueError as exc:

            raise HTTPException(
                status_code=400,
                detail=str(exc),
            )

    # ========================================================
    # RESPONSE
    # ========================================================

    return {
        "status": "success",
        "engine_version": APP_VERSION,

        "asset": asset,
        "timeframe": request.timeframe,

        "price": round(
            price,
            8,
        ),

        "signal": signal,
        "confidence": scoring[
            "confidence"
        ],

        "score": scoring[
            "score"
        ],

        "trend": scoring[
            "trend"
        ],

        "setup_quality": scoring[
            "setup_quality"
        ],

        "component_scores": scoring[
            "component_scores"
        ],

        "indicators": indicators,

        "trade_plan": trade_plan,

        "trade_plan_valid": (
            trade_plan_valid
        ),

        "risk_management": risk_management,

        "risk_valid": risk_valid,

        "candles_used": len(df),
    }


# ============================================================
# QUICK ANALYSIS
# ============================================================

@app.get("/quick-analysis")
def quick_analysis(
    asset: str = ASSET,
    timeframe: str = TIMEFRAME,
):

    df = fetch_candles(
        asset=asset,
        timeframe=timeframe,
        candles=DEFAULT_CANDLES,
    )

    indicators = calculate_indicators(
        df
    )

    scoring = calculate_scores(
        indicators
    )

    price = safe_float(
        df["close"].iloc[-1]
    )

    return {
        "status": "success",
        "engine_version": APP_VERSION,
        "asset": normalize_asset(asset),
        "timeframe": timeframe,
        "price": round(
            price,
            8,
        ),
        "signal": scoring["signal"],
        "confidence": scoring[
            "confidence"
        ],
        "score": scoring[
            "score"
        ],
        "trend": scoring[
            "trend"
        ],
        "setup_quality": scoring[
            "setup_quality"
        ],
        "component_scores": scoring[
            "component_scores"
        ],  "indicators": indicators,
    }


# ============================================================
# TEST TRADE PLAN
# ============================================================

@app.post("/test-trade-plan")
def test_trade_plan(
    request: TradePlanRequest,
):

    signal = str(
        request.signal
    ).strip().upper()

    if signal not in {"BUY", "SELL"}:

        raise HTTPException(
            status_code=400,
            detail=(
                "Le signal doit être BUY ou SELL."
            ),
        )

    try:

        trade_plan = build_trade_plan(
            signal=signal,
            price=request.price,
            atr=request.atr,
        )

        trade_plan_valid = (
            validate_trade_plan(
                trade_plan
            )
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    return {
        "status": "success",
        "engine_version": APP_VERSION,
        "test": True,
        "signal": signal,
        "price": request.price,
        "atr": request.atr,
        "trade_plan": trade_plan,
        "trade_plan_valid": (
            trade_plan_valid
        ),
    }


# ============================================================
# TEST RISK
# ============================================================

@app.post("/test-risk")
def test_risk(
    request: RiskRequest,
):

    try:

        risk_percent = validate_risk_percent(
            request.risk_percent
        )

        risk_management = (
            calculate_position_size(
                account_balance=(
                    request.account_balance
                ),
                risk_percent=risk_percent,
                entry=request.entry,
                stop_loss=request.stop_loss,
            )
        )

        risk_valid = (
            validate_risk_management(
                risk_management,
                request.account_balance,
                risk_percent,
            )
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    return {
        "status": "success",
        "engine_version": APP_VERSION,
        "test": True,
        "entry": request.entry,
        "stop_loss": request.stop_loss,
        "risk_management": risk_management,
        "risk_valid": risk_valid,
    }


# ============================================================
# RENDER / LOCAL START
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv(
            "PORT",
            "8000",
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
)
