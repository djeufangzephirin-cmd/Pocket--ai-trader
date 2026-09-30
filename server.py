import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

# ============================================================
# APPLICATION
# ============================================================

APP_VERSION = "4.1.0"
APP_NAME = "Pocket AI Trader"

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Moteur d'analyse technique Forex avec gestion du risque.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# CONFIGURATION
# ============================================================

TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "").strip()

DEFAULT_ASSET = "EURUSD"
DEFAULT_TIMEFRAME = "15m"

DEFAULT_CANDLES = 100
MIN_CANDLES = 30
MAX_CANDLES = 500

# Scoring
EMA_SEPARATION_THRESHOLD = 0.0005

# Risk management
DEFAULT_ACCOUNT_BALANCE = 1000.0
DEFAULT_RISK_PERCENT = 1.0

MIN_RISK_PERCENT = 0.1
MAX_RISK_PERCENT = 2.0

PIP_SIZE_EURUSD = 0.0001
PIP_VALUE_PER_STANDARD_LOT = 10.0

MIN_RR_TP1 = 1.0
MIN_RR_TP2 = 1.5

# ATR
ATR_STOP_MULTIPLIER = 1.5
TP1_RR = 1.5
TP2_RR = 2.5

REQUEST_TIMEOUT = 20

# ============================================================
# DATA MODELS
# ============================================================


class AnalyzeRequest(BaseModel):
    asset: str = DEFAULT_ASSET
    timeframe: str = DEFAULT_TIMEFRAME
    candles: int = Field(DEFAULT_CANDLES, ge=MIN_CANDLES, le=MAX_CANDLES)

    account_balance: float = Field(
        DEFAULT_ACCOUNT_BALANCE,
        gt=0,
    )

    risk_percent: float = Field(
        DEFAULT_RISK_PERCENT,
        ge=MIN_RISK_PERCENT,
        le=MAX_RISK_PERCENT,
    )

    @field_validator("asset")
    @classmethod
    def validate_asset(cls, value: str) -> str:
        value = value.strip().upper()

        if not value:
            raise ValueError("L'actif ne peut pas être vide.")

        return value

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, value: str) -> str:
        value = value.strip().lower()

        if not value:
            raise ValueError("Le timeframe ne peut pas être vide.")

        return value


class TradePlanRequest(BaseModel):
    signal: str
    price: float = Field(gt=0)
    atr: float = Field(gt=0)

    account_balance: float = Field(
        DEFAULT_ACCOUNT_BALANCE,
        gt=0,
    )

    risk_percent: float = Field(
        DEFAULT_RISK_PERCENT,
        ge=MIN_RISK_PERCENT,
        le=MAX_RISK_PERCENT,
    )


class RiskManagementRequest(BaseModel):
    account_balance: float = Field(
        DEFAULT_ACCOUNT_BALANCE,
        gt=0,
    )

    risk_percent: float = Field(
        DEFAULT_RISK_PERCENT,
        ge=MIN_RISK_PERCENT,
        le=MAX_RISK_PERCENT,
    )

    stop_loss_pips: float = Field(gt=0)

    pip_value_per_lot: float = Field(
        PIP_VALUE_PER_STANDARD_LOT,
        gt=0,
    )


# ============================================================
# UTILITIES
# ============================================================


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def normalize_asset(asset: str) -> str:
    """
    Normalise les symboles Forex pour Twelve Data.

    EURUSD -> EUR/USD
    EUR/USD -> EUR/USD
    """

    asset = asset.strip().upper()

    forex_pairs = {
        "EURUSD": "EUR/USD",
        "GBPUSD": "GBP/USD",
        "USDJPY": "USD/JPY",
        "USDCHF": "USD/CHF",
        "AUDUSD": "AUD/USD",
        "USDCAD": "USD/CAD",
        "NZDUSD": "NZD/USD",
        "EURGBP": "EUR/GBP",
        "EURJPY": "EUR/JPY",
        "GBPJPY": "GBP/JPY",
    }

    return forex_pairs.get(asset, asset)


def normalize_interval(timeframe: str) -> str:
    """
    Conversion interne -> format Twelve Data.

    1m   -> 1min
    5m   -> 5min
    15m  -> 15min
    30m  -> 30min
    1h   -> 1h
    4h   -> 4h
    1d   -> 1day
    """

    timeframe = timeframe.strip().lower()

    mapping = {
        "1m": "1min",
        "1min": "1min",
        "5m": "5min",
        "5min": "5min",
        "15m": "15min",
        "15min": "15min",
        "30m": "30min",
        "30min": "30min",
        "45m": "45min",
        "45min": "45min",
        "1h": "1h",
        "60m": "1h",
        "2h": "2h",
        "4h": "4h",
        "8h": "8h",
        "1d": "1day",
        "1day": "1day",
        "1w": "1week",
        "1week": "1week",
    }

    if timeframe not in mapping:
        raise ValueError(
            f"Timeframe non supporté: {timeframe}. "
            f"Utilisez par exemple 15m."
        )

    return mapping[timeframe]


def validate_risk_percent(risk_percent: float) -> None:
    if not (
        MIN_RISK_PERCENT
        <= risk_percent
        <= MAX_RISK_PERCENT
    ):
        raise ValueError(
            f"Le risque doit être compris entre "
            f"{MIN_RISK_PERCENT}% et {MAX_RISK_PERCENT}%."
        )


# ============================================================
# TWELVE DATA
# ============================================================


def fetch_candles(
    asset: str,
    timeframe: str,
    candles: int,
) -> pd.DataFrame:

    if not TWELVE_DATA_API_KEY:
        raise RuntimeError(
            "TWELVE_DATA_API_KEY n'est pas configurée."
        )

    if candles < MIN_CANDLES:
        raise ValueError(
            f"Minimum requis: {MIN_CANDLES} bougies."
        )

    if candles > MAX_CANDLES:
        candles = MAX_CANDLES

    symbol = normalize_asset(asset)
    interval = normalize_interval(timeframe)

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": candles,
        "apikey": TWELVE_DATA_API_KEY,
        "format": "JSON",
    }

    try:
        response = requests.get(
            TWELVE_DATA_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Impossible de contacter Twelve Data: {exc}"
        ) from exc

    # Ne pas utiliser response.raise_for_status() seul :
    # Twelve Data peut renvoyer une erreur JSON avec HTTP 200.
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Réponse Twelve Data invalide: "
            f"{response.text[:300]}"
        ) from exc

    if response.status_code != 200:
        message = payload.get(
            "message",
            payload.get("code", response.text),
        )

        raise RuntimeError(
            f"Erreur Twelve Data HTTP {response.status_code}: "
            f"{message}"
        )

    if isinstance(payload, dict) and payload.get("status") == "error":
        message = payload.get(
            "message",
            "Erreur inconnue Twelve Data",
        )

        raise RuntimeError(
            f"Erreur Twelve Data: {message}"
        )

    values = payload.get("values")

    if not values:
        raise RuntimeError(
            "Twelve Data n'a retourné aucune bougie."
        )

    df = pd.DataFrame(values)

    required_columns = [
        "datetime",
        "open",
        "high",
        "low",
        "close",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Données Twelve Data incomplètes. "
            f"Colonnes manquantes: {missing}"
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce",
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "datetime",
            "open",
            "high",
            "low",
            "close",
        ]
    )

    df = df.sort_values("datetime").reset_index(drop=True)

    if len(df) < MIN_CANDLES:
        raise RuntimeError(
            f"Nombre insuffisant de bougies: {len(df)}. "
            f"Minimum requis: {MIN_CANDLES}."
        )

    return df


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

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    # Cas particuliers :
    # hausse sans baisse -> RSI 100
    # baisse sans hausse -> RSI 0
    rsi = rsi.where(
        ~((avg_loss == 0) & (avg_gain > 0)),
        100,
    )

    rsi = rsi.where(
        ~((avg_gain == 0) & (avg_loss > 0)),
        0,
    )

    return rsi


def calculate_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:

    high = df["high"]
    low = df["low"]
    close = df["close"]

    previous_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    return atr


def calculate_macd(
    series: pd.Series,
) -> Dict[str, pd.Series]:

    ema12 = calculate_ema(series, 12)
    ema26 = calculate_ema(series, 26)

    macd = ema12 - ema26

    signal = calculate_ema(macd, 9)

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

    ema9 = calculate_ema(close, 9)
    ema21 = calculate_ema(close, 21)

    rsi14 = calculate_rsi(close, 14)

    atr14 = calculate_atr(df, 14)

    macd_data = calculate_macd(close)

    momentum_5 = close.pct_change(5).iloc[-1]

    values = {
        "ema9": safe_float(ema9.iloc[-1]),
        "ema21": safe_float(ema21.iloc[-1]),
        "rsi14": safe_float(rsi14.iloc[-1]),
        "momentum_5": safe_float(momentum_5),
        "atr14": safe_float(atr14.iloc[-1]),
        "macd": safe_float(macd_data["macd"].iloc[-1]),
        "macd_signal": safe_float(
            macd_data["signal"].iloc[-1]
        ),
        "macd_histogram": safe_float(
            macd_data["histogram"].iloc[-1]
        ),
    }

    return values


# ============================================================
# SCORING ENGINE
# ============================================================


def calculate_score(
    indicators: Dict[str, float],
) -> Dict[str, Any]:

    ema9 = indicators["ema9"]
    ema21 = indicators["ema21"]
    rsi = indicators["rsi14"]
    momentum = indicators["momentum_5"]
    macd = indicators["macd"]
    macd_signal = indicators["macd_signal"]

    score = 0
    reasons: List[str] = []

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    ema_difference = ema9 - ema21

    if ema_difference > 0:
        if abs(ema_difference) >= EMA_SEPARATION_THRESHOLD:
            score += 2
            reasons.append("EMA9 nettement au-dessus de EMA21")
        else:
            score += 1
            reasons.append("EMA9 légèrement au-dessus de EMA21")

    elif ema_difference < 0:
        if abs(ema_difference) >= EMA_SEPARATION_THRESHOLD:
            score -= 2
            reasons.append("EMA9 nettement sous EMA21")
        else:
            score -= 1
            reasons.append("EMA9 légèrement sous EMA21")

    else:
        reasons.append("EMA9 et EMA21 pratiquement égales")

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    # Règle validée :
    # 55 <= RSI < 70 : +1
    # 30 < RSI <= 45 : -1
    # >70 : surachat, pas de bonus automatique
    # <30 : survente, pas de bonus automatique

    if 55 <= rsi < 70:
        score += 1
        reasons.append("RSI favorable aux acheteurs")

    elif 30 < rsi <= 45:
        score -= 1
        reasons.append("RSI favorable aux vendeurs")

    elif rsi >= 70:
        reasons.append("RSI en zone de surachat")

    elif rsi <= 30:
        reasons.append("RSI en zone de survente")

    else:
        reasons.append("RSI neutre")

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    if abs(momentum) < 0.03:
        momentum_score = 0
        reasons.append("Momentum faible")

    elif 0.03 <= momentum < 0.15:
        momentum_score = 1
        reasons.append("Momentum haussier modéré")

    elif -0.15 < momentum <= -0.03:
        momentum_score = -1
        reasons.append("Momentum baissier modéré")

    elif momentum >= 0.15:
        momentum_score = 2
        reasons.append("Momentum haussier fort")

    else:
        momentum_score = -2
        reasons.append("Momentum baissier fort")

    score += momentum_score

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    if macd > macd_signal and macd > 0:
        score += 2
        reasons.append("MACD haussier confirmé")

    elif macd > macd_signal:
        score += 1
        reasons.append("MACD en amélioration")

    elif macd < macd_signal and macd < 0:
        score -= 2
        reasons.append("MACD baissier confirmé")

    elif macd < macd_signal:
        score -= 1
        reasons.append("MACD en dégradation")

    else:
        reasons.append("MACD neutre")

    # --------------------------------------------------------
    # TREND / SIGNAL
    # --------------------------------------------------------

    if score >= 3:
        trend = "BULLISH"

    elif score <= -3:
        trend = "BEARISH"

    else:
        trend = "NEUTRAL"

    if score >= 4:
        signal = "BUY"

    elif score <= -4:
        signal = "SELL"

    else:
        signal = "WAIT"

    confidence = int(
        max(
            50,
            min(
                95,
                50 + abs(score) * 8,
            ),
        )
    )

    absolute_score = abs(score)

    if absolute_score >= 5:
        setup_quality = "STRONG"

    elif absolute_score >= 4:
        setup_quality = "GOOD"

    elif absolute_score >= 2:
        setup_quality = "MODERATE"

    else:
        setup_quality = "WEAK"

    return {
        "score": score,
        "trend": trend,
        "signal": signal,
        "confidence": confidence,
        "setup_quality": setup_quality,
        "reasons": reasons,
    }


# ============================================================
# TRADE PLAN
# ============================================================


def build_trade_plan(
    signal: str,
    price: float,
    atr: float,
) -> Dict[str, float]:

    signal = signal.upper().strip()

    if signal not in {"BUY", "SELL"}:
        return {
            "entry": price,
            "stop_loss": price,
            "take_profit_1": price,
            "take_profit_2": price,
            "risk_distance": 0.0,
            "rr_tp1": 0.0,
            "rr_tp2": 0.0,
        }

    stop_distance = atr * ATR_STOP_MULTIPLIER

    if signal == "BUY":

        entry = price

        stop_loss = price - stop_distance

        tp1_distance = stop_distance * TP1_RR
        tp2_distance = stop_distance * TP2_RR

        take_profit_1 = price + tp1_distance
        take_profit_2 = price + tp2_distance

    else:

        entry = price

        stop_loss = price + stop_distance

        tp1_distance = stop_distance * TP1_RR
        tp2_distance = stop_distance * TP2_RR

        take_profit_1 = price - tp1_distance
        take_profit_2 = price - tp2_distance

    risk_distance = abs(entry - stop_loss)

    rr_tp1 = (
        abs(take_profit_1 - entry)
        / risk_distance
        if risk_distance > 0
        else 0.0
    )

    rr_tp2 = (
        abs(take_profit_2 - entry)
        / risk_distance
        if risk_distance > 0
        else 0.0
    )

    return {
        "entry": round(entry, 8),
        "stop_loss": round(stop_loss, 8),
        "take_profit_1": round(take_profit_1, 8),
        "take_profit_2": round(take_profit_2, 8),
        "risk_distance": round(risk_distance, 8),
        "rr_tp1": round(rr_tp1, 4),
        "rr_tp2": round(rr_tp2, 4),
    }


def validate_trade_plan(
    signal: str,
    trade_plan: Dict[str, float],
) -> Dict[str, Any]:

    signal = signal.upper()

    if signal not in {"BUY", "SELL"}:
        return {
            "valid": False,
            "reason": "Le signal n'est pas tradable.",
        }

    entry = trade_plan["entry"]
    stop_loss = trade_plan["stop_loss"]
    tp1 = trade_plan["take_profit_1"]
    tp2 = trade_plan["take_profit_2"]

    rr_tp1 = trade_plan["rr_tp1"]
    rr_tp2 = trade_plan["rr_tp2"]

    if signal == "BUY":

        direction_valid = (
            stop_loss < entry
            and tp1 > entry
            and tp2 > tp1
        )

    else:

        direction_valid = (
            stop_loss > entry
            and tp1 < entry
            and tp2 < tp1
        )

    rr_valid = (
        rr_tp1 >= MIN_RR_TP1
        and rr_tp2 >= MIN_RR_TP2
    )

    return {
        "valid": bool(direction_valid and rr_valid),
        "direction_valid": bool(direction_valid),
        "rr_valid": bool(rr_valid),
        "min_rr_tp1": MIN_RR_TP1,
        "min_rr_tp2": MIN_RR_TP2,
    }


# ============================================================
# RISK MANAGEMENT
# ============================================================


def calculate_position_size(
    account_balance: float,
    risk_percent: float,
    stop_loss_pips: float,
    pip_value_per_lot: float = PIP_VALUE_PER_STANDARD_LOT,
) -> Dict[str, float]:

    validate_risk_percent(risk_percent)

    if account_balance <= 0:
        raise ValueError(
            "Le solde du compte doit être supérieur à zéro."
        )

    if stop_loss_pips <= 0:
        raise ValueError(
            "La distance du stop-loss doit être supérieure à zéro."
        )

    if pip_value_per_lot <= 0:
        raise ValueError(
            "La valeur du pip doit être supérieure à zéro."
        )

    risk_amount = (
        account_balance * risk_percent / 100
    )

    position_size_lots = (
        risk_amount
        / (stop_loss_pips * pip_value_per_lot)
    )

    max_loss = (
        position_size_lots
        * stop_loss_pips
        * pip_value_per_lot
    )

    return {
        "account_balance": round(account_balance, 2),
        "risk_percent": round(risk_percent, 4),
        "risk_amount": round(risk_amount, 2),
        "stop_loss_pips": round(stop_loss_pips, 2),
        "pip_value_per_lot": round(
            pip_value_per_lot,
            4,
        ),
     ),
        "position_size_lots": round(
            position_size_lots,
            4,
        ),
        "max_loss": round(max_loss, 2),
    }


def calculate_stop_loss_pips(
    entry: float,
    stop_loss: float,
    pip_size: float,
) -> float:

    distance = abs(entry - stop_loss)

    return distance / pip_size


# ============================================================
# COMPLETE ANALYSIS
# ============================================================


def analyze_market(
    asset: str,
    timeframe: str,
    candles: int,
    account_balance: float,
    risk_percent: float,
) -> Dict[str, Any]:

    validate_risk_percent(risk_percent)

    df = fetch_candles(
        asset=asset,
        timeframe=timeframe,
        candles=candles,
    )

    indicators = calculate_indicators(df)

    scoring = calculate_score(indicators)

    price = safe_float(
        df["close"].iloc[-1]
    )

    atr = indicators["atr14"]

    trade_plan = build_trade_plan(
        signal=scoring["signal"],
        price=price,
        atr=atr,
    )

    trade_plan_validation = validate_trade_plan(
        scoring["signal"],
        trade_plan,
    )

    risk_management: Optional[
        Dict[str, float]
    ] = None

    if scoring["signal"] in {"BUY", "SELL"}:

        stop_loss_pips = calculate_stop_loss_pips(
            entry=trade_plan["entry"],
            stop_loss=trade_plan["stop_loss"],
            pip_size=PIP_SIZE_EURUSD,
        )

        risk_management = calculate_position_size(
            account_balance=account_balance,
            risk_percent=risk_percent,
            stop_loss_pips=stop_loss_pips,
            pip_value_per_lot=PIP_VALUE_PER_STANDARD_LOT,
        )

    return {
        "status": "success",
        "engine_version": APP_VERSION,

        "market": {
            "asset": asset.upper(),
            "twelve_data_symbol": normalize_asset(asset),
            "timeframe": timeframe,
            "twelve_data_interval": normalize_interval(
                timeframe
            ),
            "candles_used": len(df),
            "last_candle": str(
                df["datetime"].iloc[-1]
            ),
            "price": round(price, 8),
        },

        "signal": scoring["signal"],
        "confidence": scoring["confidence"],
        "score": scoring["score"],
        "trend": scoring["trend"],
        "setup_quality": scoring["setup_quality"],

        "indicators": {
            key: round(value, 10)
            for key, value in indicators.items()
        },

        "analysis": {
            "reasons": scoring["reasons"],
        },

        "trade_plan": trade_plan,

        "trade_plan_validation":
            trade_plan_validation,

        "risk_management":
            risk_management,
    }


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")def root() -> Dict[str, Any]:

    return {
        "status": "online",
        "app": APP_NAME,
        "engine_version": APP_VERSION,
        "message": "Pocket AI Trader API opérationnelle.",
    }


@app.get("/health")
def health() -> Dict[str, Any]:

    return {
        "status": "healthy",
        "engine_version": APP_VERSION,
        "twelve_data_configured": bool(
            TWELVE_DATA_API_KEY
        ),
    }


@app.get("/version")
def version() -> Dict[str, str]:

    return {
        "app": APP_NAME,
        "engine_version": APP_VERSION,
    }


@app.post("/analyze")
def analyze(
    request: AnalyzeRequest,
) -> Dict[str, Any]:

    try:

        return analyze_market(
            asset=request.asset,
            timeframe=request.timeframe,
            candles=request.candles,
            account_balance=request.account_balance,
            risk_percent=request.risk_percent,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne du moteur: {exc}",
        ) from exc


@app.post("/trade-plan")
def trade_plan(
    request: TradePlanRequest,
) -> Dict[str, Any]:

    try:

        signal = request.signal.upper().strip()

        if signal not in {"BUY", "SELL"}:
            raise ValueError(
                "Le signal doit être BUY ou SELL."
            )

        plan = build_trade_plan(
            signal=signal,
            price=request.price,
            atr=request.atr,
        )

        validation = validate_trade_plan(
            signal,
            plan,
        )

        risk_management = None

        if validation["valid"]:

            stop_loss_pips = calculate_stop_loss_pips(
                entry=plan["entry"],
                stop_loss=plan["stop_loss"],
                pip_size=PIP_SIZE_EURUSD,
            )

            risk_management = calculate_position_size(
                account_balance=request.account_balance,
                risk_percent=request.risk_percent,
                stop_loss_pips=stop_loss_pips,
                pip_value_per_lot=PIP_VALUE_PER_STANDARD_LOT,
            )

        return {
            "status": "success",
            "engine_version": APP_VERSION,
            "signal": signal,
            "trade_plan": plan,
            "validation": validation,
            "risk_management": risk_management,
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Erreur Trade Plan: {exc}",
        ) from exc


@app.post("/risk-management")
def risk_management(
    request: RiskManagementRequest,
) -> Dict[str, Any]:

    try:

        result = calculate_position_size(
            account_balance=request.account_balance,
            risk_percent=request.risk_percent,
            stop_loss_pips=request.stop_loss_pips,
            pip_value_per_lot=request.pip_value_per_lot,
        )

        return {
            "status": "success",
            "engine_version": APP_VERSION,
            "risk_management": result,
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Erreur Risk Management: {exc}",
        ) from exc


# ============================================================
# VALIDATION / TEST ENDPOINTS
# ============================================================

@app.get("/test")def root() -> Dict[str, Any]:

    return {
        "status": "online",
        "app": APP_NAME,
        "engine_version": APP_VERSION,
        "message": "Pocket AI Trader API opérationnelle.",
    }


@app.get("/health")
def health() -> Dict[str, Any]:

    return {
        "status": "healthy",
        "engine_version": APP_VERSION,
        "twelve_data_configured": bool(
            TWELVE_DATA_API_KEY
        ),
    }


@app.get("/version")
def version() -> Dict[str, str]:

    return {
        "app": APP_NAME,
        "engine_version": APP_VERSION,
    }


@app.post("/analyze")
def analyze(
    request: AnalyzeRequest,
) -> Dict[str, Any]:

    try:

        return analyze_market(
            asset=request.asset,
            timeframe=request.timeframe,
            candles=request.candles,
            account_balance=request.account_balance,
            risk_percent=request.risk_percent,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne du moteur: {exc}",
        ) from exc


@app.post("/trade-plan")
def trade_plan(
    request: TradePlanRequest,
) -> Dict[str, Any]:

    try:

        signal = request.signal.upper().strip()

        if signal not in {"BUY", "SELL"}:
            raise ValueError(
                "Le signal doit être BUY ou SELL."
            )

        plan = build_trade_plan(
            signal=signal,
            price=request.price,
            atr=request.atr,
        )

        validation = validate_trade_plan(
            signal,
            plan,
        )

        risk_management = None

        if validation["valid"]:

            stop_loss_pips = calculate_stop_loss_pips(
                entry=plan["entry"],
                stop_loss=plan["stop_loss"],
                pip_size=PIP_SIZE_EURUSD,
            )

            risk_management = calculate_position_size(
                account_balance=request.account_balance,
                risk_percent=request.risk_percent,
                stop_loss_pips=stop_loss_pips,
                pip_value_per_lot=PIP_VALUE_PER_STANDARD_LOT,
            )

        return {
            "status": "success",
            "engine_version": APP_VERSION,
            "signal": signal,
            "trade_plan": plan,
            "validation": validation,
            "risk_management": risk_management,
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Erreur Trade Plan: {exc}",
        ) from exc


@app.post("/risk-management")
def risk_management(
    request: RiskManagementRequest,
) -> Dict[str, Any]:

    try:

        result = calculate_position_size(
            account_balance=request.account_balance,
            risk_percent=request.risk_percent,
            stop_loss_pips=request.stop_loss_pips,
            pip_value_per_lot=request.pip_value_per_lot,
        )

        return {
            "status": "success",
            "engine_version": APP_VERSION,
            "risk_management": result,
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Erreur Risk Management: {exc}",
        ) from exc


# ============================================================
# VALIDATION / TEST ENDPOINTS
# ============================================================

@app.get("/test")
def test() -> Dict[str, Any]:

    return {
        "status": "success",
        "engine_version": APP_VERSION,
        "test": True,
        "message": "API fonctionnelle.",
    }


@app.get("/test/risk-management")
def test_risk_management() -> Dict[str, Any]:

    result = calculate_position_size(
        account_balance=1000.0,
        risk_percent=1.0,
        stop_loss_pips=15.0,
        pip_value_per_lot=10.0,
    )

    return {
        "status": "success",
        "engine_version": APP_VERSION,
        "test": True,
        "risk_management": result,
    }


@app.get("/test/trade-plan")
def test_trade_plan() -> Dict[str, Any]:

    price = 1.137
    atr = 0.001
    signal = "BUY"

    plan = build_trade_plan(
        signal=signal,
        price=price,
        atr=atr,
    )

    validation = validate_trade_plan(
        signal,
        plan,
    )

    return {
        "status": "success",
        "engine_version": APP_VERSION,
        "test": True,
        "signal": signal,
        "entry": plan["entry"],
        "stop_loss": plan["stop_loss"],
        "take_profit_1": plan["take_profit_1"],
        "take_profit_2": plan["take_profit_2"],
        "risk_distance": plan["risk_distance"],
        "rr_tp1": plan["rr_tp1"],
        "rr_tp2": plan["rr_tp2"],
        "validation": validation,
    }


# ============================================================
# LOCAL EXECUTION
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
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
