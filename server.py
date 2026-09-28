# ============================================================
# POCKET AI TRADER - SERVER V3
# Moteur d'analyse technique Forex
# FastAPI + Twelve Data
# ============================================================

import os
import math
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# CONFIGURATION
# ============================================================

APP_VERSION = "3.0.0"

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")

TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"

DEFAULT_SYMBOL = "EUR/USD"
DEFAULT_TIMEFRAME = "15min"
DEFAULT_OUTPUTSIZE = 100


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Pocket AI Trader",
    description="Moteur d'analyse technique Forex avec données réelles",
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
# MODELES
# ============================================================

class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = 0


class AnalyzeRequest(BaseModel):
    asset: str = "EURUSD"
    timeframe: str = "15m"
    candles: List[Candle] = Field(..., min_length=30)


# ============================================================
# OUTILS
# ============================================================

def normalize_symbol(asset: str) -> str:
    """
    EURUSD -> EUR/USD
    EUR/USD -> EUR/USD
    """
    asset = asset.upper().strip()

    if "/" in asset:
        return asset

    if len(asset) == 6:
        return f"{asset[:3]}/{asset[3:]}"

    return asset


def normalize_interval(timeframe: str) -> str:
    """
    15m -> 15min
    5m  -> 5min
    1h  -> 1h
    """
    tf = timeframe.lower().strip()

    mapping = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1day",
    }

    return mapping.get(tf, tf)


def clean_number(value):
    if value is None:
        return None

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    return value


def round_price(value: Optional[float], digits: int = 5):
    if value is None:
        return None

    return round(float(value), digits)


# ============================================================
# INDICATEURS
# ============================================================

def ema(values: List[float], period: int) -> float:
    if len(values) < period:
        raise ValueError(f"Pas assez de données pour EMA{period}")

    multiplier = 2 / (period + 1)

    result = sum(values[:period]) / period

    for price in values[period:]:
        result = (price - result) * multiplier + result

    return result


def calculate_rsi(values: List[float], period: int = 14) -> float:
    if len(values) < period + 1:
        raise ValueError("Pas assez de données pour RSI")

    gains = []
    losses = []

    for i in range(1, len(values)):
        change = values[i] - values[i - 1]

        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def calculate_atr(candles: List[Candle], period: int = 14) -> float:
    if len(candles) < period + 1:
        raise ValueError("Pas assez de données pour ATR")

    true_ranges = []

    for i in range(1, len(candles)):
        current = candles[i]
        previous = candles[i - 1]

        tr = max(
            current.high - current.low,
            abs(current.high - previous.close),
            abs(current.low - previous.close),
        )

        true_ranges.append(tr)

    return sum(true_ranges[-period:]) / period


def calculate_momentum(closes: List[float], period: int = 5) -> float:
    if len(closes) <= period:
        return 0.0

    return closes[-1] - closes[-1 - period]


def calculate_average_range(
    candles: List[Candle],
    period: int = 20
) -> float:

    if len(candles) < period:
        period = len(candles)

    ranges = [
        candle.high - candle.low
        for candle in candles[-period:]
    ]

    return sum(ranges) / len(ranges)


def calculate_support_resistance(
    candles: List[Candle],
    lookback: int = 20
):
    recent = candles[-lookback:]

    support = min(c.low for c in recent)
    resistance = max(c.high for c in recent)

    return support, resistance


# ============================================================
# ANALYSE PRINCIPALE
# ============================================================

def analyze_candles(
    candles: List[Candle],
    asset: str = "EURUSD",
    timeframe: str = "15m"
):

    if len(candles) < 30:
        raise ValueError(
            "Minimum 30 bougies nécessaires pour l'analyse."
        )

    # --------------------------------------------------------
    # DONNEES
    # --------------------------------------------------------

    closes = [c.close for c in candles]

    current = candles[-1]
    previous = candles[-2]

    price = current.close

    # --------------------------------------------------------
    # INDICATEURS
    # --------------------------------------------------------

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)

    rsi14 = calculate_rsi(closes, 14)

    momentum_5 = calculate_momentum(closes, 5)

    atr14 = calculate_atr(candles, 14)

    average_range = calculate_average_range(
        candles,
        20
    )

    recent_average_range = calculate_average_range(
        candles,
        5
    )

    support, resistance = calculate_support_resistance(
        candles,
        20
    )

    # --------------------------------------------------------
    # VOLATILITE
    # --------------------------------------------------------

    if average_range <= 0:
        volatility_ratio = 0
    else:
        volatility_ratio = (
            recent_average_range / average_range
        )

    if volatility_ratio < 0.70:
        volatility_state = "LOW"

    elif volatility_ratio > 1.30:
        volatility_state = "HIGH"

    else:
        volatility_state = "NORMAL"

    # --------------------------------------------------------
    # TENDANCE EMA
    # --------------------------------------------------------

    ema_difference = ema9 - ema21

    if ema_difference > atr14 * 0.10:
        ema_trend = "BULLISH"

    elif ema_difference < -atr14 * 0.10:
        ema_trend = "BEARISH"

    else:
        ema_trend = "NEUTRAL"

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if rsi14 >= 55:
        rsi_direction = "BULLISH"

    elif rsi14 <= 45:
        rsi_direction = "BEARISH"

    else:
        rsi_direction = "NEUTRAL"

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    momentum_threshold = max(
        atr14 * 0.10,
        0.00001
    )

    if momentum_5 > momentum_threshold:
        momentum_direction = "BULLISH"

    elif momentum_5 < -momentum_threshold:
        momentum_direction = "BEARISH"

    else:
        momentum_direction = "NEUTRAL"

    # --------------------------------------------------------
    # BOUGIE
    # --------------------------------------------------------

    if current.close > current.open:
        candle_direction = "BULLISH"

    elif current.close < current.open:
        candle_direction = "BEARISH"

    else:
        candle_direction = "DOJI"

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    score = 0
    reasons = []

    # EMA
    if ema_trend == "BULLISH":
        score += 2
        reasons.append("EMA9 au-dessus de EMA21")

    elif ema_trend == "BEARISH":
        score -= 2
        reasons.append("EMA9 sous EMA21")

    else:
        reasons.append(
            "EMA9 et EMA21 proches - zone neutre"
        )

    # RSI
    if rsi14 >= 55:
        score += 1
        reasons.append(
            "RSI confirme une pression haussière"
        )

    elif rsi14 <= 45:
        score -= 1
        reasons.append(
            "RSI confirme une pression baissière"
        )

    else:
        reasons.append(
            "RSI neutre"
        )

    # Momentum
    if momentum_direction == "BULLISH":
        score += 1
        reasons.append("Momentum positif")

    elif momentum_direction == "BEARISH":
        score -= 1
        reasons.append("Momentum négatif")

    else:
        reasons.append("Momentum faible")

    # Bougie
    if candle_direction == "BULLISH":
        score += 1
        reasons.append("Dernière bougie haussière")

    elif candle_direction == "BEARISH":
        score -= 1
        reasons.append("Dernière bougie baissière")

    # --------------------------------------------------------
    # FILTRE VOLATILITE
    # --------------------------------------------------------

    if volatility_state == "LOW":
        reasons.append("Volatilité faible")

    elif volatility_state == "HIGH":
        reasons.append("Volatilité élevée")

    else:
        reasons.append("Volatilité normale")

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

    confidence = 50 + (abs(score) * 8)

    if confidence > 95:
        confidence = 95

    # Faible volatilité = prudence
    if volatility_state == "LOW" and signal != "WAIT":
        confidence -= 10

    confidence = max(
        50,
        min(95, confidence)
    )

    # --------------------------------------------------------
    # TENDANCE FINALE
    # --------------------------------------------------------

    if score >= 2:
        trend = "BULLISH"

    elif score <= -2:
        trend = "BEARISH"

    else:
        trend = "NEUTRAL"

    # --------------------------------------------------------
    # STOP LOSS / TAKE PROFIT
    # --------------------------------------------------------

    entry = price

    stop_loss = None
    take_profit = None
    risk_reward = None

    if signal == "BUY":

        # SL sous le support ou ATR
        atr_stop = entry - (atr14 * 1.5)

        stop_loss = min(
            atr_stop,
            support
        )

        risk = entry - stop_loss

        if risk > 0:
            take_profit = entry + (risk * 2.0)
            risk_reward = 2.0

    elif signal == "SELL":

        # SL au-dessus de la résistance ou ATR
        atr_stop = entry + (atr14 * 1.5)

        stop_loss = max(
            atr_stop,
            resistance
        )

        risk = stop_loss - entry

        if risk > 0:
            take_profit = entry - (risk * 2.0)
            risk_reward = 2.0

    # --------------------------------------------------------
    # DISTANCES
    # --------------------------------------------------------

    if stop_loss is not None:
        stop_distance = abs(entry - stop_loss)
    else:
        stop_distance = None

    if take_profit is not None:
        target_distance = abs(take_profit - entry)
    else:
        target_distance = None

    # --------------------------------------------------------
    # RESULTAT
    # --------------------------------------------------------

    return {
        "status": "success",
        "engine_version": APP_VERSION,

        "asset": asset,
        "timeframe": timeframe,

        "price": round_price(price),

        "signal": signal,
        "confidence": confidence,
        "score": score,
        "trend": trend,

        "indicators": {
            "ema9": round_price(ema9),
            "ema21": round_price(ema21),
            "rsi14": round(rsi14, 2),
            "momentum_5": round(momentum_5, 6),
            "atr14": round(atr14, 6),
        },

        "market": {
            "candle_direction": candle_direction,
            "average_range": round(
                average_range,
                6
            ),
            "recent_average_range": round(
                recent_average_range,
                6
            ),
            "volatility_ratio": round(
                volatility_ratio,
                3
            ),
            "volatility_state": volatility_state,
        },

        "levels": {
            "support": round_price(support),
            "resistance": round_price(resistance),
        },

        "trade_plan": {
            "entry": round_price(entry),
            "stop_loss": round_price(stop_loss),
            "take_profit": round_price(take_profit),
            "risk_reward": risk_reward,
            "stop_distance": round_price(
                stop_distance
            ),
            "target_distance": round_price(
                target_distance
            ),
        },

        "reasons": reasons,

        "candles_count": len(candles),
    }


# ============================================================
# TWELVE DATA
# ============================================================

def get_market_data(
    asset: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    outputsize: int = DEFAULT_OUTPUTSIZE
):

    if not TWELVE_DATA_API_KEY:
        raise HTTPException(
            status_code=500,
            detail=(
                "La variable TWELVE_DATA_API_KEY "
                "n'est pas configurée sur Render."
            )
        )

    symbol = normalize_symbol(asset)
    interval = normalize_interval(timeframe)

    outputsize = max(
        30,
        min(outputsize, 5000)
    )

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
        "format": "JSON",
    }

    try:
        response = requests.get(
            TWELVE_DATA_URL,
            params=params,
            timeout=15
        )

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Erreur connexion Twelve Data: {exc}"
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Twelve Data HTTP "
                f"{response.status_code}"
            )
        )

    try:
        data = response.json()

    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="Réponse Twelve Data invalide."
        )

    if "status" in data and data["status"] == "error":
        raise HTTPException(
            status_code=502,
            detail=data.get(
                "message",
                "Erreur Twelve Data."
            )
        )

    values = data.get("values")

    if not values:
        raise HTTPException(
            status_code=502,
            detail="Aucune bougie reçue de Twelve Data."
        )

    candles = []

    for item in reversed(values):

        try:
            candles.append(
                Candle(
                    open=float(item["open"]),
                    high=float(item["high"]),
                    low=float(item["low"]),
                    close=float(item["close"]),
                    volume=float(
                        item.get("volume", 0) or 0
                    ),
                )
            )

        except (KeyError, TypeError, ValueError):
            continue

    if len(candles) < 30:
        raise HTTPException(
            status_code=502,
            detail=(
                "Pas assez de bougies valides "
                f"reçues: {len(candles)}."
            )
        )

    return candles


# ============================================================
# POST /analyze
# ============================================================

@app.post("/analyze")
def analyze_market(request: AnalyzeRequest):

    try:

        result = analyze_candles(
            request.candles,
            request.asset,
            request.timeframe
        )

        # Retourner les bougies utilisées
        result["candles"] = [
            {
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
            }
            for c in request.candles
        ]

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Erreur analyse: {exc}"
        )


# ============================================================
# GET /market-data
# ============================================================

@app.get("/market-data")
def market_data(
    asset: str = Query(
        "EURUSD",
        description="Symbole Forex, ex: EURUSD"
    ),
    timeframe: str = Query(
        "15m",
        description="Timeframe, ex: 15m"
    ),
    outputsize: int = Query(
        100,
        ge=30,
        le=5000,
        description="Nombre de bougies"
    )
):

    candles = get_market_data(
        asset,
        timeframe,
        outputsize
    )

    return {
        "status": "success",
        "source": "Twelve Data",
        "asset": asset.upper(),
        "timeframe": timeframe,
        "candles_count": len(candles),

        "candles": [
            {
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in candles
        ],
    }


# ============================================================
# GET /analyze-live
# ============================================================

@app.get("/analyze-live")
def analyze_live(
    asset: str = Query(
        "EURUSD",
        description="Symbole Forex"
    ),
    timeframe: str = Query(
        "15m",
        description="Timeframe"
    ),
    outputsize: int = Query(
        100,
        ge=30,
        le=5000,
        description="Nombre de bougies"
    )
):

    candles = get_market_data(
        asset,
        timeframe,
        outputsize
    )

    try:

        result = analyze_candles(
            candles,
            asset,
            timeframe
        )

        result["candles"] = [
            {
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
            }
            for c in candles
        ]

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Erreur analyse live: {exc}"
        )


# ============================================================
# GET /
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "application": "Pocket AI Trader",
        "version": APP_VERSION,
        "engine": "Technical Forex Analysis V3",
        "data_source": "Twelve Data",

        "endpoints": {
            "analyze": "POST /analyze",
            "market_data": "GET /market-data",
            "analyze_live": "GET /analyze-live",
            "documentation": "/docs",
        }
    }


# ============================================================
# LANCEMENT LOCAL
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(
            os.getenv("PORT", "8000")
        )
    )
