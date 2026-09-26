from __future__ import annotations

import math
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from Config import (
    APP_NAME,
    APP_VERSION,
    MIN_CANDLES,
    MIN_CONFIDENCE,
    EMA_FAST_PERIOD,
    EMA_SLOW_PERIOD,
    RSI_PERIOD,
)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title=APP_NAME,
    description="API d'analyse technique pour Pocket AI Trader",
    version=APP_VERSION,
)


# ============================================================
# CORS
# ============================================================

# CORS_ORIGINS peut être défini dans .env.
# Exemple :
# CORS_ORIGINS=https://mon-frontend.com,http://localhost:3000

cors_origins_raw = os.getenv("CORS_ORIGINS", "").strip()

if cors_origins_raw:
    cors_origins = [
        origin.strip()
        for origin in cors_origins_raw.split(",")
        if origin.strip()
    ]
else:
    # Aucun frontend externe autorisé par défaut.
    cors_origins = []


app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ============================================================
# MODELS
# ============================================================

class Candle(BaseModel):
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)

    @field_validator("open", "high", "low", "close")
    @classmethod
    def validate_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Les valeurs OHLC doivent être des nombres finis.")
        return value

    @field_validator("high")
    @classmethod
    def validate_high(cls, value: float, info):
        data = info.data

        if "open" in data and value < data["open"]:
            raise ValueError("high doit être supérieur ou égal à open.")

        if "close" in data and value < data["close"]:
            raise ValueError("high doit être supérieur ou égal à close.")

        return value

    @field_validator("low")
    @classmethod
    def validate_low(cls, value: float, info):
        data = info.data

        if "open" in data and value > data["open"]:
            raise ValueError("low doit être inférieur ou égal à open.")

        if "close" in data and value > data["close"]:
            raise ValueError("low doit être inférieur ou égal à close.")

        return value


class AnalyzeRequest(BaseModel):
    asset: str = Field(..., min_length=1, max_length=30)
    timeframe: str = Field(..., min_length=1, max_length=10)
    candles: List[Candle]

    @field_validator("asset")
    @classmethod
    def validate_asset(cls, value: str) -> str:
        value = value.strip().upper()

        if not value:
            raise ValueError("asset ne peut pas être vide.")

        return value

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, value: str) -> str:
        value = value.strip().lower()

        if not value:
            raise ValueError("timeframe ne peut pas être vide.")

        return value


class MarketData(BaseModel):
    asset: str = Field(..., min_length=1, max_length=30)
    price: float = Field(..., gt=0)
    previous_price: Optional[float] = Field(None, gt=0)
    timeframe: str = Field("1m", min_length=1, max_length=10)

    @field_validator("price", "previous_price")
    @classmethod
    def validate_finite(cls, value):
        if value is not None and not math.isfinite(value):
            raise ValueError("Le prix doit être un nombre fini.")
        return value

    @field_validator("asset")
    @classmethod
    def validate_asset(cls, value: str) -> str:
        return value.strip().upper()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "status": "online",
        "message": "Pocket AI Trader API is running",
        "endpoints": [
            "/",
            "/health",
            "/analyze",
            "/quick-analysis",
            "/docs",
        ],
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "pocket-ai-trader",
        "version": APP_VERSION,
    }


# ============================================================
# EMA
# ============================================================

def calculate_ema(
    values: List[float],
    period: int,
) -> float:

    if len(values) < period:
        raise ValueError(
            f"Il faut au moins {period} valeurs pour calculer l'EMA."
        )

    multiplier = 2 / (period + 1)

    ema = sum(values[:period]) / period

    for price in values[period:]:
        ema = (price - ema) * multiplier + ema

    return ema


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
    values: List[float],
    period: int = RSI_PERIOD,
) -> float:

    if len(values) < period + 1:
        raise ValueError(
            f"Il faut au moins {period + 1} clôtures "
            f"pour calculer le RSI."
        )

    gains = []
    losses = []

    for i in range(1, len(values)):
        difference = values[i] - values[i - 1]

        if difference > 0:
            gains.append(difference)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(difference))

    average_gain = sum(gains[:period]) / period
    average_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        average_gain = (
            (average_gain * (period - 1)) + gains[i]
        ) / period

        average_loss = (
            (average_loss * (period - 1)) + losses[i]
        ) / period

    if average_loss == 0:
        return 100.0

    rs = average_gain / average_loss

    rsi = 100 - (100 / (1 + rs))

    return round(rsi, 2)


# ============================================================
# MOMENTUM
# ============================================================

def calculate_momentum(
    values: List[float],
    period: int = 5,
) -> float:

    if len(values) <= period:
        return 0.0

    previous = values[-period - 1]
    current = values[-1]

    if previous == 0:
        return 0.0

    return ((current - previous) / previous) * 100


# ============================================================
# TREND
# ============================================================

def determine_trend(
    fast_ema: float,
    slow_ema: float,
) -> str:

    if fast_ema > slow_ema:
        return "BULLISH"

    if fast_ema < slow_ema:
        return "BEARISH"

    return "NEUTRAL"


# ============================================================
# SIGNAL ANALYSIS
# ============================================================

def generate_signal(
    rsi: float,
    fast_ema: float,
    slow_ema: float,
    momentum: float,
):
    score = 0
    reasons = []

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    if fast_ema > slow_ema:
        score += 2
        reasons.append(
            f"EMA{EMA_FAST_PERIOD} au-dessus de "
            f"EMA{EMA_SLOW_PERIOD}"
        )

    elif fast_ema < slow_ema:
        score -= 2
        reasons.append(
            f"EMA{EMA_FAST_PERIOD} sous "
            f"EMA{EMA_SLOW_PERIOD}"
        )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if rsi >= 70:
        score -= 1
        reasons.append("RSI en zone de surachat")

    elif rsi <= 30:
        score += 1
        reasons.append("RSI en zone de survente")

    elif rsi > 50:
        score += 1
        reasons.append("RSI supérieur à 50")

    elif rsi < 50:
        score -= 1
        reasons.append("RSI inférieur à 50")

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    if momentum > 0:
        score += 1
        reasons.append("Momentum positif")

    elif momentum < 0:
        score -= 1
        reasons.append("Momentum négatif")

    # --------------------------------------------------------
    # SIGNAL BRUT
    # --------------------------------------------------------

    if score >= 3:
        signal = "CALL"

    elif score <= -3:
        signal = "PUT"

    else:
        signal = "WAIT"

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    # Score maximum théorique = 4
    confidence = min(abs(score) / 4 * 100, 95)

    confidence = round(confidence, 1)

    # --------------------------------------------------------
    # FILTRE DE CONFIDENCE
    # --------------------------------------------------------

    if signal != "WAIT" and confidence < MIN_CONFIDENCE:
        signal = "WAIT"
        reasons.append(
            f"Confiance inférieure au seuil configuré "
            f"({MIN_CONFIDENCE}%)"
        )

    if signal == "WAIT":
        confidence = min(confidence, 55.0)

    return signal, confidence, score, reasons


# ============================================================
# ANALYZE
# ============================================================

@app.post("/analyze")
def analyze_market(request: AnalyzeRequest):

    # --------------------------------------------------------
    # Validation du nombre de bougies
    # --------------------------------------------------------

    if len(request.candles) < MIN_CANDLES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Minimum {MIN_CANDLES} bougies nécessaires "
                f"pour l'analyse."
            ),
        )

    closes = [c.close for c in request.candles]

    # --------------------------------------------------------
    # Calculs
    # --------------------------------------------------------

    try:
        fast_ema = calculate_ema(
            closes,
            EMA_FAST_PERIOD,
        )

        slow_ema = calculate_ema(
            closes,
            EMA_SLOW_PERIOD,
        )

        rsi = calculate_rsi(
            closes,
            RSI_PERIOD,
        )

        momentum = calculate_momentum(
            closes,
            5,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    trend = determine_trend(
        fast_ema,
        slow_ema,
    )

    # --------------------------------------------------------
    # Signal
    # --------------------------------------------------------

    signal, confidence, score, reasons = generate_signal(
        rsi,
        fast_ema,
        slow_ema,
        momentum,
    )

    # --------------------------------------------------------
    # Dernière bougie
    # --------------------------------------------------------

    last_candle = request.candles[-1]

    current_price = last_candle.close

    # --------------------------------------------------------
    # Volatilité simple
    # --------------------------------------------------------

    recent_candles = request.candles[-10:]

    recent_ranges = [
        candle.high - candle.low
        for candle in recent_candles
    ]

    average_range = (
        sum(recent_ranges) / len(recent_ranges)
        if recent_ranges
        else 0.0
    )

    # --------------------------------------------------------
    # Direction de la dernière bougie
    # --------------------------------------------------------

    if last_candle.close > last_candle.open:
        candle_direction = "BULLISH"

    elif last_candle.close < last_candle.open:
        candle_direction = "BEARISH"

    else:
        candle_direction = "NEUTRAL"

    # --------------------------------------------------------
    # Réponse
    # --------------------------------------------------------

    return {
        "status": "success",
        "asset": request.asset,
        "timeframe": request.timeframe,
        "price": round(current_price, 6),
        "signal": signal,
        "confidence": confidence,
        "score": score,
        "trend": trend,

        "indicators": {
            f"ema{EMA_FAST_PERIOD}": round(
                fast_ema,
                6,
            ),
            f"ema{EMA_SLOW_PERIOD}": round(
                slow_ema,
                6,
            ),
            f"rsi{RSI_PERIOD}": rsi,
            "momentum_5": round(
                momentum,
                6,
            ),
        },

        "market": {
            "candle_direction": candle_direction,
            "average_range": round(
                average_range,
                6,
            ),
        },

        "reasons": reasons,

        "candles_count": len(request.candles),

        "candles": [
            {
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
            }
            for candle in request.candles
        ],
    }


# ============================================================
# QUICK ANALYSIS
# ============================================================

@app.post("/quick-analysis")
def quick_analysis(data: MarketData):

    if data.previous_price is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "previous_price est nécessaire "
                "pour calculer la variation."
            ),
        )

    variation = (
        (data.price - data.previous_price)
        / data.previous_price
    ) * 100

    if variation > 0:
        direction = "UP"

    elif variation < 0:
        direction = "DOWN"

    else:
        direction = "FLAT"

    return {
        "status": "success",
        "asset": data.asset,
        "timeframe": data.timeframe,
        "price": data.price,
        "previous_price": data.previous_price,
        "variation_percent": round(
            variation,
            4,
        ),
        "direction": direction,
    }


# ============================================================
# RUN
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
