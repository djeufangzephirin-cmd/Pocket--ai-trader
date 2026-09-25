from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import math


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Pocket AI Trader",
    description="API d'analyse technique pour Pocket AI Trader",
    version="2.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODELS
# ============================================================

class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float


class AnalyzeRequest(BaseModel):
    asset: str = Field(..., example="EURUSD")
    timeframe: str = Field(..., example="1m")
    candles: List[Candle]


class MarketData(BaseModel):
    asset: str = Field(..., example="EURUSD")
    price: float = Field(..., gt=0)
    previous_price: Optional[float] = Field(None, gt=0)
    timeframe: str = Field("1m", example="1m")


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "name": "Pocket AI Trader",
        "version": "2.0.0",
        "status": "online",
        "message": "Pocket AI Trader API is running",
        "endpoints": [
            "/",
            "/health",
            "/analyze",
            "/quick-analysis",
            "/docs"
        ]
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "pocket-ai-trader",
        "version": "2.0.0"
    }


# ============================================================
# EMA
# ============================================================

def calculate_ema(values: List[float], period: int) -> float:
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

def calculate_rsi(values: List[float], period: int = 14) -> float:

    if len(values) < period + 1:
        raise ValueError(
            f"Il faut au moins {period + 1} clôtures pour calculer le RSI."
        )

    gains = []
    losses = []

    for i in range(1, len(values)):
        difference = values[i] - values[i - 1]

        if difference > 0:
            gains.append(difference)
            losses.append(0)
        else:
            gains.append(0)
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

def calculate_momentum(values: List[float], period: int = 5) -> float:

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

def determine_trend(ema9: float, ema20: float) -> str:

    if ema9 > ema20:
        return "BULLISH"

    if ema9 < ema20:
        return "BEARISH"

    return "NEUTRAL"


# ============================================================
# SIGNAL ANALYSIS
# ============================================================

def generate_signal(
    rsi: float,
    ema9: float,
    ema20: float,
    momentum: float
):

    score = 0
    reasons = []

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    if ema9 > ema20:
        score += 2
        reasons.append("EMA9 au-dessus de EMA20")

    elif ema9 < ema20:
        score -= 2
        reasons.append("EMA9 sous EMA20")

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
    # SIGNAL
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

    # Score théorique maximum = +4 / -4
    confidence = min(abs(score) / 4 * 100, 95)

    # WAIT ne doit pas afficher une confiance artificiellement élevée
    if signal == "WAIT":
        confidence = min(confidence, 55)

    confidence = round(confidence, 1)

    return signal, confidence, score, reasons


# ============================================================
# ANALYZE
# ============================================================

@app.post("/analyze")
def analyze_market(request: AnalyzeRequest):

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if len(request.candles) < 20:
        raise HTTPException(
            status_code=400,
            detail="Minimum 20 bougies nécessaires pour l'analyse."
        )

    closes = [c.close for c in request.candles]

    highs = [c.high for c in request.candles]
    lows = [c.low for c in request.candles]

    # --------------------------------------------------------
    # Calculs
    # --------------------------------------------------------

    try:
        ema9 = calculate_ema(closes, 9)
        ema20 = calculate_ema(closes, 20)
        rsi14 = calculate_rsi(closes, 14)
        momentum = calculate_momentum(closes, 5)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    trend = determine_trend(
        ema9,
        ema20
    )

    # --------------------------------------------------------
    # Signal
    # --------------------------------------------------------

    signal, confidence, score, reasons = generate_signal(
        rsi14,
        ema9,
        ema20,
        momentum
    )

    # --------------------------------------------------------
    # Dernière bougie
    # --------------------------------------------------------

    last_candle = request.candles[-1]

    current_price = last_candle.close

    # --------------------------------------------------------
    # Volatilité simple
    # --------------------------------------------------------

    recent_ranges = [
        candle.high - candle.low
        for candle in request.candles[-10:]
    ]

    average_range = (
        sum(recent_ranges) / len(recent_ranges)
        if recent_ranges
        else 0
    )

    # --------------------------------------------------------
    # Direction du prix
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
            "rsi14": rsi14,
            "ema9": round(ema9, 6),
            "ema20": round(ema20, 6),
            "momentum_5": round(momentum, 6)
        },

        "market": {
            "candle_direction": candle_direction,
            "average_range": round(average_range, 6)
        },

        "reasons": reasons,

        "candles_count": len(request.candles),

        "candles": [
            {
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close
            }
            for candle in request.candles
        ]
    }


# ============================================================
# QUICK ANALYSIS
# ============================================================

@app.post("/quick-analysis")
def quick_analysis(data: MarketData):

    if data.previous_price is None:
        raise HTTPException(
            status_code=400,
            detail="previous_price est nécessaire pour calculer la variation."
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
        "variation_percent": round(variation, 4),
        "direction": direction
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
