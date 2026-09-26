from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import math


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Pocket AI Trader",
    description="Moteur d'analyse technique Forex",
    version="2.0.0"
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
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: float = Field(default=0, ge=0)


class AnalyzeRequest(BaseModel):
    asset: str = "EURUSD"
    timeframe: str = "15m"
    candles: List[Candle]


# ============================================================
# INDICATEURS
# ============================================================

def calculate_ema(values: List[float], period: int) -> float:
    if len(values) < period:
        return sum(values) / len(values)

    multiplier = 2 / (period + 1)

    ema = sum(values[:period]) / period

    for price in values[period:]:
        ema = (price - ema) * multiplier + ema

    return ema


def calculate_rsi(values: List[float], period: int = 14) -> float:
    if len(values) <= period:
        return 50.0

    gains = []
    losses = []

    for i in range(1, len(values)):
        change = values[i] - values[i - 1]

        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period

    if avg_loss == 0:
        if avg_gain == 0:
            return 50.0
        return 100.0

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def calculate_momentum(values: List[float], period: int = 5) -> float:
    if len(values) <= period:
        return 0.0

    previous = values[-period - 1]

    if previous == 0:
        return 0.0

    return ((values[-1] - previous) / previous) * 100


# ============================================================
# ANALYSE
# ============================================================

@app.post("/analyze")
def analyze_market(data: AnalyzeRequest):

    candles = data.candles

    if len(candles) < 25:
        raise HTTPException(
            status_code=400,
            detail="Au moins 25 bougies sont nécessaires."
        )

    closes = [c.close for c in candles]

    # --------------------------------------------------------
    # INDICATEURS
    # --------------------------------------------------------

    ema9 = calculate_ema(closes, 9)
    ema21 = calculate_ema(closes, 21)
    rsi14 = calculate_rsi(closes, 14)
    momentum5 = calculate_momentum(closes, 5)

    last_candle = candles[-1]

    # --------------------------------------------------------
    # DIRECTION DE LA DERNIERE BOUGIE
    # --------------------------------------------------------

    if last_candle.close > last_candle.open:
        candle_direction = "BULLISH"

    elif last_candle.close < last_candle.open:
        candle_direction = "BEARISH"

    else:
        candle_direction = "NEUTRAL"

    # --------------------------------------------------------
    # RANGE MOYEN
    # --------------------------------------------------------

    ranges = [
        candle.high - candle.low
        for candle in candles
    ]

    average_range = sum(ranges) / len(ranges)

    # ========================================================
    # SCORE
    # ========================================================

    score = 0
    reasons = []

    # --------------------------------------------------------
    # 1. EMA
    # --------------------------------------------------------

    if ema9 > ema21:

        score += 2

        reasons.append("EMA9 au-dessus EMA21")

    elif ema9 < ema21:

        score -= 2

        reasons.append("EMA9 sous EMA21")

    # --------------------------------------------------------
    # 2. RSI
    # --------------------------------------------------------

    if rsi14 >= 70:

        score += 1

        reasons.append("RSI en zone de surachat")

    elif rsi14 <= 30:

        score -= 1

        reasons.append("RSI en zone de survente")

    # --------------------------------------------------------
    # 3. MOMENTUM
    # --------------------------------------------------------

    if momentum5 > 0:

        score += 1

        reasons.append("Momentum positif")

    elif momentum5 < 0:

        score -= 1

        reasons.append("Momentum négatif")

    # --------------------------------------------------------
    # 4. CONFIRMATION PAR LA DERNIERE BOUGIE
    # --------------------------------------------------------

    if candle_direction == "BULLISH":

        score += 1

        reasons.append("Dernière bougie haussière")

    elif candle_direction == "BEARISH":

        score -= 1

        reasons.append("Dernière bougie baissière")

    # ========================================================
    # TENDANCE
    # ========================================================

    if ema9 > ema21:
        trend = "BULLISH"

    elif ema9 < ema21:
        trend = "BEARISH"

    else:
        trend = "NEUTRAL"

    # ========================================================
    # SIGNAL
    # ========================================================

    if score >= 3:

        signal = "BUY"

    elif score <= -3:

        signal = "SELL"

    else:

        signal = "WAIT"

    # ========================================================
    # CONFIANCE
    # ========================================================

    if signal == "WAIT":

        confidence = 50

    else:

        confidence = 50 + (abs(score) * 10)

        # Limite maximale
        confidence = min(confidence, 95)

    # ========================================================
    # REPONSE
    # ========================================================

    return {
        "status": "success",
        "asset": data.asset,
        "timeframe": data.timeframe,
        "price": closes[-1],
        "signal": signal,
        "confidence": confidence,
        "score": score,
        "trend": trend,

        "indicators": {
            "ema9": round(ema9, 6),
            "ema21": round(ema21, 6),
            "rsi14": round(rsi14, 6),
            "momentum_5": round(momentum5, 6)
        },

        "market": {
            "candle_direction": candle_direction,
            "average_range": round(average_range, 6)
        },

        "reasons": reasons,

        "candles_count": len(candles),

        "candles": [
            {
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close
            }
            for c in candles
        ]
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "Pocket AI Trader",
        "version": "2.0.0",
        "endpoint": "/analyze"
    }
