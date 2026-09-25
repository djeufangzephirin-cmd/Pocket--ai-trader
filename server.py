from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from statistics import mean

app = FastAPI(
    title="Pocket AI Trader",
    description="API d'analyse technique de marché",
    version="2.0.0"
)


# =========================
# MODÈLES DE DONNÉES
# =========================

class Candle(BaseModel):
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)


class MarketData(BaseModel):
    asset: str
    price: float = Field(gt=0)
    previous_price: Optional[float] = Field(default=None, gt=0)
    timeframe: str = "1m"


class AnalysisRequest(BaseModel):
    asset: str
    timeframe: str = "1m"
    candles: List[Candle]


# =========================
# FONCTIONS TECHNIQUES
# =========================

def calculate_ema(values: List[float], period: int) -> float:
    if len(values) < period:
        return mean(values)

    multiplier = 2 / (period + 1)
    ema = mean(values[:period])

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

    avg_gain = mean(gains[-period:])
    avg_loss = mean(losses[-period:])

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def analyze_market(candles: List[Candle]):
    closes = [c.close for c in candles]

    if len(closes) < 20:
        raise HTTPException(
            status_code=400,
            detail="Il faut au minimum 20 bougies pour analyser le marché."
        )

    current_price = closes[-1]

    ema9 = calculate_ema(closes, 9)
    ema20 = calculate_ema(closes, 20)
    rsi = calculate_rsi(closes)

    # Momentum récent
    momentum = ((current_price - closes[-5]) / closes[-5]) * 100

    score = 0

    # Tendance EMA
    if ema9 > ema20:
        score += 2
    elif ema9 < ema20:
        score -= 2

    # Momentum
    if momentum > 0:
        score += 1
    elif momentum < 0:
        score -= 1

    # RSI
    if 50 < rsi < 70:
        score += 1
    elif 30 < rsi < 50:
        score -= 1

    # Décision
    if score >= 3:
        signal = "CALL"
        confidence = min(95, 60 + score * 7)

    elif score <= -3:
        signal = "PUT"
        confidence = min(95, 60 + abs(score) * 7)

    else:
        signal = "WAIT"
        confidence = 50

    return {
        "signal": signal,
        "confidence": round(confidence, 2),
        "score": score,
        "price": round(current_price, 6),
        "ema9": round(ema9, 6),
        "ema20": round(ema20, 6),
        "rsi": round(rsi, 2),
        "momentum_percent": round(momentum, 4)
    }


# =========================
# ROUTES
# =========================

@app.get("/")
def home():
    return {
        "status": "online",
        "service": "Pocket AI Trader",
        "version": "2.0.0",
        "message": "Serveur opérationnel"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Pocket AI Trader"
    }


@app.post("/analyze")
def analyze(request: AnalysisRequest):
    result = analyze_market(request.candles)

    return {
        "asset": request.asset,
        "timeframe": request.timeframe,
        "analysis": result
    }


@app.post("/quick-analysis")
def quick_analysis(data: MarketData):
    if data.previous_price is None:
        raise HTTPException(
            status_code=400,
            detail="previous_price est obligatoire."
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
        "asset": data.asset,
        "timeframe": data.timeframe,
        "price": data.price,
        "previous_price": data.previous_price,
        "variation_percent": round(variation, 4),
        "direction": direction
    }
