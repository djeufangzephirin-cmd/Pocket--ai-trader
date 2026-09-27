from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Pocket AI Trader",
    description="Moteur d'analyse technique Forex",
    version="2.2.0"
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

    if not values:
        return 0.0

    if len(values) < period:
        return sum(values) / len(values)

    multiplier = 2 / (period + 1)

    ema = sum(values[:period]) / period

    for price in values[period:]:
        ema = ((price - ema) * multiplier) + ema

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
            losses.append(0.0)

        elif change < 0:
            gains.append(0.0)
            losses.append(abs(change))

        else:
            gains.append(0.0)
            losses.append(0.0)

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):

        avg_gain = (
            (avg_gain * (period - 1)) + gains[i]
        ) / period

        avg_loss = (
            (avg_loss * (period - 1)) + losses[i]
        ) / period

    if avg_loss == 0:

        if avg_gain == 0:
            return 50.0

        return 100.0

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return max(0.0, min(100.0, rsi))


def calculate_momentum(
    values: List[float],
    period: int = 5
) -> float:

    if len(values) <= period:
        return 0.0

    previous = values[-period - 1]

    if previous <= 0:
        return 0.0

    return ((values[-1] - previous) / previous) * 100


# ============================================================
# UTILITAIRES
# ============================================================

def clamp(value: int, minimum: int, maximum: int) -> int:

    return max(minimum, min(maximum, value))


# ============================================================
# ANALYSE
# ============================================================

@app.post("/analyze")
def analyze_market(data: AnalyzeRequest):

    candles = data.candles

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if len(candles) < 25:

        raise HTTPException(
            status_code=400,
            detail="Au moins 25 bougies sont nécessaires."
        )

    closes = [c.close for c in candles]

    # ========================================================
    # INDICATEURS
    # ========================================================

    ema9 = calculate_ema(closes, 9)

    ema21 = calculate_ema(closes, 21)

    rsi14 = calculate_rsi(closes, 14)

    momentum5 = calculate_momentum(closes, 5)

    last_candle = candles[-1]

    # ========================================================
    # DIRECTION DERNIERE BOUGIE
    # ========================================================

    if last_candle.close > last_candle.open:

        candle_direction = "BULLISH"

    elif last_candle.close < last_candle.open:

        candle_direction = "BEARISH"

    else:

        candle_direction = "NEUTRAL"

    # ========================================================
    # RANGE MOYEN
    # ========================================================

    ranges = [
        max(0.0, candle.high - candle.low)
        for candle in candles
    ]

    average_range = sum(ranges) / len(ranges)

    # Range des 5 dernières bougies
    recent_ranges = ranges[-5:]

    recent_average_range = (
        sum(recent_ranges) / len(recent_ranges)
    )

    # ========================================================
    # SCORE
    # ========================================================

    score = 0

    reasons = []

    # ========================================================
    # 1. EMA9 / EMA21
    # ========================================================

    EMA_SEPARATION_THRESHOLD = 0.0005

    ema_difference = abs(ema9 - ema21)

    ema_threshold = closes[-1] * EMA_SEPARATION_THRESHOLD

    if ema_difference <= ema_threshold:

        ema_signal = 0

        ema_state = "NEUTRAL"

        reasons.append(
            "EMA9 et EMA21 proches - zone neutre"
        )

    elif ema9 > ema21:

        ema_signal = 2

        ema_state = "BULLISH"

        score += 2

        reasons.append(
            "EMA9 au-dessus EMA21"
        )

    else:

        ema_signal = -2

        ema_state = "BEARISH"

        score -= 2

        reasons.append(
            "EMA9 sous EMA21"
        )

    # ========================================================
    # 2. RSI - CONFIRMATION CONTEXTUELLE
    # ========================================================
    #
    # On ne donne plus automatiquement +1 lorsque RSI > 70
    # ou -1 lorsque RSI < 30.
    #
    # RSI extrême indique également un risque de retournement.
    # On utilise donc une zone de confirmation plus modérée.
    #
    # 50-70 : biais haussier
    # 30-50 : biais baissier
    # >70   : surachat / prudence
    # <30   : survente / prudence
    #
    # ========================================================

    rsi_signal = 0

    if 55 <= rsi14 < 70:

        rsi_signal = 1

        score += 1

        reasons.append(
            "RSI confirme une pression haussière"
        )

    elif 30 < rsi14 <= 45:

        rsi_signal = -1

        score -= 1

        reasons.append(
            "RSI confirme une pression baissière"
        )

    elif rsi14 >= 70:

        reasons.append(
            "RSI en surachat - risque de correction"
        )

    elif rsi14 <= 30:

        reasons.append(
            "RSI en survente - risque de rebond"
        )

    else:

        reasons.append(
            "RSI neutre"
        )

    # ========================================================
    # 3. MOMENTUM - INTENSITE
    # ========================================================

    momentum_signal = 0

    # Faible mouvement
    if abs(momentum5) < 0.03:

        momentum_signal = 0

        reasons.append(
            "Momentum faible"
        )

    # Mouvement modéré
    elif 0.03 <= momentum5 < 0.15:

        momentum_signal = 1

        score += 1

        reasons.append(
            "Momentum haussier modéré"
        )

    elif -0.15 < momentum5 <= -0.03:

        momentum_signal = -1

        score -= 1

        reasons.append(
            "Momentum baissier modéré"
        )

    # Mouvement fort
    elif momentum5 >= 0.15:

        momentum_signal = 2

        score += 2

        reasons.append(
            "Momentum haussier fort"
        )

    elif momentum5 <= -0.15:

        momentum_signal = -2

        score -= 2

        reasons.append(
            "Momentum baissier fort"
        )

    # ========================================================
    # 4. DERNIERE BOUGIE
    # ========================================================

    candle_signal = 0

    if candle_direction == "BULLISH":

        candle_signal = 1

        score += 1

        reasons.append(
            "Dernière bougie haussière"
        )

    elif candle_direction == "BEARISH":

        candle_signal = -1

        score -= 1

        reasons.append(
            "Dernière bougie baissière"
        )

    else:

        reasons.append(
            "Dernière bougie neutre"
        )

    # ========================================================
    # 5. VOLATILITE
    # ========================================================
    #
    # La volatilité ne crée pas directement un BUY ou SELL.
    # Elle sert principalement à éviter de qualifier un marché
    # extrêmement faible comme un signal fort.
    #
    # ========================================================

    volatility_ratio = 0.0

    if average_range > 0:

        volatility_ratio = (
            recent_average_range / average_range
        )

    if average_range == 0:

        volatility_state = "VERY_LOW"

    elif volatility_ratio < 0.50:

        volatility_state = "LOW"

    elif volatility_ratio > 1.80:

        volatility_state = "HIGH"

    else:

        volatility_state = "NORMAL"

    if volatility_state == "VERY_LOW":

        reasons.append(
            "Volatilité très faible"
        )

    elif volatility_state == "LOW":

        reasons.append(
            "Volatilité faible"
        )

    elif volatility_state == "HIGH":

        reasons.append(
            "Volatilité élevée"
        )

    else:

        reasons.append(
            "Volatilité normale"
        )

    # ========================================================
    # NORMALISATION DU SCORE
    # ========================================================

    score = clamp(score, -5, 5)

    # ========================================================
    # TENDANCE
    # ========================================================

    if ema_state == "BULLISH":

        trend = "BULLISH"

    elif ema_state == "BEARISH":

        trend = "BEARISH"

    else:

        if score >= 3:

            trend = "BULLISH"

        elif score <= -3:

            trend = "BEARISH"

        else:

            trend = "NEUTRAL"

    # ========================================================
    # FILTRE DE CONTRADICTION
    # ========================================================
    #
    # Si EMA et momentum s'opposent fortement, on évite de
    # produire immédiatement un signal directionnel.
    #
    # ========================================================

    strong_contradiction = False

    if ema_state == "BULLISH" and momentum_signal <= -2:

        strong_contradiction = True

        reasons.append(
            "Contradiction forte entre tendance EMA et momentum"
        )

    elif ema_state == "BEARISH" and momentum_signal >= 2:

        strong_contradiction = True

        reasons.append(
            "Contradiction forte entre tendance EMA et momentum"
        )

    # ========================================================
    # SIGNAL
    # ========================================================

    if strong_contradiction:

        signal = "WAIT"

    elif score >= 3:

        signal = "BUY"

    elif score <= -3:

        signal = "SELL"

    else:

        signal = "WAIT"

    # ========================================================
    # FILTRE RSI EXTREME
    # ========================================================
    #
    # Un RSI > 70 ne provoque plus automatiquement un BUY.
    # Un RSI < 30 ne provoque plus automatiquement un SELL.
    #
    # ========================================================

    if signal == "BUY" and rsi14 >= 75:

        reasons.append(
            "BUY limité par RSI fortement suracheté"
        )

        signal = "WAIT"

    elif signal == "SELL" and rsi14 <= 25:

        reasons.append(
            "SELL limité par RSI fortement survendu"
        )

        signal = "WAIT"

    # ========================================================
    # FILTRE DE VOLATILITE EXTREMEMENT FAIBLE
    # ========================================================

    if signal in ("BUY", "SELL"):

        if average_range > 0:

            # Seuil volontairement prudent.
            # Pour EURUSD autour de 1.08, cela correspond
            # approximativement à 1.6 pip de range moyen.

            MIN_RANGE_RATIO = 0.00015

            range_ratio = average_range / closes[-1]

            if range_ratio < MIN_RANGE_RATIO:

                reasons.append(
                    "Volatilité trop faible pour un signal fort"
                )

                signal = "WAIT"

    # ========================================================
    # CONFIANCE
    # ========================================================
    #
    # Ceci est un indice de force du signal.
    # Ce n'est PAS une probabilité de gain.
    #
    # ========================================================

    if signal == "WAIT":

        confidence = 50

    else:

        confidence = 50 + (abs(score) * 8)

        if volatility_state == "HIGH":

            confidence -= 5

        if rsi14 >= 70 or rsi14 <= 30:

            confidence -= 5

        confidence = max(50, min(confidence, 90))

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

            "average_range": round(
                average_range,
                6
            ),

            "recent_average_range": round(
                recent_average_range,
                6
            ),

            "volatility_state": volatility_state

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

        "version": "2.2.0",

        "endpoint": "/analyze"

                  }
