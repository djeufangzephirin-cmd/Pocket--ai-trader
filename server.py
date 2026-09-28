import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# CONFIGURATION
# ============================================================

APP_VERSION = "4.0.1"

# Format interne du moteur
ASSET = "EURUSD"
TIMEFRAME = "15m"

# Format Twelve Data
TWELVE_DATA_SYMBOL = "EUR/USD"
TWELVE_DATA_INTERVAL = "15min"

CANDLE_LIMIT = 100


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Pocket AI Trader",
    description="Moteur d'analyse technique Forex avec donnees reelles",
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
# MODELS
# ============================================================

class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class AnalyzeRequest(BaseModel):
    asset: str = Field(default=ASSET)
    timeframe: str = Field(default=TIMEFRAME)
    candles: List[Candle] = Field(min_length=30)


# ============================================================
# TWELVE DATA
# ============================================================

def get_api_key() -> str:

    key = os.getenv("TWELVE_DATA_API_KEY")

    if not key:
        raise HTTPException(
            status_code=500,
            detail=(
                "La variable TWELVE_DATA_API_KEY "
                "n'est pas configuree sur Render."
            ),
        )

    return key


def fetch_candles(
    asset: str = ASSET,
    timeframe: str = TIMEFRAME,
    outputsize: int = CANDLE_LIMIT,
) -> List[Dict[str, float]]:

    key = get_api_key()

    url = "https://api.twelvedata.com/time_series"

    # Conversion format interne -> Twelve Data
    api_symbol = (
        TWELVE_DATA_SYMBOL
        if asset == "EURUSD"
        else asset
    )

    api_interval = (
        TWELVE_DATA_INTERVAL
        if timeframe == "15m"
        else timeframe
    )

    params = {
        "symbol": api_symbol,
        "interval": api_interval,
        "outputsize": outputsize,
        "apikey": key,
        "format": "JSON",
        "order": "ASC",
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=15,
        )

        payload = response.json()

    except requests.RequestException as exc:

        raise HTTPException(
            status_code=502,
            detail=(
                f"Erreur de connexion Twelve Data: "
                f"{exc}"
            ),
        )

    except ValueError:

        raise HTTPException(
            status_code=502,
            detail=(
                "Twelve Data a retourne "
                "une reponse JSON invalide."
            ),
        )

    # Gestion explicite des erreurs Twelve Data
    if payload.get("status") == "error":

        raise HTTPException(
            status_code=502,
            detail=(
                f"Erreur Twelve Data "
                f"{payload.get('code', '')}: "
                f"{payload.get('message', 'Erreur inconnue')}"
            ),
        )

    values = payload.get("values")

    if not values:

        raise HTTPException(
            status_code=502,
            detail=(
                "Twelve Data n'a retourne "
                "aucune bougie."
            ),
        )

    candles: List[Dict[str, float]] = []

    for row in values:

        try:

            candles.append(
                {
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(
                        row.get("volume", 0) or 0
                    ),
                }
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):

            continue

    if len(candles) < 30:

        raise HTTPException(
            status_code=502,
            detail=(
                f"Nombre de bougies insuffisant: "
                f"{len(candles)}"
            ),
        )

    return candles[-outputsize:]


# ============================================================
# DATAFRAME
# ============================================================

def to_dataframe(
    candles: List[Dict[str, float]]
) -> pd.DataFrame:

    df = pd.DataFrame(candles).copy()

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = (
        df.dropna(
            subset=[
                "open",
                "high",
                "low",
                "close",
            ]
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# RSI
# ============================================================

def rsi(
    series: pd.Series,
    period: int = 14,
) -> pd.Series:

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    # Valeur neutre par défaut
    result = pd.Series(
        50.0,
        index=series.index,
        dtype=float,
    )

    # Marché fortement haussier
    bullish = (
        (avg_loss == 0)
        & (avg_gain > 0)
    )

    result.loc[bullish] = 100.0

    # Marché fortement baissier
    bearish = (
        (avg_gain == 0)
        & (avg_loss > 0)
    )

    result.loc[bearish] = 0.0

    # Cas normal
    normal = (
        (avg_gain > 0)
        & (avg_loss > 0)
    )

    rs = (
        avg_gain[normal]
        / avg_loss[normal]
    )

    result.loc[normal] = (
        100
        - (
            100
            / (1 + rs)
        )
    )

    return result


# ============================================================
# ATR
# ============================================================

def atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:

    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (
                df["high"] - prev_close
            ).abs(),
            (
                df["low"] - prev_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


# ============================================================
# INDICATEURS
# ============================================================

def calculate_indicators(
    df: pd.DataFrame,
) -> Dict[str, float]:

    close = df["close"]

    ema9 = close.ewm(
        span=9,
        adjust=False,
    ).mean()

    ema21 = close.ewm(
        span=21,
        adjust=False,
    ).mean()

    rsi14 = rsi(
        close,
        14,
    )

    atr14 = atr(
        df,
        14,
    )

    momentum_5 = (
        close.iloc[-1]
        - close.iloc[-6]
    )

    return {
        "ema9": float(ema9.iloc[-1]),
        "ema21": float(ema21.iloc[-1]),
        "rsi14": float(rsi14.iloc[-1]),
        "momentum_5": float(momentum_5),
        "atr14": float(atr14.iloc[-1]),
    }


# ============================================================
# CONTEXTE MARCHE
# ============================================================

def market_context(
    df: pd.DataFrame,
    indicators: Dict[str, float],
) -> Dict[str, Any]:

    ranges = (
        df["high"]
        - df["low"]
    )

    average_range = float(
        ranges.tail(50).mean()
    )

    recent_average_range = float(
        ranges.tail(10).mean()
    )

    volatility_ratio = (
        recent_average_range
        / average_range
        if average_range > 0
        else 1.0
    )

    last = df.iloc[-1]

    direction = (
        "BULLISH"
        if last["close"] > last["open"]
        else "BEARISH"
        if last["close"] < last["open"]
        else "NEUTRAL"
    )

    if volatility_ratio < 0.70:

        volatility_state = "LOW"

    elif volatility_ratio > 1.30:

        volatility_state = "HIGH"

    else:

        volatility_state = "NORMAL"

    return {
        "candle_direction": direction,
        "average_range": average_range,
        "recent_average_range": recent_average_range,
        "volatility_ratio": volatility_ratio,
        "volatility_state": volatility_state,
    }


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def levels(
    df: pd.DataFrame,
) -> Dict[str, float]:

    window = df.tail(50)

    return {
        "support": float(
            window["low"].min()
        ),
        "resistance": float(
            window["high"].max()
        ),
    }


# ============================================================
# MOTEUR DE DECISION
# 4 BLOCS
# ============================================================

def decision_engine(
    df: pd.DataFrame,
    ind: Dict[str, float],
    market: Dict[str, Any],
    lv: Dict[str, float],
) -> Dict[str, Any]:

    """
    Four blocs:

    1. Tendance
    2. Confirmation
    3. Signal
    4. Trade plan
    """

    close = float(
        df["close"].iloc[-1]
    )

    ema9 = ind["ema9"]
    ema21 = ind["ema21"]

    rsi14 = ind["rsi14"]
    momentum = ind["momentum_5"]
    atr14 = ind["atr14"]

    # ========================================================
    # BLOC 1 — FILTRE DE TENDANCE
    # ========================================================

    trend_score = 0

    trend_reasons: List[str] = []

    if ema9 > ema21:

        trend_score += 1

        trend_reasons.append(
            "EMA9 au-dessus de EMA21"
        )

    elif ema9 < ema21:

        trend_score -= 1

        trend_reasons.append(
            "EMA9 sous EMA21"
        )

    else:

        trend_reasons.append(
            "EMA9 et EMA21 proches"
        )

    if close > ema9:

        trend_score += 1

        trend_reasons.append(
            "Prix au-dessus de EMA9"
        )

    elif close < ema9:

        trend_score -= 1

        trend_reasons.append(
            "Prix sous EMA9"
        )

    trend = (
        "BULLISH"
        if trend_score >= 2
        else "BEARISH"
        if trend_score <= -2
        else "NEUTRAL"
    )

    # ========================================================
    # BLOC 2 — CONFIRMATION
    # ========================================================

    confirmation_score = 0

    reasons: List[str] = []

    if rsi14 >= 55:

        confirmation_score += 1

        reasons.append(
            "RSI haussier"
        )

    elif rsi14 <= 45:

        confirmation_score -= 1

        reasons.append(
            "RSI baissier"
        )

    else:

        reasons.append(
            "RSI neutre"
        )

    if momentum > 0:

        confirmation_score += 1

        reasons.append(
            "Momentum positif"
        )

    elif momentum < 0:

        confirmation_score -= 1

        reasons.append(
            "Momentum négatif"
        )

    if (
        market["candle_direction"]
        == "BULLISH"
    ):

        confirmation_score += 1

        reasons.append(
            "Dernière bougie haussière"
        )

    elif (
        market["candle_direction"]
        == "BEARISH"
    ):

        confirmation_score -= 1

        reasons.append(
            "Dernière bougie baissière"
        )

    if (
        market["volatility_state"]
        == "LOW"
    ):

        reasons.append(
            "Volatilité faible"
        )

    elif (
        market["volatility_state"]
        == "HIGH"
    ):

        reasons.append(
            "Volatilité élevée"
        )

    # ========================================================
    # BLOC 3 — SCORE / SIGNAL
    # ========================================================

    score = (
        trend_score
        + confirmation_score
    )

    # Une faible volatilité réduit
    # la conviction mais n'inverse
    # jamais le signal.

    if (
        market["volatility_state"]
        == "LOW"
        and abs(score) >= 3
    ):

        score = int(
            np.sign(score) * 2
        )

        reasons.append(
            "Filtre de volatilité: "
            "conviction réduite"
        )

    if score >= 3:

        signal = "BUY"

    elif score <= -3:

        signal = "SELL"

    else:

        signal = "WAIT"

    confidence = (
        50
        + abs(score) * 8
    )

    if signal == "WAIT":

        confidence = (
            50
            + min(
                abs(score) * 8,
                16,
            )
        )

    confidence = int(
        min(
            95,
            max(
                50,
                confidence,
            ),
        )
    )

    # ========================================================
    # BLOC 4 — PLAN DE TRADE
    # ========================================================

    entry = close

    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward: Optional[float] = None
    stop_distance: Optional[float] = None
    target_distance: Optional[float] = None

    recent = df.tail(20)

    if signal == "BUY":

        structural_sl = float(
            recent["low"].min()
        )

        atr_sl = (
            entry
            - (1.5 * atr14)
        )

        stop_loss = min(
            structural_sl,
            atr_sl,
        )

        stop_distance = (
            entry
            - stop_loss
        )

        if stop_distance > 0:

            target_distance = (
                stop_distance * 2.0
            )

            take_profit = (
                entry
                + target_distance
            )

            risk_reward = 2.0

    elif signal == "SELL":

        structural_sl = float(
            recent["high"].max()
        )

        atr_sl = (
            entry
            + (1.5 * atr14)
        )

        stop_loss = max(
            structural_sl,
            atr_sl,
        )

        stop_distance = (
            stop_loss
            - entry
        )

        if stop_distance > 0:

            target_distance = (
                stop_distance * 2.0
            )

            take_profit = (
                entry
                - target_distance
            )

            risk_reward = 2.0

    return {
        "trend": trend,
        "score": int(score),
        "signal": signal,
        "confidence": confidence,
        "reasons": (
            trend_reasons
            + reasons
        ),
        "trade_plan": {
            "entry": entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward": risk_reward,
            "stop_distance": stop_distance,
            "target_distance": target_distance,
        },
    }


# ============================================================
# ANALYSE
# ============================================================

def analyze_candles(
    candles: List[Dict[str, float]],
    asset: str,
    timeframe: str,
) -> Dict[str, Any]:

    if len(candles) < 30:

        raise HTTPException(
            status_code=422,
            detail=(
                "Au moins 30 bougies "
                "sont necessaires."
            ),
        )

    df = to_dataframe(
        candles
    )

    if len(df) < 30:

        raise HTTPException(
            status_code=422,
            detail=(
                "Bougies invalides "
                "ou insuffisantes."
            ),
        )

    ind = calculate_indicators(
        df
    )

    market = market_context(
        df,
        ind,
    )

    lv = levels(
        df
    )

    decision = decision_engine(
        df,
        ind,
        market,
        lv,
    )

    return {
        "status": "success",
        "engine_version": APP_VERSION,
        "asset": asset,
        "timeframe": timeframe,
        "price": float(
            df["close"].iloc[-1]
        ),
        "signal": decision["signal"],
        "confidence": decision[
            "confidence"
        ],
        "score": decision["score"],
        "trend": decision["trend"],
        "indicators": {
            k: round(v, 8)
            for k, v in ind.items()
        },
        "market": {
            "candle_direction": market[
                "candle_direction"
            ],
            "average_range": round(
                market["average_range"],
                8,
            ),
            "recent_average_range": round(
                market[
                    "recent_average_range"
                ],
                8,
            ),
            "volatility_ratio": round(
                market[
                    "volatility_ratio"
                ],
                3,
            ),
            "volatility_state": market[
                "volatility_state"
            ],
        },
        "levels": {
            k: round(v, 8)
            for k, v in lv.items()
        },
        "trade_plan": {
            k: (
                None
                if v is None
                else round(
                    float(v),
                    8,
                )
            )
            for k, v in decision[
                "trade_plan"
            ].items()
        },
        "reasons": decision[
            "reasons"
        ],
        "candles_count": len(df),
        "candles": (
            df[
                [
                    "open",
                    "high",
                    "low",
                    "close",
                ]
            ]
            .round(8)
            .to_dict(
                orient="records"
            )
        ),
    }


# ============================================================
# ROUTE RACINE
# ============================================================

@app.get("/")
def root() -> Dict[str, str]:

    return {
        "name": "Pocket AI Trader",
        "version": APP_VERSION,
        "status": "online",
        "engine": "4-bloc decision engine",
    }


# ============================================================
# MARKET DATA
# ============================================================

@app.get("/market-data")
def market_data() -> Dict[str, Any]:

    candles = fetch_candles()

    return {
        "status": "success",
        "source": "Twelve Data",
        "asset": ASSET,
        "timeframe": TIMEFRAME,
        "candles_count": len(candles),
        "candles": candles,
    }


# ============================================================
# ANALYSE LIVE
# ============================================================

@app.get("/analyze-live")
def analyze_live() -> Dict[str, Any]:

    candles = fetch_candles()

    return analyze_candles(
        candles,
        ASSET,
        TIMEFRAME,
    )


# ============================================================
# ANALYSE MANUELLE
# ============================================================

@app.post("/analyze")
def analyze(
    request: AnalyzeRequest,
) -> Dict[str, Any]:

    return analyze_candles(
        [
            c.model_dump()
            for c in request.candles
        ],
        request.asset,
        request.timeframe,
    )
