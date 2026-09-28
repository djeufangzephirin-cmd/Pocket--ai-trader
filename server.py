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

APP_VERSION = "4.1.0"

ASSET = "EURUSD"
TIMEFRAME = "15m"

TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"

DEFAULT_CANDLES = 100
MIN_CANDLES = 30
MAX_CANDLES = 500


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Pocket AI Trader",
    description="Trading analysis engine - EURUSD 15m",
    version=APP_VERSION,
)

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

class AnalyzeRequest(BaseModel):
    asset: str = Field(default=ASSET)
    timeframe: str = Field(default=TIMEFRAME)
    candles: int = Field(
        default=DEFAULT_CANDLES,
        ge=MIN_CANDLES,
        le=MAX_CANDLES,
    )


# ============================================================
# UTILITAIRES
# ============================================================

def normalize_asset(asset: str) -> str:
    asset = asset.strip().upper()

    if "/" in asset:
        return asset

    if len(asset) == 6:
        return f"{asset[:3]}/{asset[3:]}"

    return asset


def normalize_interval(timeframe: str) -> str:
    timeframe = timeframe.strip().lower()

    mapping = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1day",
    }

    return mapping.get(timeframe, timeframe)


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None

        result = float(value)

        if not np.isfinite(result):
            return None

        return result

    except (ValueError, TypeError):
        return None


# ============================================================
# TWELVE DATA
# ============================================================

def get_api_key() -> str:
    api_key = os.getenv("TWELVE_DATA_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail=(
                "La variable TWELVE_DATA_API_KEY "
                "n'est pas configurée sur Render."
            ),
        )

    return api_key


def fetch_candles(
    asset: str = ASSET,
    timeframe: str = TIMEFRAME,
    outputsize: int = DEFAULT_CANDLES,
) -> pd.DataFrame:

    api_key = get_api_key()

    symbol = normalize_asset(asset)
    interval = normalize_interval(timeframe)

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": api_key,
        "format": "JSON",
    }

    try:
        response = requests.get(
            TWELVE_DATA_URL,
            params=params,
            timeout=20,
        )

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Erreur de connexion à Twelve Data: {exc}",
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Twelve Data HTTP "
                f"{response.status_code}: "
                f"{response.text[:500]}"
            ),
        )

    try:
        payload = response.json()

    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="Réponse Twelve Data invalide.",
        )

    if payload.get("status") == "error":
        message = payload.get(
            "message",
            "Erreur inconnue Twelve Data.",
        )

        raise HTTPException(
            status_code=502,
            detail=f"Twelve Data: {message}",
        )

    values = payload.get("values")

    if not values:
        raise HTTPException(
            status_code=502,
            detail="Twelve Data n'a retourné aucune bougie.",
        )

    rows: List[Dict[str, Any]] = []

    for item in values:
        try:
            rows.append(
                {
                    "datetime": item.get("datetime"),
                    "open": float(item["open"]),
                    "high": float(item["high"]),
                    "low": float(item["low"]),
                    "close": float(item["close"]),
                    "volume": float(
                        item.get("volume", 0) or 0
                    ),
                }
            )

        except (KeyError, TypeError, ValueError):
            continue

    if len(rows) < MIN_CANDLES:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Données insuffisantes: "
                f"{len(rows)} bougies reçues. "
                f"Minimum requis: {MIN_CANDLES}."
            ),
        )

    df = pd.DataFrame(rows)

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["datetime"]
    )

    df = df.sort_values(
        "datetime",
        ascending=True,
    ).reset_index(drop=True)

    return df


# ============================================================
# INDICATEURS
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

    return rsi.fillna(50)


def calculate_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:

    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]

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

    df["ema9"] = calculate_ema(
        close,
        9,
    )

    df["ema21"] = calculate_ema(
        close,
        21,
    )

    df["rsi14"] = calculate_rsi(
        close,
        14,
    )

    df["momentum5"] = (
        close.pct_change(5) * 100
    )

    df["atr14"] = calculate_atr(
        df,
        14,
    )

    macd = calculate_macd(
        close
    )

    df["macd"] = macd["macd"]
    df["macd_signal"] = macd["signal"]
    df["macd_histogram"] = macd["histogram"]

    latest = df.iloc[-1]

    return {
        "ema9": safe_float(
            latest["ema9"]
        ) or 0.0,

        "ema21": safe_float(
            latest["ema21"]
        ) or 0.0,

        "rsi14": safe_float(
            latest["rsi14"]
        ) or 50.0,

        "momentum_5": safe_float(
            latest["momentum5"]
        ) or 0.0,

        "atr14": safe_float(
            latest["atr14"]
        ) or 0.0,

        "macd": safe_float(
            latest["macd"]
        ) or 0.0,

        "macd_signal": safe_float(
            latest["macd_signal"]
        ) or 0.0,

        "macd_histogram": safe_float(
            latest["macd_histogram"]
        ) or 0.0,
    }


# ============================================================
# SCORE DE MARCHÉ
# ============================================================

def calculate_market_score(
    indicators: Dict[str, float],
) -> Dict[str, Any]:

    ema9 = indicators["ema9"]
    ema21 = indicators["ema21"]
    rsi = indicators["rsi14"]
    momentum = indicators["momentum_5"]
    macd = indicators["macd"]
    macd_signal = indicators["macd_signal"]
    macd_histogram = indicators["macd_histogram"]

    score = 0

    reasons: List[str] = []

    # EMA
    if ema9 > ema21:
        score += 2
        reasons.append(
            "EMA9 > EMA21"
        )

    elif ema9 < ema21:
        score -= 2
        reasons.append(
            "EMA9 < EMA21"
        )

    # RSI
    if 52 <= rsi <= 70:
        score += 1
        reasons.append(
            "RSI favorable aux acheteurs"
        )

    elif 30 <= rsi <= 48:
        score -= 1
        reasons.append(
            "RSI favorable aux vendeurs"
        )

    elif rsi > 70:
        score -= 1
        reasons.append(
            "RSI en zone de surachat"
        )

    elif rsi < 30:
        score += 1
        reasons.append(
            "RSI en zone de survente"
        )

    # Momentum
    if momentum > 0:
        score += 1
        reasons.append(
            "Momentum positif"
        )

    elif momentum < 0:
        score -= 1
        reasons.append(
            "Momentum négatif"
        )

    # MACD
    if (
        macd > macd_signal
        and macd_histogram > 0
    ):
        score += 1
        reasons.append(
            "MACD haussier"
        )

    elif (
        macd < macd_signal
        and macd_histogram < 0
    ):
        score -= 1
        reasons.append(
            "MACD baissier"
        )

    # Tendance
    if score >= 3:
        trend = "BULLISH"

    elif score <= -3:
        trend = "BEARISH"

    else:
        trend = "NEUTRAL"

    # Signal
    if score >= 4:
        signal = "BUY"

    elif score <= -4:
        signal = "SELL"

    else:
        signal = "WAIT"

    # Confidence
    confidence = int(
        max(
            50,
            min(
                95,
                50 + abs(score) * 8,
            ),
        )
    )

    # Qualité setup
    if abs(score) >= 5:
        setup_state = "STRONG"
        setup_score = 90

    elif abs(score) >= 4:
        setup_state = "GOOD"
        setup_score = 75

    elif abs(score) >= 2:
        setup_state = "MODERATE"
        setup_score = 60

    else:
        setup_state = "WEAK"
        setup_score = 40

    return {
        "score": score,
        "signal": signal,
        "confidence": confidence,
        "trend": trend,
        "reasons": reasons,
        "setup_quality": {
            "score": setup_score,
            "state": setup_state,
        },
    }


# ============================================================
# TRADE PLAN
# ============================================================

def build_trade_plan(
    signal: str,
    price: float,
    atr: float,
) -> Optional[Dict[str, Any]]:

    if signal not in [
        "BUY",
        "SELL",
    ]:
        return None

    if atr <= 0:
        return None

    stop_distance = atr * 1.5

    tp1_distance = (
        stop_distance * 1.5
    )

    tp2_distance = (
        stop_distance * 2.5
    )

    if signal == "BUY":

        stop_loss = (
            price - stop_distance
        )

        take_profit_1 = (
            price + tp1_distance
        )

        take_profit_2 = (
            price + tp2_distance
        )

    else:

        stop_loss = (
            price + stop_distance
        )

        take_profit_1 = (
            price - tp1_distance
        )

        take_profit_2 = (
            price - tp2_distance
        )

    risk = abs(
        price - stop_loss
    )

    reward_1 = abs(
        take_profit_1 - price
    )

    reward_2 = abs(
        take_profit_2 - price
    )

    rr1 = (
        reward_1 / risk
        if risk
        else 0
    )

    rr2 = (
        reward_2 / risk
        if risk
        else 0
    )

    return {
        "direction": signal,
        "entry": round(
            price,
            5,
        ),
        "stop_loss": round(
            stop_loss,
            5,
        ),
        "take_profit_1": round(
            take_profit_1,
            5,
        ),
        "take_profit_2": round(
            take_profit_2,
            5,
        ),
        "risk_distance": round(
            risk,
            5,
        ),
        "rr_tp1": round(
            rr1,
            2,
        ),
        "rr_tp2": round(
            rr2,
            2,
        ),
        "method": "ATR",
    }


# ============================================================
# BOUGIES
# ============================================================

def dataframe_to_candles(
    df: pd.DataFrame,
    limit: int = 25,
) -> List[Dict[str, Any]]:

    result: List[Dict[str, Any]] = []

    for _, row in df.tail(
        limit
    ).iterrows():

        result.append(
            {
                "datetime": row[
                    "datetime"
                ].isoformat(),

                "open": round(
                    float(row["open"]),
                    6,
                ),

                "high": round(
                    float(row["high"]),
                    6,
                ),

                "low": round(
                    float(row["low"]),
                    6,
                ),

                "close": round(
                    float(row["close"]),
                    6,
                ),

                "volume": round(
                    float(row["volume"]),
                    2,
                ),
            }
        )

    return result


# ============================================================
# ANALYSE PRINCIPALE
# ============================================================

def run_analysis(
    asset: str = ASSET,
    timeframe: str = TIMEFRAME,
    candles: int = DEFAULT_CANDLES,
) -> Dict[str, Any]:

    df = fetch_candles(
        asset=asset,
        timeframe=timeframe,
        outputsize=candles,
    )

    indicators = calculate_indicators(
        df
    )

    market = calculate_market_score(
        indicators
    )

    price = (
        safe_float(
            df.iloc[-1]["close"]
        )
        or 0.0
    )

    atr = indicators["atr14"]

    trade_plan = build_trade_plan(
        signal=market["signal"],
        price=price,
        atr=atr,
    )

    latest_time = df.iloc[-1][
        "datetime"
    ]

    return {
        "status": "success",
        "engine_version": APP_VERSION,
        "source": "Twelve Data",
        "asset": asset.upper(),
        "symbol": normalize_asset(asset),
        "timeframe": timeframe,
        "price": round(
            price,
            6,
        ),
        "signal": market["signal"],
        "confidence": market[
            "confidence"
        ],
        "score": market["score"],
        "trend": market["trend"],
        "setup_quality": market[
            "setup_quality"
        ],

        "indicators": {
            "ema9": round(
                indicators["ema9"],
                6,
            ),

            "ema21": round(
                indicators["ema21"],
                6,
            ),

            "rsi14": round(
                indicators["rsi14"],
                2,
            ),

            "momentum_5": round(
                indicators["momentum_5"],
                4,
            ),

            "atr14": round(
                indicators["atr14"],
                6,
            ),

            "macd": round(
                indicators["macd"],
                6,
            ),

            "macd_signal": round(
                indicators["macd_signal"],
                6,
            ),

            "macd_histogram": round(
                indicators[
                    "macd_histogram"
                ],
                6,
            ),
        },

        "scores": {
            "ema": (
                2
                if indicators["ema9"]
                > indicators["ema21"]
                else -2
                if indicators["ema9"]
                < indicators["ema21"]
                else 0
            ),

            "rsi": (
                1
                if 52
                <= indicators["rsi14"]
                <= 70

                else -1
                if 30
                <= indicators["rsi14"]
                <= 48

                else 0
            ),

            "momentum": (
                1
                if indicators[
                    "momentum_5"
                ] > 0

                else -1
                if indicators[
                    "momentum_5"
                ] < 0

                else 0
            ),

            "macd": (
                1
                if (
                    indicators["macd"]
                    > indicators[
                        "macd_signal"
                    ]
                    and indicators[
                        "macd_histogram"
                    ] > 0
                )

                else -1
                if (
                    indicators["macd"]
                    < indicators[
                        "macd_signal"
                    ]
                    and indicators[
                        "macd_histogram"
                    ] < 0
                )

                else 0
            ),
        },

        "reasons": market[
            "reasons"
        ],

        "trade_plan": trade_plan,

        "data": {
            "candles_count": len(df),
            "last_candle": latest_time.isoformat(),
        },
    }


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def root() -> Dict[str, Any]:

    return {
        "status": "online",
        "service": "Pocket AI Trader",
        "engine_version": APP_VERSION,
        "asset": ASSET,
        "timeframe": TIMEFRAME,
    }


@app.get("/health")
def health() -> Dict[str, Any]:

    return {
        "status": "healthy",
        "engine_version": APP_VERSION,
        "asset": ASSET,
        "timeframe": TIMEFRAME,
        "twelve_data_configured": bool(
            os.getenv(
                "TWELVE_DATA_API_KEY"
            )
        ),
    }


@app.post("/analyze")
def analyze(
    request: AnalyzeRequest,
) -> Dict[str, Any]:

    return run_analysis(
        asset=request.asset,
        timeframe=request.timeframe,
        candles=request.candles,
    )


@app.get("/analyze")
def analyze_get(
    asset: str = ASSET,
    timeframe: str = TIMEFRAME,
    candles: int = DEFAULT_CANDLES,
) -> Dict[str, Any]:

    candles = max(
        MIN_CANDLES,
        min(
            MAX_CANDLES,
            candles,
        ),
    )

    return run_analysis(
        asset=asset,
        timeframe=timeframe,
        candles=candles,
    )


@app.get("/quick-analysis")
def quick_analysis() -> Dict[str, Any]:

    result = run_analysis(
        asset=ASSET,
        timeframe=TIMEFRAME,
        candles=DEFAULT_CANDLES,
    )

    return {
        "status": result["status"],
        "engine_version": result[
            "engine_version"
        ],
        "asset": result["asset"],
        "timeframe": result[
            "timeframe"
        ],
        "price": result["price"],
        "signal": result["signal"],
        "confidence": result[
            "confidence"
        ],
        "score": result["score"],
        "trend": result["trend"],
        "setup_quality": result[
            "setup_quality"
        ],
        "trade_plan": result[
            "trade_plan"
        ],
    }


@app.get("/candles")
def candles(
    asset: str = ASSET,
    timeframe: str = TIMEFRAME,
    limit: int = 25,
) -> Dict[str, Any]:

    limit = max(
        1,
        min(
            MAX_CANDLES,
            limit,
        ),
    )

    df = fetch_candles(
        asset=asset,
        timeframe=timeframe,
        outputsize=max(
            DEFAULT_CANDLES,
            limit,
        ),
    )

    result = dataframe_to_candles(
        df,
        limit,
    )

    return {
        "status": "success",
        "source": "Twelve Data",
        "asset": asset.upper(),
        "timeframe": timeframe,
        "candles_count": len(result),
        "candles": result,
    }


# ============================================================
# TEST TRADE PLAN
# ============================================================

@app.get("/test-trade-plan")
def test_trade_plan(
    signal: str = "BUY",
    price: float = 1.13700,
    atr: float = 0.00100,
) -> Dict[str, Any]:

    signal = signal.strip().upper()

    if signal not in [
        "BUY",
        "SELL",
    ]:
        raise HTTPException(
            status_code=400,
            detail=(
                "signal doit être BUY ou SELL."
            ),
        )

    if price <= 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "price doit être supérieur à 0."
            ),
        )

    if atr <= 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "atr doit être supérieur à 0."
            ),
        )

    trade_plan = build_trade_plan(
        signal=signal,
        price=price,
        atr=atr,
    )

    return {
        "status": "success",
        "engine_version": APP_VERSION,
        "test": True,
        "signal": signal,
        "price": price,
        "atr": atr,
        "trade_plan": trade_plan,
    }


@app.get("/version")
def version() -> Dict[str, Any]:

    return {
        "engine_version": APP_VERSION,
        "asset": ASSET,
        "timeframe": TIMEFRAME,
        "source": "Twelve Data",
    }


# ============================================================
# LANCEMENT LOCAL
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
