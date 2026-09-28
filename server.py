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

# Format interne du moteur
ASSET = "EURUSD"
TIMEFRAME = "15m"

# Format Twelve Data
TWELVE_DATA_SYMBOL = "EUR/USD"
TWELVE_DATA_INTERVAL = "15min"

CANDLE_LIMIT = 100

# Paramètres moteur 4.1
LOW_VOLATILITY_THRESHOLD = 0.70
HIGH_VOLATILITY_THRESHOLD = 1.30

# Distance minimale d'un niveau opposé exprimée en ATR
LEVEL_BUFFER_ATR = 0.50

# Ratio risque/rendement minimal accepté
MIN_RR = 1.50


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

    if (
        volatility_ratio
        < LOW_VOLATILITY_THRESHOLD
    ):

        volatility_state = "LOW"

    elif (
        volatility_ratio
        > HIGH_VOLATILITY_THRESHOLD
    ):

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
# BLOC 1 — FILTRE SUPPORT / RESISTANCE
# ============================================================

def level_context(
    close: float,
    atr14: float,
    lv: Dict[str, float],
) -> Dict[str, Any]:

    support = lv["support"]
    resistance = lv["resistance"]

    distance_to_support = max(
        0.0,
        close - support,
    )

    distance_to_resistance = max(
        0.0,
        resistance - close,
    )

    if atr14 > 0:

        support_atr = (
            distance_to_support
            / atr14
        )

        resistance_atr = (
            distance_to_resistance
            / atr14
        )

    else:

        support_atr = 0.0
        resistance_atr = 0.0

    # BUY dangereux si résistance trop proche
    buy_blocked = (
        atr14 > 0
        and distance_to_resistance
        <= LEVEL_BUFFER_ATR * atr14
    )

    # SELL dangereux si support trop proche
    sell_blocked = (
        atr14 > 0
        and distance_to_support
        <= LEVEL_BUFFER_ATR * atr14
    )

    if buy_blocked:

        level_bias = "RESISTANCE_NEAR"

    elif sell_blocked:

        level_bias = "SUPPORT_NEAR"

    else:

        level_bias = "ROOM_AVAILABLE"

    return {
        "distance_to_support": distance_to_support,
        "distance_to_resistance": distance_to_resistance,
        "support_atr": support_atr,
        "resistance_atr": resistance_atr,
        "buy_blocked": buy_blocked,
        "sell_blocked": sell_blocked,
        "level_bias": level_bias,
    }


# ============================================================
# BLOC 2 — CONFIRMATION RENFORCEE
# ============================================================

def confirmation_block(
    df: pd.DataFrame,
    ind: Dict[str, float],
    market: Dict[str, Any],
) -> Dict[str, Any]:

    rsi14 = ind["rsi14"]
    momentum = ind["momentum_5"]

    confirmation_score = 0
    reasons: List[str] = []

    # RSI
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

    # Momentum
    if momentum > 0:

        confirmation_score += 1

        reasons.append(
            "Momentum positif"
        )

    elif momentum < 0:

        confirmation_score -= 1

        reasons.append(
            "Momentum negatif"
        )

    else:

        reasons.append(
            "Momentum neutre"
        )

    # Dernière bougie
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

    else:

        reasons.append(
            "Dernière bougie neutre"
        )

    return {
        "score": int(
            confirmation_score
        ),
        "reasons": reasons,
    }


# ============================================================
# BLOC 3 — QUALITE DU SETUP
# ============================================================

def calculate_setup_quality(
    signal: str,
    trend_score: int,
    confirmation_score: int,
    ind: Dict[str, float],
    market: Dict[str, Any],
    level_ctx: Dict[str, Any],
) -> Dict[str, Any]:

    # Base
    if signal == "WAIT":

        quality = 35

    else:

        quality = 50

    # --------------------------------------------------------
    # Force de tendance
    # --------------------------------------------------------

    if abs(trend_score) == 2:

        quality += 15

    elif abs(trend_score) == 1:

        quality += 7

    # --------------------------------------------------------
    # Confirmation
    # --------------------------------------------------------

    if abs(confirmation_score) == 3:

        quality += 20

    elif abs(confirmation_score) == 2:

        quality += 12

    elif abs(confirmation_score) == 1:

        quality += 5

    # --------------------------------------------------------
    # Cohérence signal / confirmation
    # --------------------------------------------------------

    if signal == "BUY":

        if confirmation_score >= 2:

            quality += 10

        elif confirmation_score <= -1:

            quality -= 15

    elif signal == "SELL":

        if confirmation_score <= -2:

            quality += 10

        elif confirmation_score >= 1:

            quality -= 15

    # --------------------------------------------------------
    # Volatilité
    # --------------------------------------------------------

    if (
        market["volatility_state"]
        == "NORMAL"
    ):

        quality += 5

    elif (
        market["volatility_state"]
        == "HIGH"
    ):

        quality += 2

    elif (
        market["volatility_state"]
        == "LOW"
    ):

        quality -= 8

    # --------------------------------------------------------
    # Espace devant le trade
    # --------------------------------------------------------

    if signal == "BUY":

        room_atr = (
            level_ctx["resistance_atr"]
        )

    elif signal == "SELL":

        room_atr = (
            level_ctx["support_atr"]
        )

    else:

        room_atr = 0.0

    if room_atr >= 2.0:

        quality += 8

    elif room_atr >= 1.0:

        quality += 4

    elif (
        room_atr < 0.5
        and signal != "WAIT"
    ):

        quality -= 15

    # --------------------------------------------------------
    # Blocage S/R
    # --------------------------------------------------------

    if (
        signal == "BUY"
        and level_ctx["buy_blocked"]
    ):

        quality -= 20

    if (
        signal == "SELL"
        and level_ctx["sell_blocked"]
    ):

        quality -= 20

    # Limites
    quality = int(
        max(
            0,
            min(
                100,
                quality,
            ),
        )
    )

    if quality >= 75:

        state = "STRONG"

    elif quality >= 55:

        state = "MODERATE"

    else:

        state = "WEAK"

    return {
        "score": quality,
        "state": state,
    }


# ============================================================
# BLOC 4 — PLAN DE TRADE DYNAMIQUE
# ============================================================

def calculate_trade_plan(
    df: pd.DataFrame,
    signal: str,
    atr14: float,
    lv: Dict[str, float],
) -> Dict[str, Optional[float]]:

    entry = float(
        df["close"].iloc[-1]
    )

    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward: Optional[float] = None
    stop_distance: Optional[float] = None
    target_distance: Optional[float] = None

    if atr14 <= 0:

        return {
            "entry": entry,
            "stop_loss": None,
            "take_profit": None,
            "risk_reward": None,
            "stop_distance": None,
            "target_distance": None,
        }

    recent = df.tail(20)

    # ========================================================
    # BUY
    # ========================================================

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

        if stop_distance <= 0:

            return {
                "entry": entry,
                "stop_loss": None,
                "take_profit": None,
                "risk_reward": None,
                "stop_distance": None,
                "target_distance": None,
            }

        # Objectif standard = 2R
        target_distance = (
            stop_distance * 2.0
        )

        raw_take_profit = (
            entry
            + target_distance
        )

        # Résistance
        resistance = lv["resistance"]

        room = max(
            0.0,
            resistance - entry,
        )

        # Si résistance avant le TP standard,
        # on adapte le TP à cette résistance.
        if (
            room > 0
            and room < target_distance
        ):

            target_distance = room

            take_profit = resistance

            risk_reward = (
                target_distance
                / stop_distance
            )

        else:

            take_profit = raw_take_profit

            risk_reward = 2.0

    # ========================================================
    # SELL
    # ========================================================

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

        if stop_distance <= 0:

            return {
                "entry": entry,
                "stop_loss": None,
                "take_profit": None,
                "risk_reward": None,
                "stop_distance": None,
                "target_distance": None,
            }

        # Objectif standard = 2R
        target_distance = (
            stop_distance * 2.0
        )

        raw_take_profit = (
            entry
            - target_distance
        )

        # Support
        support = lv["support"]

        room = max(
            0.0,
            entry - support,
        )

        # Si support avant le TP standard,
        # on adapte le TP à ce support.
        if (
            room > 0
            and room < target_distance
        ):

            target_distance = room

            take_profit = support

            risk_reward = (
                target_distance
                / stop_distance
            )

        else:

            take_profit = raw_take_profit

            risk_reward = 2.0

    # ========================================================
    # VALIDATION RR
    # ========================================================

    if (
        risk_reward is not None
        and risk_reward < MIN_RR
    ):

        stop_loss = None
        take_profit = None
        risk_reward = None
        stop_distance = None
        target_distance = None

    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_reward": risk_reward,
        "stop_distance": stop_distance,
        "target_distance": target_distance,
    }


# ============================================================
# MOTEUR DE DECISION — 4 BLOCS
# ============================================================

def decision_engine(
    df: pd.DataFrame,
    ind: Dict[str, float],
    market: Dict[str, Any],
    lv: Dict[str, float],
) -> Dict[str, Any]:

    """
    4 blocs :

    BLOC 1 : Tendance + Support/Resistance
    BLOC 2 : Confirmation multi-indicateurs
    BLOC 3 : Qualité du setup + signal
    BLOC 4 : Plan de trade dynamique
    """

    close = float(
        df["close"].iloc[-1]
    )

    ema9 = ind["ema9"]
    ema21 = ind["ema21"]

    # ========================================================
    # BLOC 1 — FILTRE DE TENDANCE
    # ========================================================

    trend_score = 0

    trend_reasons: List[str] = []

    # EMA9 / EMA21
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

    # Prix / EMA9
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

    else:

        trend_reasons.append(
            "Prix proche de EMA9"
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

    confirmation = confirmation_block(
        df,
        ind,
        market,
    )

    confirmation_score = (
        confirmation["score"]
    )

    confirmation_reasons = (
        confirmation["reasons"]
    )

    # IMPORTANT :
    # On conserve le calcul de score validé en 4.0.1 :
    #
    # tendance = 2 points maximum
    # confirmation = 3 points maximum
    #
    # Total = -5 à +5
    score = (
        trend_score
        + confirmation_score
    )

    # ========================================================
    # SUPPORT / RESISTANCE
    # ========================================================

    level_ctx = level_context(
        close,
        ind["atr14"],
        lv,
    )

    # BUY bloqué si résistance trop proche
    if (
        score >= 3
        and level_ctx["buy_blocked"]
    ):

        score = 2

        confirmation_reasons.append(
            "BUY bloque: resistance trop proche"
        )

    # SELL bloqué si support trop proche
    if (
        score <= -3
        and level_ctx["sell_blocked"]
    ):

        score = -2

        confirmation_reasons.append(
            "SELL bloque: support trop proche"
        )

    # ========================================================
    # FILTRE DE VOLATILITE
    # ========================================================

    if (
        market["volatility_state"]
        == "LOW"
        and abs(score) >= 3
    ):

        score = int(
            np.sign(score) * 2
        )

        confirmation_reasons.append(
            "Filtre de volatilite: "
            "conviction reduite"
        )

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
    # CONFIDENCE
    # ========================================================

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
    # BLOC 3 — QUALITE DU SETUP
    # ========================================================

    quality = calculate_setup_quality(
        signal=signal,
        trend_score=trend_score,
        confirmation_score=confirmation_score,
        ind=ind,
        market=market,
        level_ctx=level_ctx,
    )

    # ========================================================
    # BLOC 4 — PLAN DE TRADE
    # ========================================================

    plan = calculate_trade_plan(
        df=df,
        signal=signal,
        atr14=ind["atr14"],
        lv=lv,
    )

    # ========================================================
    # VALIDATION FINALE DU PLAN
    # ========================================================

    if (
        signal in ("BUY", "SELL")
        and plan["risk_reward"] is None
    ):

        confirmation_reasons.append(
            "Plan annule: espace ou "
            "ratio risque/rendement insuffisant"
        )

        signal = "WAIT"

        confidence = min(
            confidence,
            66,
        )

        plan = calculate_trade_plan(
            df=df,
            signal="WAIT",
            atr14=ind["atr14"],
            lv=lv,
        )

    # ========================================================
    # VOLATILITE DANS LES RAISONS
    # ========================================================

    if (
        market["volatility_state"]
        == "LOW"
    ):

        confirmation_reasons.append(
            "Volatilite faible"
        )

    elif (
        market["volatility_state"]
        == "HIGH"
    ):

        confirmation_reasons.append(
            "Volatilite elevee"
        )

    # ========================================================
    # RESULTAT DU MOTEUR
    # ========================================================

    return {
        "trend": trend,
        "score": int(score),
        "signal": signal,
        "confidence": confidence,

        "setup_quality": quality,

        "trend_score": int(
            trend_score
        ),

        "confirmation_score": int(
            confirmation_score
        ),

        "level_context": level_ctx,

        "reasons": (
            trend_reasons
            + confirmation_reasons
        ),

        "trade_plan": plan,
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

    # --------------------------------------------------------
    # Calculs
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Réponse
    # --------------------------------------------------------

    return {
        "status": "success",

        "engine_version": APP_VERSION,

        "asset": asset,

        "timeframe": timeframe,

        "price": float(
            df["close"].iloc[-1]
        ),

        "signal": decision[
            "signal"
        ],

        "confidence": decision[
            "confidence"
        ],

        "score": decision[
            "score"
        ],

        "trend": decision[
            "trend"
        ],

        # ====================================================
        # NOUVEAU 4.1
        # ====================================================

        "setup_quality": decision[
            "setup_quality"
        ],

        "scores": {
            "trend": decision[
                "trend_score"
            ],

            "confirmation": decision[
                "confirmation_score"
            ],

            "total": decision[
                "score"
            ],
        },

        # ====================================================
        # INDICATEURS
        # ====================================================

        "indicators": {
            k: round(
                v,
                8,
            )
            for k, v in ind.items()
        },

        # ====================================================
        # MARCHE
        # ====================================================

        "market": {
            "candle_direction": market[
                "candle_direction"
            ],

            "average_range": round(
                market[
                    "average_range"
                ],
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

        # ====================================================
        # SUPPORT / RESISTANCE
        # ====================================================

        "levels": {
            k: round(
                v,
                8,
            )
            for k, v in lv.items()
        },

        "level_context": {
            "distance_to_support": round(
                decision[
                    "level_context"
                ][
                    "distance_to_support"
                ],
                8,
            ),

            "distance_to_resistance": round(
                decision[
                    "level_context"
                ][
                    "distance_to_resistance"
                ],
                8,
            ),

            "support_atr": round(
                decision[
                    "level_context"
                ][
                    "support_atr"
                ],
                3,
            ),

            "resistance_atr": round(
                decision[
                    "level_context"
                ][
                    "resistance_atr"
                ],
                3,
            ),

            "buy_blocked": decision[
                "level_context"
            ][
                "buy_blocked"
            ],

            "sell_blocked": decision[
                "level_context"
            ][
                "sell_blocked"
            ],

            "level_bias": decision[
                "level_context"
            ][
                "level_bias"
            ],
        },

        # ====================================================
        # PLAN DE TRADE
        # ====================================================

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

        # ====================================================
        # RAISONS
        # ====================================================

        "reasons": decision[
            "reasons"
        ],

        # ====================================================
        # BOUGIES
        # ====================================================

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
        "engine": "4-bloc decision engine v4.1",
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

    
# ============================================================
# TEST TRADE PLAN — BUY / SELL
# ============================================================

def test_trade_plan(
    forced_signal: str,
) -> Dict[str, Any]:

    candles = fetch_candles(
        ASSET,
        TIMEFRAME,
        CANDLE_LIMIT,
    )

    df = to_dataframe(candles)

    if len(df) < 30:
        raise HTTPException(
            status_code=422,
            detail="Nombre de bougies insuffisant pour le test."
        )

    # Calcul des indicateurs réels
    ind = calculate_indicators(df)

    # Niveaux réels
    lv = levels(df)

    # Plan forcé uniquement pour tester le moteur SL/TP
    trade_plan = calculate_trade_plan(
        df=df,
        signal=forced_signal,
        atr14=ind["atr14"],
        lv=lv,
    )

    return {
        "status": "success",
        "engine_version": APP_VERSION,
        "test_mode": True,
        "test_type": forced_signal,
        "asset": ASSET,
        "timeframe": TIMEFRAME,

        "price": float(
            df["close"].iloc[-1]
        ),

        "signal": forced_signal,

        "indicators": {
            "ema9": round(
                ind["ema9"],
                8,
            ),
            "ema21": round(
                ind["ema21"],
                8,
            ),
            "rsi14": round(
                ind["rsi14"],
                8,
            ),
            "momentum_5": round(
                ind["momentum_5"],
                8,
            ),
            "atr14": round(
                ind["atr14"],
                8,
            ),
        },

        "levels": {
            "support": round(
                lv["support"],
                8,
            ),
            "resistance": round(
                lv["resistance"],
                8,
            ),
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
            for k, v in trade_plan.items()
        },

        "validation": {
            "entry_valid": (
                trade_plan["entry"] is not None
            ),

            "stop_loss_valid": (
                trade_plan["stop_loss"] is not None
            ),

            "take_profit_valid": (
                trade_plan["take_profit"] is not None
            ),

            "risk_reward_valid": (
                trade_plan["risk_reward"] is not None
                and trade_plan["risk_reward"] >= MIN_RR
            ),

            "trade_plan_valid": (
                trade_plan["entry"] is not None
                and trade_plan["stop_loss"] is not None
                and trade_plan["take_profit"] is not None
                and trade_plan["risk_reward"] is not None
                and trade_plan["risk_reward"] >= MIN_RR
            ),
        },
    }


# ============================================================
# TEST BUY
# ============================================================

@app.get("/test/buy")
def test_buy() -> Dict[str, Any]:

    return test_trade_plan("BUY")


# ============================================================
# TEST SELL
# ============================================================

@app.get("/test/sell")
def test_sell() -> Dict[str, Any]:

    return test_trade_plan("SELL")

    return analyze_candles(
        [
            c.model_dump()
            for c in request.candles
        ],
        request.asset,
        request.timeframe,
    )
