import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# POCKET AI TRADER
# ENGINE v4.1.1
# ============================================================

APP_VERSION = "4.1.1"

ASSET = "EURUSD"
TIMEFRAME = "15m"

DEFAULT_CANDLES = 100
MIN_CANDLES = 30
MAX_CANDLES = 500

TWELVE_DATA_URL = "https://api.twelvedata.com/time_series"


# ============================================================
# SCORING CONFIGURATION
# ============================================================

EMA_SEPARATION_THRESHOLD = 0.0005


# ============================================================
# RISK MANAGEMENT CONFIGURATION
# ============================================================

DEFAULT_ACCOUNT_BALANCE = 1000.0
DEFAULT_RISK_PERCENT = 1.0

PIP_SIZE_EURUSD = 0.0001
PIP_VALUE_PER_STANDARD_LOT = 10.0

MIN_RISK_PERCENT = 0.1
MAX_RISK_PERCENT = 2.0

MIN_RR_TP1 = 1.0
MIN_RR_TP2 = 1.5

ATR_STOP_MULTIPLIER = 1.5
TP1_R_MULTIPLIER = 1.5
TP2_R_MULTIPLIER = 2.5


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Pocket AI Trader",
    description=(
        "Trading analysis engine for EURUSD. "
        "Technical analysis, trade plan and risk management."
    ),
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
# PYDANTIC MODELS
# ============================================================

class AnalyzeRequest(BaseModel):
    asset: str = Field(
        default=ASSET,
        min_length=1,
        max_length=20,
    )

    timeframe: str = Field(
        default=TIMEFRAME,
        min_length=1,
        max_length=10,
    )

    candles: int = Field(
        default=DEFAULT_CANDLES,
        ge=MIN_CANDLES,
        le=MAX_CANDLES,
    )

    account_balance: float = Field(
        default=DEFAULT_ACCOUNT_BALANCE,
        gt=0,
    )

    risk_percent: float = Field(
        default=DEFAULT_RISK_PERCENT,
        ge=MIN_RISK_PERCENT,
        le=MAX_RISK_PERCENT,
    )


class TradePlanRequest(BaseModel):
    signal: str = Field(
        min_length=1,
        max_length=10,
    )

    price: float = Field(
        gt=0,
    )

    atr: float = Field(
        gt=0,
    )


class RiskRequest(BaseModel):
    account_balance: float = Field(
        gt=0,
    )

    risk_percent: float = Field(
        ge=MIN_RISK_PERCENT,
        le=MAX_RISK_PERCENT,
    )

    stop_loss_pips: float = Field(
        gt=0,
    )


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize_asset(asset: str) -> str:
    """
    Normalise EURUSD -> EUR/USD.
    Conserve les symboles déjà formatés.
    """

    asset = asset.strip().upper()

    if "/" in asset:
        return asset

    if len(asset) == 6:
        return f"{asset[:3]}/{asset[3:]}"

    return asset


def normalize_interval(timeframe: str) -> str:
    """
    Convertit les timeframes internes vers le format Twelve Data.
    """

    timeframe = timeframe.strip().lower()

    mapping = {
        "1m": "1min",
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "45m": "45min",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "8h": "8h",
        "1d": "1day",
        "1w": "1week",
        "1mo": "1month",
    }

    return mapping.get(timeframe, timeframe)


def safe_float(value: Any) -> Optional[float]:
    """
    Convertit une valeur en float fini.
    Retourne None pour NaN, inf ou valeur invalide.
    """

    try:
        if value is None:
            return None

        result = float(value)

        if not np.isfinite(result):
            return None

        return result

    except (TypeError, ValueError):
        return None


def validate_positive_number(
    value: float,
    field_name: str,
) -> float:

    number = safe_float(value)

    if number is None or number <= 0:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} doit être supérieur à 0.",
        )

    return number


def validate_risk_percent(
    risk_percent: float,
) -> float:

    value = safe_float(risk_percent)

    if value is None:
        raise HTTPException(
            status_code=422,
            detail="Le risque doit être numérique.",
        )

    if not MIN_RISK_PERCENT <= value <= MAX_RISK_PERCENT:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Le risque doit être compris entre "
                f"{MIN_RISK_PERCENT}% et "
                f"{MAX_RISK_PERCENT}%."
            ),
        )

    return value


# ============================================================
# TWELVE DATA
# ============================================================

def get_api_key() -> str:
    """
    Récupère la clé Twelve Data depuis la variable d'environnement.

    IMPORTANT :
    Ne jamais écrire la clé directement dans le code.
    """

    api_key = os.getenv("TWELVE_DATA_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail=(
                "TWELVE_DATA_API_KEY n'est pas configurée. "
                "Ajoutez-la dans les variables d'environnement "
                "de Render ou dans le fichier .env local."
            ),
        )

    api_key = api_key.strip()

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="TWELVE_DATA_API_KEY est vide.",
        )

    return api_key


def fetch_candles(
    asset: str = ASSET,
    timeframe: str = TIMEFRAME,
    outputsize: int = DEFAULT_CANDLES,
) -> pd.DataFrame:

    api_key = get_api_key()

    try:
        outputsize = int(outputsize)
    except (TypeError, ValueError):
        outputsize = DEFAULT_CANDLES

    outputsize = max(
        MIN_CANDLES,
        min(MAX_CANDLES, outputsize),
    )

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

    except requests.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Twelve Data n'a pas répondu dans le délai prévu.",
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
                f"Twelve Data HTTP {response.status_code}: "
                f"{response.text[:500]}"
            ),
        )

    try:
        payload = response.json()

    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="Réponse JSON invalide reçue de Twelve Data.",
        )

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=502,
            detail="Format de réponse Twelve Data invalide.",
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

    if not isinstance(values, list) or not values:
        raise HTTPException(
            status_code=502,
            detail="Twelve Data n'a retourné aucune bougie.",
        )

    rows: List[Dict[str, Any]] = []

    for item in values:

        if not isinstance(item, dict):
            continue

        try:
            datetime_value = item.get("datetime")

            open_price = float(item["open"])
            high_price = float(item["high"])
            low_price = float(item["low"])
            close_price = float(item["close"])

            volume = float(
                item.get("volume", 0) or 0
            )

            prices = [
                open_price,
                high_price,
                low_price,
                close_price,
                volume,
            ]

            if not all(np.isfinite(x) for x in prices):
                continue

            if high_price < low_price:
                continue

            rows.append(
                {
                    "datetime": datetime_value,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                }
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

    if len(rows) < MIN_CANDLES:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Données insuffisantes : "
                f"{len(rows)} bougies valides reçues. "
                f"Minimum requis : {MIN_CANDLES}."
            ),
        )

    df = pd.DataFrame(rows)

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce",
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
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

    df = df.sort_values(
        "datetime",
        ascending=True,
    )

    df = df.drop_duplicates(
        subset=["datetime"],
        keep="last",
    )

    df = df.reset_index(drop=True)

    if len(df) < MIN_CANDLES:
        raise HTTPException(
            status_code=502,
            detail=(
                "Nombre de bougies valides insuffisant "
                "après nettoyage des données."
            ),
        )

    return df.tail(outputsize).reset_index(drop=True)


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
        min_periods=period,
    ).mean()


def calculate_rsi(
    series: pd.Series,
    period: int = 14,
) -> pd.Series:

    delta = series.diff()

    gain = delta.clip(
        lower=0,
    )

    loss = -delta.clip(
        upper=0,
    )

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

    rsi = pd.Series(
        np.nan,
        index=series.index,
        dtype=float,
    )

    normal = avg_loss > 0

    rs = pd.Series(
        np.nan,
        index=series.index,
        dtype=float,
    )

    rs.loc[normal] = (
        avg_gain.loc[normal]
        / avg_loss.loc[normal]
    )

    rsi.loc[normal] = (
        100
        - (
            100
            / (1 + rs.loc[normal])
        )
    )

    # Cas sans pertes : RSI = 100
    no_loss = (
        (avg_loss == 0)
        & (avg_gain > 0)
    )

    rsi.loc[no_loss] = 100.0

    # Cas sans gains ni pertes : RSI neutre.
    flat = (
        (avg_loss == 0)
        & (avg_gain == 0)
    )

    rsi.loc[flat] = 50.0

    return rsi.fillna(50.0)


def calculate_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:

    previous_close = df["close"].shift(1)

    tr1 = (
        df["high"]
        - df["low"]
    )

    tr2 = (
        df["high"]
        - previous_close
    ).abs()

    tr3 = (
        df["low"]
        - previous_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
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

    histogram = (
        macd - signal
    )

    return {
        "macd": macd,
        "signal": signal,
        "histogram": histogram,
    }


def calculate_indicators(
    df: pd.DataFrame,
) -> Dict[str, float]:

    if len(df) < MIN_CANDLES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Au moins {MIN_CANDLES} bougies "
                "sont nécessaires pour calculer les indicateurs."
            ),
        )

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

    # Momentum en pourcentage sur 5 bougies.
    #
    # Exemple :
    # 0.0141 = +0.0141 %
    #
    df["momentum5"] = (
        close.pct_change(5)
        * 100
    )

    df["atr14"] = calculate_atr(
        df,
        14,
    )

    macd = calculate_macd(
        close,
    )

    df["macd"] = macd["macd"]
    df["macd_signal"] = macd["signal"]
    df["macd_histogram"] = macd["histogram"]

    latest = df.iloc[-1]

    ema9 = safe_float(
        latest["ema9"]
    )

    ema21 = safe_float(
        latest["ema21"]
    )

    rsi14 = safe_float(
        latest["rsi14"]
    )

    momentum = safe_float(
        latest["momentum5"]
    )

    atr14 = safe_float(
        latest["atr14"]
    )

    macd_value = safe_float(
        latest["macd"]
    )

    macd_signal = safe_float(
        latest["macd_signal"]
    )

    macd_histogram = safe_float(
        latest["macd_histogram"]
    )

    if ema9 is None or ema21 is None:
        raise HTTPException(
            status_code=502,
            detail="EMA indisponibles.",
        )

    if atr14 is None or atr14 <= 0:
        raise HTTPException(
            status_code=502,
            detail="ATR14 invalide ou indisponible.",
        )

    return {
        "ema9": ema9,
        "ema21": ema21,
        "rsi14": rsi14 if rsi14 is not None else 50.0,
        "momentum_5": (
            momentum
            if momentum is not None
            else 0.0
        ),
        "atr14": atr14,
        "macd": (
            macd_value
            if macd_value is not None
            else 0.0
        ),
        "macd_signal": (
            macd_signal
            if macd_signal is not None
            else 0.0
        ),
        "macd_histogram": (
            macd_histogram
            if macd_histogram is not None
            else 0.0
        ),
    }


# ============================================================
# MARKET SCORING
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
    macd_histogram = indicators[
        "macd_histogram"
    ]

    score = 0

    reasons: List[str] = []

    component_scores = {
        "ema": 0,
        "rsi": 0,
        "momentum": 0,
        "macd": 0,
    }


    # --------------------------------------------------------
    # EMA 9 / EMA 21
    # --------------------------------------------------------

    ema_distance = abs(
        ema9 - ema21
    )

    if (
        ema9 > ema21
        and ema_distance
        >= EMA_SEPARATION_THRESHOLD
    ):
        component_scores["ema"] = 2
        score += 2

        reasons.append(
            "EMA9 > EMA21 avec séparation suffisante"
        )

    elif (
        ema9 < ema21
        and ema_distance
        >= EMA_SEPARATION_THRESHOLD
    ):
        component_scores["ema"] = -2
        score -= 2

        reasons.append(
            "EMA9 < EMA21 avec séparation suffisante"
        )

    elif ema9 > ema21:
        component_scores["ema"] = 1
        score += 1

        reasons.append(
            "EMA9 > EMA21 mais séparation faible"
        )

    elif ema9 < ema21:
        component_scores["ema"] = -1
        score -= 1

        reasons.append(
            "EMA9 < EMA21 mais séparation faible"
        )

    else:
        reasons.append(
            "EMA9 = EMA21"
        )


    # --------------------------------------------------------
    # RSI 14
    # --------------------------------------------------------

    if 55 <= rsi < 70:

        component_scores["rsi"] = 1
        score += 1

        reasons.append(
            "RSI favorable aux acheteurs"
        )

    elif 30 < rsi <= 45:

        component_scores["rsi"] = -1
        score -= 1

        reasons.append(
            "RSI favorable aux vendeurs"
        )

    elif rsi >= 70:

        reasons.append(
            "RSI en surachat - prudence"
        )

    elif rsi <= 30:

        reasons.append(
            "RSI en survente - prudence"
        )

    else:

        reasons.append(
            "RSI neutre"
        )


    # --------------------------------------------------------
    # MOMENTUM 5
    # --------------------------------------------------------

    if abs(momentum) < 0.03:

        component_scores["momentum"] = 0

        reasons.append(
            "Momentum faible"
        )

    elif 0.03 <= momentum < 0.15:

        component_scores["momentum"] = 1
        score += 1

        reasons.append(
            "Momentum positif modéré"
        )

    elif -0.15 < momentum <= -0.03:

        component_scores["momentum"] = -1
        score -= 1

        reasons.append(
            "Momentum négatif modéré"
        )

    elif momentum >= 0.15:

        component_scores["momentum"] = 2
        score += 2

        reasons.append(
            "Momentum fortement positif"
        )

    elif momentum <= -0.15:

        component_scores["momentum"] = -2
        score -= 2

        reasons.append(
            "Momentum fortement négatif"
        )


    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    if (
        macd > macd_signal
        and macd_histogram > 0
    ):

        component_scores["macd"] = 1
        score += 1

        reasons.append(
            "MACD haussier"
        )

    elif (
        macd < macd_signal
        and macd_histogram < 0
    ):

        component_scores["macd"] = -1
        score -= 1

        reasons.append(
            "MACD baissier"
        )

    else:

        reasons.append(
            "MACD neutre"
        )


    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    if score >= 3:

        trend = "BULLISH"

    elif score <= -3:

        trend = "BEARISH"

    else:

        trend = "NEUTRAL"


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

    confidence = int(
        max(
            50,
            min(
                95,
                50 + abs(score) * 8,
            ),
        )
    )


    # --------------------------------------------------------
    # SETUP QUALITY
    # --------------------------------------------------------

    absolute_score = abs(score)

    if absolute_score >= 5:

        setup_state = "STRONG"
        setup_score = 90

    elif absolute_score >= 4:

        setup_state = "GOOD"
        setup_score = 75

    elif absolute_score >= 2:

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
        "setup_quality": {
            "score": setup_score,
            "state": setup_state,
        },
        "component_scores": component_scores,
        "reasons": reasons,
    }


# ============================================================
# TRADE PLAN
# ============================================================
def build_trade_plan(
    signal: str,
    price: float,
    atr: float,
) -> Optional[Dict[str, Any]]:

    signal = (
        signal
        .strip()
        .upper()
    )

    if signal not in {
        "BUY",
        "SELL",
    }:
        return None

    price = safe_float(price)
    atr = safe_float(atr)

    if price is None or price <= 0:
        return None

    if atr is None or atr <= 0:
        return None


    # --------------------------------------------------------
    # STOP LOSS
    # --------------------------------------------------------

    stop_distance = (
        atr
        * ATR_STOP_MULTIPLIER
    )


    # --------------------------------------------------------
    # TAKE PROFITS
    # --------------------------------------------------------

    tp1_distance = (
        stop_distance
        * TP1_R_MULTIPLIER
    )

    tp2_distance = (
        stop_distance
        * TP2_R_MULTIPLIER
    )


    if signal == "BUY":

        stop_loss = (
            price
            - stop_distance
        )

        take_profit_1 = (
            price
            + tp1_distance
        )

        take_profit_2 = (
            price
            + tp2_distance
        )

    else:

        stop_loss = (
            price
            + stop_distance
        )

        take_profit_1 = (
            price
            - tp1_distance
        )

        take_profit_2 = (
            price
            - tp2_distance
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

    if risk <= 0:
        return None

    rr1 = reward_1 / risk
    rr2 = reward_2 / risk


    if rr1 < MIN_RR_TP1:
        return None

    if rr2 < MIN_RR_TP2:
        return None


    return {
        "direction": signal,
        "entry": round(price, 5),
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
        "atr_multiplier": ATR_STOP_MULTIPLIER,
    }


# ============================================================
# RISK MANAGEMENT
# ============================================================
    
    def calculate_position_size(
    account_balance: float,
    risk_percent: float,
    stop_loss_pips: float,
    pip_value_per_lot: float = (
        PIP_VALUE_PER_STANDARD_LOT
    ),
) -> Dict[str, float]:

    account_balance = validate_positive_number(
        account_balance,
        "Le solde du compte",
    )

    stop_loss_pips = validate_positive_number(
        stop_loss_pips,
        "La distance du stop-loss",
    )

    pip_value_per_lot = validate_positive_number(
        pip_value_per_lot,
        "La valeur du pip",
    )

    risk_percent = validate_risk_percent(
        risk_percent,
    )

    risk_amount = (
        account_balance
        * risk_percent
        / 100.0
    )

    position_size_lots = (
        risk_amount
        / (
            stop_loss_pips
            * pip_value_per_lot
        )
    )

    max_loss = (
        position_size_lots
        * stop_loss_pips
        * pip_value_per_lot
    )

    return {
        "account_balance": round(
            account_balance,
            2,
        ),
        "risk_percent": round(
            risk_percent,
            2,
        ),
        "risk_amount": round(
            risk_amount,
            2,
        ),
        "stop_loss_pips": round(
            stop_loss_pips,
            2,
        ),
        "pip_value_per_lot": round(
            pip_value_per_lot,
            2,
        ),
        "position_size_lots": round(
            position_size_lots,
            4,
        ),
        "max_loss": round(
            max_loss,
            2,
        ),
    }


def calculate_risk_from_trade_plan(
    trade_plan: Optional[
        Dict[str, Any]
    ],
    account_balance: float,
    risk_percent: float,
) -> Optional[Dict[str, float]]:

    if not trade_plan:
        return None

    entry = safe_float(
        trade_plan.get("entry")
    )

    stop_loss = safe_float(
        trade_plan.get("stop_loss")
    )

    if entry is None or stop_loss is None:
        return None

    stop_distance = abs(
        entry - stop_loss
    )

    if stop_distance <= 0:
        return None

    stop_loss_pips = (
        stop_distance
        / PIP_SIZE_EURUSD
    )

    return calculate_position_size(
        account_balance=account_balance,
        risk_percent=risk_percent,
        stop_loss_pips=stop_loss_pips,
        pip_value_per_lot=(
            PIP_VALUE_PER_STANDARD_LOT
        ),
    )


# ============================================================
# CANDLE SERIALIZATION
# ============================================================

def dataframe_to_candles(
    df: pd.DataFrame,
    limit: int = 25,
) -> List[Dict[str, Any]]:

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 25

    limit = max(
        1,
        min(
            MAX_CANDLES,
            limit,
        ),
    )

    result: List[
        Dict[str, Any]
    ] = []

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
# MAIN ANALYSIS
# ============================================================

def run_analysis(
    asset: str = ASSET,
    timeframe: str = TIMEFRAME,
    candles: int = DEFAULT_CANDLES,
    account_balance: float = (
        DEFAULT_ACCOUNT_BALANCE
    ),
    risk_percent: float = (
        DEFAULT_RISK_PERCENT
    ),
) -> Dict[str, Any]:

    try:
        candles = int(candles)

    except (TypeError, ValueError):
        raise HTTPException(
            status_code=422,
            detail="Le nombre de bougies doit être un entier.",
        )

    if not (
        MIN_CANDLES
        <= candles
        <= MAX_CANDLES
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Le nombre de bougies doit être "
                f"compris entre {MIN_CANDLES} "
                f"et {MAX_CANDLES}."
            ),
        )

    account_balance = validate_positive_number(
        account_balance,
        "Le solde du compte",
    )

    risk_percent = validate_risk_percent(
        risk_percent,
    )

    df = fetch_candles(
        asset=asset,
        timeframe=timeframe,
        outputsize=candles,
    )

    indicators = calculate_indicators(
        df,
    )

    market = calculate_market_score(
        indicators,
    )

    price = safe_float(
        df.iloc[-1]["close"]
    )

    if price is None or price <= 0:
        raise HTTPException(
            status_code=502,
            detail="Prix actuel invalide.",
        )

    atr = indicators["atr14"]

    trade_plan = build_trade_plan(
        signal=market["signal"],
        price=price,
        atr=atr,
    )

    risk_management = (
        calculate_risk_from_trade_plan(
            trade_plan=trade_plan,
            account_balance=account_balance,
            risk_percent=risk_percent,
        )
    )

    latest_time = df.iloc[-1][
        "datetime"
    ]


    return {
        "status": "success",

        "engine_version": APP_VERSION,

        "source": "Twelve Data",

        "asset": asset.upper(),

        "symbol": normalize_asset(
            asset
        ),

        "timeframe": timeframe,

        "price": round(
            price,
            6,
        ),

        "signal": market[
            "signal"
        ],

        "confidence": market[
            "confidence"
        ],

        "score": market[
            "score"
        ],

        "trend": market[
            "trend"
        ],

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
                indicators[
                    "momentum_5"
                ],
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
                indicators[
                    "macd_signal"
                ],
                6,
            ),

            "macd_histogram": round(
                indicators[
                    "macd_histogram"
                ],
                6,
            ),
        },

        "scores": market[
            "component_scores"
        ],

        "reasons": market[
            "reasons"
        ],

        "trade_plan": trade_plan,

        "risk_management": (
            risk_management
        ),

        "data": {
            "candles_count": len(
                df
            ),

            "last_candle": (
                latest_time.isoformat()
            ),
        },
    }


# ============================================================
# ENDPOINTS - CORE
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


@app.get("/version")
def version() -> Dict[str, Any]:

    return {
        "engine_version": APP_VERSION,
        "asset": ASSET,
        "timeframe": TIMEFRAME,
        "source": "Twelve Data",
    }


# ============================================================
# ANALYZE - POST
# ============================================================

@app.post("/analyze")
def analyze(
    request: AnalyzeRequest,
) -> Dict[str, Any]:

    return run_analysis(
        asset=request.asset,
        timeframe=request.timeframe,
        candles=request.candles,
        account_balance=(
            request.account_balance
        ),
        risk_percent=(
            request.risk_percent
        ),
    )


# ============================================================
# ANALYZE - GET
# ============================================================

@app.get("/analyze")
def analyze_get(
    asset: str = ASSET,
    timeframe: str = TIMEFRAME,
    candles: int = DEFAULT_CANDLES,
    account_balance: float = (
        DEFAULT_ACCOUNT_BALANCE
    ),
    risk_percent: float = (
        DEFAULT_RISK_PERCENT
    ),
) -> Dict[str, Any]:

    return run_analysis(
        asset=asset,
        timeframe=timeframe,
        candles=candles,
        account_balance=account_balance,
        risk_percent=risk_percent,
    )


# ============================================================
# QUICK ANALYSIS
# ============================================================

@app.get("/quick-analysis")
def quick_analysis() -> Dict[str, Any]:

    result = run_analysis(
        asset=ASSET,
        timeframe=TIMEFRAME,
        candles=DEFAULT_CANDLES,
        account_balance=(
            DEFAULT_ACCOUNT_BALANCE
        ),
        risk_percent=(
            DEFAULT_RISK_PERCENT
        ),
    )

    return {
        "status": result[
            "status"
        ],

        "engine_version": result[
            "engine_version"
        ],

        "asset": result[
            "asset"
        ],

        "timeframe": result[
            "timeframe"
        ],

        "price": result[
            "price"
        ],

        "signal": result[
            "signal"
        ],

        "confidence": result[
            "confidence"
        ],

        "score": result[
            "score"
        ],

        "trend": result[
            "trend"
        ],

        "setup_quality": result[
            "setup_quality"
        ],

        "trade_plan": result[
            "trade_plan"
        ],

        "risk_management": result[
            "risk_management"
        ],
    }


# ============================================================
# CANDLES
# ============================================================

@app.get("/candles")
def candles(
    asset: str = ASSET,
    timeframe: str = TIMEFRAME,
    limit: int = 25,
) -> Dict[str, Any]:

    try:
        limit = int(limit)

    except (TypeError, ValueError):
        raise HTTPException(
            status_code=422,
            detail="limit doit être un entier.",
        )

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
        "symbol": normalize_asset(
            asset
        ),
        "timeframe": timeframe,
        "candles_count": len(
            result
        ),
        "candles": result,
    }


# ============================================================
# TRADE PLAN TEST
# ============================================================

@app.post("/test-trade-plan")
def test_trade_plan(
    request: TradePlanRequest,
) -> Dict[str, Any]:

    trade_plan = build_trade_plan(
        signal=request.signal,
        price=request.price,
        atr=request.atr,
    )

    if not trade_plan:
        raise HTTPException(
            status_code=422,
            detail=(
                "Trade Plan invalide "
                "pour les paramètres fournis."
            ),
        )

    return {
        "status": "success",
        "engine_version": APP_VERSION,
        "test": True,
        "trade_plan": trade_plan,
    }


# ============================================================
# RISK TEST
# ============================================================

@app.post("/test-risk")
def test_risk(
    request: RiskRequest,
) -> Dict[str, Any]:

    risk = calculate_position_size(
        account_balance=(
            request.account_balance
        ),
        risk_percent=(
            request.risk_percent
        ),
        stop_loss_pips=(
            request.stop_loss_pips
        ),
    )

    return {
        "status": "success",
        "engine_version": APP_VERSION,
        "test": True,
        "risk_management": risk,
    }


# ============================================================
# RISK CONFIGURATION
# ============================================================

@app.get("/risk-config")
def risk_config() -> Dict[str, Any]:

    return {
        "status": "success",
        "engine_version": APP_VERSION,

        "default_account_balance": (
            DEFAULT_ACCOUNT_BALANCE
        ),

        "default_risk_percent": (
            DEFAULT_RISK_PERCENT
        ),

        "min_risk_percent": (
            MIN_RISK_PERCENT
        ),

        "max_risk_percent": (
            MAX_RISK_PERCENT
        ),

        "pip_size_eurusd": (
            PIP_SIZE_EURUSD
        ),

        "pip_value_per_standard_lot": (
            PIP_VALUE_PER_STANDARD_LOT
        ),

        "min_rr_tp1": (
            MIN_RR_TP1
        ),

        "min_rr_tp2": (
            MIN_RR_TP2
        ),

        "atr_stop_multiplier": (
            ATR_STOP_MULTIPLIER
        ),

        "tp1_r_multiplier": (
            TP1_R_MULTIPLIER
        ),

        "tp2_r_multiplier": (
            TP2_R_MULTIPLIER
        ),
    }


# ============================================================
# APPLICATION INFORMATION
# ============================================================

@app.get("/config")
def config() -> Dict[str, Any]:

    return {
        "status": "success",
        "engine_version": APP_VERSION,

        "asset": ASSET,
        "timeframe": TIMEFRAME,

        "default_candles": (
            DEFAULT_CANDLES
        ),

        "min_candles": (
            MIN_CANDLES
        ),

        "max_candles": (
            MAX_CANDLES
        ),

        "source": "Twelve Data",

        "indicators": [
            "EMA9",
            "EMA21",
            "RSI14",
            "Momentum5",
            "ATR14",
            "MACD",
        ],

        "execution": {
            "managed_by": "order_manager.py",
            "connector": "po_connector.py",
            "server_executes_orders": False,
        },
    }


# ============================================================
# LOCAL / RENDER START
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
