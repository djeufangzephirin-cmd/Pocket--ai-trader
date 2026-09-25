from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="Pocket AI Trader",
    description="API d'analyse de marché avec OpenAI",
    version="1.0.0"
)


class MarketData(BaseModel):
    asset: str
    price: float
    previous_price: Optional[float] = None
    timeframe: str = "1m"


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "Pocket AI Trader",
        "message": "Serveur opérationnel"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/analyze")
def analyze_market(data: MarketData):

    if data.previous_price is None:
        signal = "WAIT"
        change = 0
    else:
        change = data.price - data.previous_price

        if change > 0:
            signal = "CALL"
        elif change < 0:
            signal = "PUT"
        else:
            signal = "WAIT"

    return {
        "asset": data.asset,
        "price": data.price,
        "previous_price": data.previous_price,
        "change": round(change, 6),
        "signal": signal,
        "timeframe": data.timeframe
    }
