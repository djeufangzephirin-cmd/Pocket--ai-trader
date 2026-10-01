"""
Module de configuration centrale pour pocket-ai-trader.
Valide rigoureusement les variables d'environnement et applique des verrous de sécurité.
"""

from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Métadonnées Application
    APP_NAME: str = "pocket-ai-trader"
    APP_VERSION: str = "4.1.0"
    ENVIRONMENT: str = "development"

    # Verrous de Sécurité Trading
    TRADING_MODE: Literal["demo", "live"] = "demo"
    TRADING_ENABLED: bool = False

    # Marché par défaut
    ASSET: str = "EURUSD"
    TIMEFRAME: str = "15m"

    # Clé API Twelve Data
    TWELVE_DATA_API_KEY: str = Field(default="", description="Clé API Twelve Data")

    # Pocket Option Credentials
    PO_SSID: str = Field(default="", description="Session ID Pocket Option")
    PO_USER_ID: str = Field(default="", description="Identifiant Pocket Option")

    # Gestion du Capital et Risque (Money Management)
    DEFAULT_ACCOUNT_BALANCE: float = 1000.0
    DEFAULT_RISK_PERCENT: float = 1.0
    MIN_RISK_PERCENT: float = 0.1
    MAX_RISK_PERCENT: float = 2.0

    # Constantes Forex / Pocket Option
    PIP_SIZE_EURUSD: float = 0.0001
    PIP_VALUE_PER_STANDARD_LOT: float = 10.0

    # Ratios Risk / Reward minimaux
    MIN_RR_TP1: float = 1.0
    MIN_RR_TP2: float = 1.5

    # Contraintes Pocket Option (Options Binaires / Digitales)
    PO_MIN_EXPIRATION_SECONDS: int = 60
    PO_MIN_PAYOUT_PERCENT: float = 80.0
    PO_MAX_PAYOUT_PERCENT: float = 95.0

    @field_validator("TRADING_MODE")
    @classmethod
    def validate_trading_mode(cls, v: str) -> str:
        if v.lower() not in ["demo", "live"]:
            raise ValueError("TRADING_MODE doit être strictement 'demo' ou 'live'")
        return v.lower()

    @field_validator("DEFAULT_RISK_PERCENT")
    @classmethod
    def validate_risk_percent(cls, v: float) -> float:
        if not (0.1 <= v <= 2.0):
            raise ValueError("DEFAULT_RISK_PERCENT doit être compris entre 0.1% et 2.0%")
        return v

    def is_live_ready(self) -> tuple[bool, str]:
        """
        Vérification stricte avant toute autorisation d'ordre en mode réel.
        Retourne (is_ready, motif).
        """
        if self.TRADING_MODE != "live":
            return False, "Le mode actuel n'est pas configuré sur 'live'."
        if not self.TRADING_ENABLED:
            return False, "Le disjoncteur général TRADING_ENABLED est désactivé (False)."
        if not self.PO_SSID or not self.PO_USER_ID:
            return False, "Les identifiants Pocket Option (PO_SSID / PO_USER_ID) sont absents."
        return True, "Prêt pour le trading live."

# Instance unique accessible par l'ensemble des modules
settings = Settings()
