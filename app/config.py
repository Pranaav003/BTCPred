"""Configuration classes for the Kalshi Signal application."""

import os

from dotenv import load_dotenv

load_dotenv()


class BaseConfig:
    """Base configuration shared across all environments."""

    SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///kalshi_signal.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "60"))
    YES_CUTOFF = float(os.getenv("YES_CUTOFF", "0.65"))
    NO_CUTOFF = float(os.getenv("NO_CUTOFF", "0.35"))
    MIN_SECONDS_TO_CLOSE = int(os.getenv("MIN_SECONDS_TO_CLOSE", "300"))
    MAX_SECONDS_TO_CLOSE = int(os.getenv("MAX_SECONDS_TO_CLOSE", "86400"))

    SCHEDULER_API_ENABLED = True


class DevelopmentConfig(BaseConfig):
    """Development settings."""

    DEBUG = True


class ProductionConfig(BaseConfig):
    """Production settings."""

    DEBUG = False


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
}
