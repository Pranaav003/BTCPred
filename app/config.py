"""Configuration classes for the Kalshi Signal application."""

import os

from dotenv import load_dotenv

load_dotenv()


class BaseConfig:
    """Base configuration shared across all environments."""

    SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///kalshi_signal.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
        "pool_use_lifo": True,
    }

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
    uri = os.environ.get("DATABASE_URL", "sqlite:///kalshi_signal.db")
    if uri.startswith("postgres://"):
        uri = uri.replace("postgres://", "postgresql://", 1)
    SQLALCHEMY_DATABASE_URI = uri
    if uri.startswith("postgresql://"):
        SQLALCHEMY_ENGINE_OPTIONS = {
            **BaseConfig.SQLALCHEMY_ENGINE_OPTIONS,
            "connect_args": {
                "sslmode": "require",
                "connect_timeout": 10,
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            },
        }


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
}
