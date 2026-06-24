"""Application factory for the Kalshi Signal Flask app."""

import json
import os

import click
from flask import Flask
from flask_wtf.csrf import CSRFProtect
from sqlalchemy import text

from app.config import config_by_name
from app.extensions import db
from app.routes.api import api_bp
from app.routes.dashboard import dashboard_bp


def _ensure_signal_schema_columns(app: Flask) -> None:
    """Backfill additive Signal columns for existing SQLite databases."""
    if db.engine.dialect.name != "sqlite":
        # SQLite-specific ALTER backfill; Postgres should use model-managed schema.
        return
    try:
        existing = db.session.execute(text("PRAGMA table_info(signals)")).fetchall()
        names = {row[1] for row in existing}
        if "orderbook_mid" not in names:
            db.session.execute(text("ALTER TABLE signals ADD COLUMN orderbook_mid FLOAT"))
            app.logger.info("Added missing signals.orderbook_mid column")
        if "orderbook_available" not in names:
            db.session.execute(text("ALTER TABLE signals ADD COLUMN orderbook_available BOOLEAN DEFAULT 0"))
            app.logger.info("Added missing signals.orderbook_available column")
        db.session.commit()
    except Exception:
        db.session.rollback()
        app.logger.exception("Failed to ensure Signal schema columns")


def create_app(config_name: str | None = None) -> Flask:
    """Create and configure the Flask application instance."""
    app = Flask(__name__)

    selected_config = config_name or "development"
    app.config.from_object(config_by_name[selected_config])

    db.init_app(app)
    csrf = CSRFProtect(app)

    # Import models after db initialization so metadata is registered.
    from app import models  # noqa: F401
    from app.db_helpers import seed_default_settings

    with app.app_context():
        db.create_all()
        _ensure_signal_schema_columns(app)
        seed_default_settings()
        # Scheduler starts automatically; auto-trade requires explicit user activation for safety.
        from app.models import AppSettings
        AppSettings.set("scheduler_running", "true")

    app.scheduler_instance = None
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(api_bp)

    # Register routes (including /api/health) before background scheduler so probes work immediately.
    should_start_scheduler = (
        not app.testing
        and selected_config != "testing"
        and os.environ.get("WERKZEUG_RUN_MAIN") != "false"
    )
    if should_start_scheduler:
        try:
            from app.scheduler import init_scheduler

            app.scheduler_instance = init_scheduler(app)
        except Exception as exc:
            app.logger.error("Scheduler failed to start: %s", exc)

    @app.cli.command("train-model")
    def train_model_command() -> None:
        """Retrain and save the raw-feature model artifact."""
        from train_raw_model import DATA_PATH, MODEL_OUTPUT, train_model

        click.echo("Starting model training...")
        try:
            metadata = train_model(DATA_PATH, MODEL_OUTPUT)
        except FileNotFoundError as exc:
            click.echo(f"Training failed: {exc}")
            return
        except Exception as exc:  # pragma: no cover - defensive CLI guard
            click.echo(f"Training failed: {exc}")
            return

        if not metadata:
            click.echo("Training cancelled.")
            return

        click.echo("Training complete.")
        click.echo(json.dumps(metadata, indent=2, default=str))

    return app
