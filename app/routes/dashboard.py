"""Dashboard page routes."""

from datetime import datetime, timezone

from flask import Blueprint, redirect, render_template, url_for

from app.db_helpers import get_setting
from app.model_loader import get_model
from app.resolver import get_resolution_summary

dashboard_bp = Blueprint("dashboard", __name__)


def _typed_settings() -> dict:
    """Return runtime settings with typed values for templates."""
    return {
        "yes_cutoff": float(get_setting("yes_cutoff", "0.65")),
        "no_cutoff": float(get_setting("no_cutoff", "0.35")),
        "min_seconds_to_close": int(get_setting("min_seconds_to_close", "30")),
        "max_seconds_to_close": int(get_setting("max_seconds_to_close", "180")),
        "poll_interval_seconds": int(get_setting("poll_interval_seconds", "30")),
        "enable_no_signals": get_setting("enable_no_signals", "false") == "true",
        "auto_trade_enabled": get_setting("auto_trade_enabled", "false") == "true",
        "paper_trade_size": float(get_setting("paper_trade_size", "10.0")),
        "risk_profile": get_setting("risk_profile", "moderate") or "moderate",
        "signal_mode": get_setting("signal_mode", "agreement") or "agreement",
        "mispricing_threshold": float(get_setting("mispricing_threshold", "0.10")),
        "max_entry_price_yes": float(get_setting("max_entry_price_yes", "0.85")),
        "max_entry_price_no": float(get_setting("max_entry_price_no", "0.85")),
        "min_expected_profit": float(get_setting("min_expected_profit", "0.10")),
        "max_reversal_risk": float(get_setting("max_reversal_risk", "0.65")),
        "max_daily_loss": float(get_setting("max_daily_loss", "200.0") or 200.0),
        "high_conviction_volatility_override": float(get_setting("high_conviction_volatility_override", "0.80")),
        "scheduler_running": get_setting("scheduler_running", "false") == "true",
    }


@dashboard_bp.route("/")
def home():
    """Root URL: send users to the main dashboard (avoids stale 'placeholder' confusion)."""
    return redirect(url_for("dashboard.dashboard"))


@dashboard_bp.route("/dashboard")
def dashboard():
    scheduler_running = get_setting("scheduler_running", "false") == "true"
    poll_interval = int(get_setting("poll_interval_seconds", "30"))

    try:
        bundle = get_model()
        model_loaded = True
    except Exception:
        bundle = None
        model_loaded = False

    model_age_days = None
    if bundle is not None:
        trained_at = bundle.get("trained_at")
        if trained_at is not None:
            try:
                if isinstance(trained_at, str):
                    trained_at = datetime.fromisoformat(trained_at)
                if trained_at.tzinfo is None:
                    trained_at = trained_at.replace(tzinfo=timezone.utc)
                model_age_days = (datetime.now(timezone.utc) - trained_at).days
            except Exception:
                model_age_days = None

    auto_trade_enabled = get_setting("auto_trade_enabled", "false") == "true"
    enable_no_signals = get_setting("enable_no_signals", "false") == "true"
    try:
        resolution_summary = get_resolution_summary()
    except Exception:
        resolution_summary = {"pending_resolution": 0}
    pending_resolution = int(resolution_summary.get("pending_resolution", 0) or 0)
    paper_trading_enabled = get_setting("paper_trading_enabled", "false") == "true"
    paper_trade_size = float(get_setting("paper_trade_size", "10.0"))

    return render_template(
        "dashboard.html",
        scheduler_running=scheduler_running,
        poll_interval=poll_interval,
        model_loaded=model_loaded,
        model_age_days=model_age_days,
        auto_trade_enabled=auto_trade_enabled,
        enable_no_signals=enable_no_signals,
        pending_resolution=pending_resolution,
        paper_trading_enabled=paper_trading_enabled,
        paper_trade_size=paper_trade_size,
    )


@dashboard_bp.route("/monitor")
def monitor():
    scheduler_running = get_setting("scheduler_running", "false") == "true"
    try:
        get_model()
        model_loaded = True
    except Exception:
        model_loaded = False
    auto_trade_enabled = get_setting("auto_trade_enabled", "false") == "true"
    enable_no_signals = get_setting("enable_no_signals", "false") == "true"
    try:
        resolution_summary = get_resolution_summary()
    except Exception:
        resolution_summary = {"pending_resolution": 0}
    pending_resolution = int(resolution_summary.get("pending_resolution", 0) or 0)
    return render_template(
        "monitor.html",
        scheduler_running=scheduler_running,
        model_loaded=model_loaded,
        auto_trade_enabled=auto_trade_enabled,
        enable_no_signals=enable_no_signals,
        pending_resolution=pending_resolution,
    )


@dashboard_bp.route("/analytics")
def analytics():
    return render_template("analytics.html", page_title="Analytics")


@dashboard_bp.route("/settings")
def settings():
    return render_template("settings.html", settings=_typed_settings())
