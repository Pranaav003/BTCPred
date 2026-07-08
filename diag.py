"""Ad-hoc production diagnostic for the Kalshi signal bot.

Dumps all runtime settings and evaluates the CURRENT live market signal so you
can see exactly why the bot is or isn't trading right now.

SAFE TO RUN ON THE LIVE INSTANCE: sets WERKZEUG_RUN_MAIN=false before importing
the app so create_app() does NOT start a second (order-placing) scheduler.

Usage (Render shell, from project root ~/project/src):
    python3 diag.py
"""
from __future__ import annotations

import os

# Must be set BEFORE importing app so create_app() skips scheduler startup
# (see app/__init__.py: should_start_scheduler checks this env var).
os.environ["WERKZEUG_RUN_MAIN"] = "false"

from app import create_app  # noqa: E402


def main() -> None:
    app = create_app()
    with app.app_context():
        from app.models import AppSettings

        print("=== settings ===")
        for s in AppSettings.query.order_by(AppSettings.key).all():
            print(f"{s.key} = {s.value}")

        print("\n=== current live signal ===")
        try:
            from app.feature_engineering import get_live_snapshot
            from app.signal_engine import evaluate_live_signal

            snap = get_live_snapshot()
            print("snapshot:", bool(snap))
            if snap:
                print("ticker  :", snap.get("market_ticker"))
                result = evaluate_live_signal(snap)
                if result is None:
                    print("signal  : (none returned — model unavailable or empty snapshot)")
                else:
                    print("signal  :", result.signal)
                    print("stc     :", result.seconds_to_close, "s to close")
                    print("p_market:", round(float(result.p_market), 3))
                    print("p_raw   :", round(float(result.p_raw), 3))
                    print("gap     :", round(abs(float(result.p_raw) - float(result.p_market)), 3))
                    print("reason  :", result.reason)
        except Exception as exc:  # diagnostic: surface, don't swallow
            import traceback

            traceback.print_exc()
            print("live signal eval failed:", exc)


if __name__ == "__main__":
    main()
