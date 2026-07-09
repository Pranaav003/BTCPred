"""NO trades must be gated to the model's calibrated zone: p_raw < no_max_p_raw."""
import importlib.util
import os
import sys
import types


def _load_signal_engine():
    app_stub = types.ModuleType("app")
    app_stub.__path__ = [os.path.join(os.path.dirname(__file__), "..", "app")]
    app_stub.__package__ = "app"
    sys.modules["app"] = app_stub
    for name in ("flask", "flask_sqlalchemy", "flask_migrate", "flask_wtf", "dotenv", "click"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sqla = types.ModuleType("sqlalchemy"); sqla.func = type("func", (), {})
    sys.modules.setdefault("sqlalchemy", sqla)
    ml = types.ModuleType("app.model_loader"); ml.predict_proba_raw = lambda _: 0.5
    sys.modules["app.model_loader"] = ml
    dbh = types.ModuleType("app.db_helpers")
    dbh.get_setting = lambda k, d=None: d
    dbh.set_setting = lambda k, v: None
    sys.modules["app.db_helpers"] = dbh
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "signal_engine.py"))
    spec = importlib.util.spec_from_file_location("app.signal_engine", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["app.signal_engine"] = mod
    spec.loader.exec_module(mod)
    return mod


SE = _load_signal_engine()


def _ensemble_no(p_raw, no_max_p_raw=0.20):
    # market 0.62, model p_raw -> bearish gap >= 0.25 threshold; in-window; NO entry ok.
    return SE.evaluate_ensemble_signal(
        p_market=0.62, p_raw=p_raw, seconds_to_close=100, entry_bucket=60,
        yes_cutoff=0.72, max_entry_yes=0.65, max_entry_no=0.80,
        mispricing_threshold=0.25, min_seconds=60, max_seconds=120,
        no_max_p_raw=no_max_p_raw,
    )


def test_ensemble_blocks_no_when_praw_at_or_above_cap():
    # gap = 0.62 - 0.36 = 0.26 >= 0.25, but p_raw 0.36 >= 0.20 -> blocked.
    result = _ensemble_no(p_raw=0.36)
    assert result.signal == "NO SIGNAL"


def test_ensemble_allows_no_when_praw_below_cap():
    # gap = 0.62 - 0.15 = 0.47 >= 0.25, p_raw 0.15 < 0.20 -> NO allowed.
    result = _ensemble_no(p_raw=0.15)
    assert result.signal == "PAPER BUY NO"


def test_ensemble_gate_respects_setting_value():
    # With a looser cap of 0.40, p_raw 0.36 should be allowed again.
    result = _ensemble_no(p_raw=0.36, no_max_p_raw=0.40)
    assert result.signal == "PAPER BUY NO"


def test_mispricing_blocks_no_when_praw_above_cap():
    result = SE.evaluate_mispricing_signal(
        p_market=0.62, p_raw=0.36, seconds_to_close=100, entry_bucket=60,
        min_seconds=60, max_seconds=120, mispricing_threshold=0.25,
        max_entry_price_yes=0.65, max_entry_price_no=0.80, no_max_p_raw=0.20,
    )
    assert result.signal == "NO SIGNAL"


def test_mispricing_allows_no_when_praw_below_cap():
    result = SE.evaluate_mispricing_signal(
        p_market=0.62, p_raw=0.15, seconds_to_close=100, entry_bucket=60,
        min_seconds=60, max_seconds=120, mispricing_threshold=0.25,
        max_entry_price_yes=0.65, max_entry_price_no=0.80, no_max_p_raw=0.20,
    )
    assert result.signal == "PAPER BUY NO"
