"""seed_default_settings must NOT revert a validated 0.25 mispricing_threshold,
and must seed the new guard keys. Regression for the config-drift mechanism."""
import importlib.util
import os
import sys
import types


def _load_db_helpers_with_fake_store():
    """Load app.db_helpers with get_setting/set_setting backed by an in-memory dict."""
    store = {}

    def fake_get(key, default=None):
        return store.get(key, default)

    def fake_set(key, value):
        store[key] = value
        return None

    # Stub the 'app' package and app.models so db_helpers imports cleanly.
    app_stub = types.ModuleType("app")
    app_stub.__path__ = [os.path.join(os.path.dirname(__file__), "..", "app")]
    sys.modules["app"] = app_stub
    models_stub = types.ModuleType("app.models")
    for name in ("AppSettings", "Market", "PaperTrade", "Portfolio", "Signal", "db"):
        setattr(models_stub, name, type(name, (), {}))
    sys.modules["app.models"] = models_stub
    sys.modules.setdefault("sqlalchemy", types.ModuleType("sqlalchemy"))
    sys.modules["sqlalchemy"].func = type("func", (), {})
    orm = types.ModuleType("sqlalchemy.orm")
    orm.contains_eager = lambda *a, **k: None
    sys.modules["sqlalchemy.orm"] = orm

    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "db_helpers.py"))
    spec = importlib.util.spec_from_file_location("app.db_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["app.db_helpers"] = mod
    spec.loader.exec_module(mod)
    # Swap in fakes for the store-backed helpers used by seed_default_settings.
    mod.get_setting = fake_get
    mod.set_setting = fake_set
    return mod, store


def test_seed_does_not_revert_025_threshold():
    mod, store = _load_db_helpers_with_fake_store()
    store["mispricing_threshold"] = "0.25"
    mod.seed_default_settings()
    assert store["mispricing_threshold"] == "0.25"  # NOT reverted to 0.10


def test_seed_default_threshold_is_025_when_missing():
    mod, store = _load_db_helpers_with_fake_store()
    mod.seed_default_settings()
    assert store["mispricing_threshold"] == "0.25"


def test_seed_corrects_absurd_threshold():
    mod, store = _load_db_helpers_with_fake_store()
    store["mispricing_threshold"] = "0.90"
    mod.seed_default_settings()
    assert store["mispricing_threshold"] == "0.25"


def test_seed_new_guard_keys():
    mod, store = _load_db_helpers_with_fake_store()
    mod.seed_default_settings()
    assert store["no_max_p_raw"] == "0.20"
    assert store["live_max_contracts"] == ""
