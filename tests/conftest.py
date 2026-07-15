"""Shared pytest fixtures for the BTCPred test suite.

Two test styles coexist here:
  * "stub" tests (e.g. test_paper_trading.py) install fake flask/sqlalchemy/app
    modules into sys.modules at collection time and importlib-load the module
    under test against those fakes;
  * "real-app" tests (fixtures below) build a real Flask app on the testing config.

Mixed in one pytest session these collide via sys.modules. The autouse
`_isolate_sys_modules` fixture keeps them apart by restoring the app/flask-family
modules to their pre-collection baseline around every test — WITHOUT touching
pytest, numpy, or the test modules themselves. Real-app fixtures import
`create_app` lazily so merely importing this conftest does not pollute collection.
"""
import sys

import pytest

# Baseline captured at conftest import — before test collection and before any
# app/flask import here (create_app is imported lazily inside the `app` fixture).
_BASELINE = dict(sys.modules)

# Only these top-level package roots are isolated; everything else is left alone.
_ISOLATE_ROOTS = {
    "app", "flask", "flask_sqlalchemy", "flask_migrate", "flask_wtf",
    "sqlalchemy", "apscheduler", "dotenv", "click",
}


def _root(name: str) -> str:
    return name.split(".", 1)[0]


def _restore_isolated() -> None:
    """Reset app/flask-family entries in sys.modules to the pre-collection baseline."""
    for name in list(sys.modules):
        if _root(name) in _ISOLATE_ROOTS and name not in _BASELINE:
            del sys.modules[name]
    for name, mod in _BASELINE.items():
        if _root(name) in _ISOLATE_ROOTS:
            sys.modules[name] = mod


@pytest.fixture(autouse=True)
def _isolate_sys_modules():
    """Give every test a clean app/flask module slate; clean up after it too."""
    _restore_isolated()
    try:
        yield
    finally:
        _restore_isolated()


@pytest.fixture
def app():
    """A fresh testing app with an app context and in-memory DB."""
    from app import create_app  # lazy: avoid polluting collection for stub tests

    application = create_app("testing")
    ctx = application.app_context()
    ctx.push()
    try:
        yield application
    finally:
        ctx.pop()


@pytest.fixture
def client(app):
    """Flask test client bound to the testing app."""
    return app.test_client()
