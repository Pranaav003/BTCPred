"""Shared pytest fixtures for the BTCPred test suite."""
import sys

import pytest

from app import create_app

# Snapshot sys.modules at conftest import time — BEFORE any test-file module-level
# code runs (e.g. test_paper_trading.py, test_no_max_p_raw_gate.py install stubs
# unconditionally at collection time).  Using this baseline lets every test start
# from the clean, real-module state instead of the stub-polluted post-collection
# state.
_CLEAN_SYS_MODULES = dict(sys.modules)


@pytest.fixture(autouse=True)
def _isolate_sys_modules():
    """Snapshot and restore sys.modules around every test.

    Several existing unit tests (e.g. test_paper_trading.py) stub out
    flask/sqlalchemy/apscheduler in sys.modules and importlib-load a module
    under test. Without cleanup that pollution leaks across tests and breaks
    later tests that need the real modules (order-dependent failures in the
    full suite). Restoring the snapshot after each test neutralizes it.
    """
    saved = dict(_CLEAN_SYS_MODULES)
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name not in saved:
                del sys.modules[name]
        sys.modules.update(saved)


@pytest.fixture
def app():
    """A fresh testing app with an app context and in-memory DB."""
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
