"""_apply_contract_cap clamps live position size to the live_max_contracts setting."""
import os
import sys
import types

import pytest

_REAL_APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
_SHADOW_ROOTS = ("app", "flask", "flask_sqlalchemy", "flask_migrate", "flask_wtf",
                 "sqlalchemy", "click", "dotenv")


def _is_shadow(key):
    return any(key == r or key.startswith(r + ".") for r in _SHADOW_ROOTS)


@pytest.fixture
def scheduler():
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if _is_shadow(k)}
    app_pkg = types.ModuleType("app"); app_pkg.__path__ = [_REAL_APP_DIR]; app_pkg.__package__ = "app"
    sys.modules["app"] = app_pkg
    try:
        import app.scheduler as scheduler
        yield scheduler
    finally:
        for k in [k for k in list(sys.modules) if _is_shadow(k)]:
            sys.modules.pop(k, None)
        sys.modules.update(saved)


def test_clamps_to_cap(scheduler):
    assert scheduler._apply_contract_cap(7, "1") == 1
    assert scheduler._apply_contract_cap(7, "3") == 3


def test_no_clamp_when_under_cap(scheduler):
    assert scheduler._apply_contract_cap(2, "5") == 2


def test_unset_means_no_cap(scheduler):
    assert scheduler._apply_contract_cap(9, "") == 9
    assert scheduler._apply_contract_cap(9, None) == 9


def test_nonnumeric_or_nonpositive_means_no_cap(scheduler):
    assert scheduler._apply_contract_cap(9, "abc") == 9
    assert scheduler._apply_contract_cap(9, "0") == 9
    assert scheduler._apply_contract_cap(9, "-2") == 9
