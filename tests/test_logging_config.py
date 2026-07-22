# tests/test_logging_config.py
# Import create_app / configure_logging lazily inside tests so collecting this
# file does not pollute sys.modules for the stub-style tests.
import logging

import pytest

_STDOUT_NAME = "btcpred-stdout"
_FILE_NAME = "btcpred-file"


@pytest.fixture(autouse=True)
def _clean_root_handlers():
    """Snapshot and restore root logger state so tests don't leak handlers."""
    root = logging.getLogger()
    saved = list(root.handlers)
    saved_level = root.level
    # Remove our named handlers so each test starts from a known slate.
    root.handlers = [h for h in root.handlers if getattr(h, "name", None) not in (_STDOUT_NAME, _FILE_NAME)]
    try:
        yield
    finally:
        root.handlers = saved
        root.setLevel(saved_level)


def _named(root, name):
    return [h for h in root.handlers if getattr(h, "name", None) == name]


def test_configure_logging_attaches_named_stdout_handler_with_format():
    from app.logging_config import configure_logging

    configure_logging()
    root = logging.getLogger()
    handlers = _named(root, _STDOUT_NAME)
    assert len(handlers) == 1
    fmt = handlers[0].formatter._fmt
    assert "%(asctime)s" in fmt
    assert "%(levelname)s" in fmt
    assert "%(name)s" in fmt
    assert "%(message)s" in fmt


def test_configure_logging_default_level_is_info(monkeypatch):
    from app.logging_config import configure_logging

    monkeypatch.delenv("LOG_LEVEL", raising=False)
    configure_logging()
    assert logging.getLogger().level == logging.INFO


def test_configure_logging_respects_log_level_env(monkeypatch):
    from app.logging_config import configure_logging

    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    configure_logging()
    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_is_idempotent_no_duplicate_handlers():
    # The reproduced refutation defect: many create_app calls must not pile up handlers.
    from app.logging_config import configure_logging

    configure_logging()
    configure_logging()
    configure_logging()
    root = logging.getLogger()
    assert len(_named(root, _STDOUT_NAME)) == 1


def test_no_rotating_file_handler_without_log_file_env(monkeypatch):
    from app.logging_config import configure_logging

    monkeypatch.delenv("LOG_FILE", raising=False)
    configure_logging()
    root = logging.getLogger()
    assert _named(root, _FILE_NAME) == []


def test_rotating_file_handler_added_when_log_file_set(monkeypatch, tmp_path):
    from logging.handlers import RotatingFileHandler

    from app.logging_config import configure_logging

    log_path = tmp_path / "app.log"
    monkeypatch.setenv("LOG_FILE", str(log_path))
    configure_logging()
    root = logging.getLogger()
    files = _named(root, _FILE_NAME)
    assert len(files) == 1
    assert isinstance(files[0], RotatingFileHandler)
    assert files[0].maxBytes == 5_000_000
    assert files[0].backupCount == 3


def test_file_handler_added_even_when_stdout_already_present(monkeypatch, tmp_path):
    # Reproduced pair defect: a blanket "if root.handlers: return" guard would
    # suppress the file handler after stdout is already attached. The guard must
    # be per-handler so the file handler can still be added when LOG_FILE is set.
    from app.logging_config import configure_logging

    configure_logging()  # attaches stdout only (no LOG_FILE)
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "app.log"))
    configure_logging()  # must now add the file handler despite stdout present
    root = logging.getLogger()
    assert len(_named(root, _STDOUT_NAME)) == 1
    assert len(_named(root, _FILE_NAME)) == 1


def test_file_handler_idempotent(monkeypatch, tmp_path):
    from app.logging_config import configure_logging

    monkeypatch.setenv("LOG_FILE", str(tmp_path / "app.log"))
    configure_logging()
    configure_logging()
    root = logging.getLogger()
    assert len(_named(root, _FILE_NAME)) == 1
