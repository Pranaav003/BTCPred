"""Central logging configuration for the BTCPred app.

Attaches handlers to the ROOT logger so every module's
``logging.getLogger(__name__)`` inherits the configuration.

Idempotency note (a refutation-confirmed defect): a module-level "_configured"
flag does NOT work here, because the test suite deletes ``app.*`` from
``sys.modules`` between tests (see tests/conftest.py), which reimports this
module and resets any module-level flag — while ``logging.root`` is a process
global that keeps its handlers. So the guard is anchored to the root logger
itself via NAMED handlers, and is applied per-handler so that the file handler
can still be added when LOG_FILE is set on a later call even if stdout is
already attached.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_STDOUT_HANDLER_NAME = "btcpred-stdout"
_FILE_HANDLER_NAME = "btcpred-file"

_FILE_MAX_BYTES = 5_000_000
_FILE_BACKUP_COUNT = 3


def _has_named_handler(root: logging.Logger, name: str) -> bool:
    return any(getattr(h, "name", None) == name for h in root.handlers)


def configure_logging(app=None) -> None:
    """Configure root logging: timestamped stdout handler + optional rotating file.

    Idempotent: repeated calls (the test suite builds many apps) never
    accumulate duplicate handlers. ``app`` is accepted for a uniform
    ``configure_logging(app)`` call site but is not required.
    """
    root = logging.getLogger()

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT)

    if not _has_named_handler(root, _STDOUT_HANDLER_NAME):
        stream = logging.StreamHandler(sys.stdout)
        stream.set_name(_STDOUT_HANDLER_NAME)
        stream.setFormatter(formatter)
        root.addHandler(stream)

    # Rotating file handler only where a file sink is meaningful (local/other
    # deploys). On Render stdout is captured by the platform, so a file on the
    # ephemeral container disk would be wasted — hence gated on LOG_FILE.
    log_file = os.environ.get("LOG_FILE")
    if log_file and not _has_named_handler(root, _FILE_HANDLER_NAME):
        file_handler = RotatingFileHandler(
            log_file, maxBytes=_FILE_MAX_BYTES, backupCount=_FILE_BACKUP_COUNT
        )
        file_handler.set_name(_FILE_HANDLER_NAME)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
