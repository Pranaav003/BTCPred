# tests/test_app_factory.py
from app import create_app
from app.config import config_by_name


def test_testing_config_registered():
    assert "testing" in config_by_name


def test_create_app_testing_uses_memory_db_and_testing_flag():
    app = create_app("testing")
    assert app.testing is True
    assert app.config["SQLALCHEMY_DATABASE_URI"] == "sqlite:///:memory:"


def test_create_app_testing_does_not_start_scheduler():
    app = create_app("testing")
    assert getattr(app, "scheduler_instance", None) is None
