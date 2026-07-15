# tests/test_app_factory.py
# Import create_app lazily inside each test (not at module top) so collecting
# this file does not load the real `app` package into sys.modules, which would
# defeat the collection-time stubbing in tests like test_paper_trading.py.


def test_testing_config_registered():
    from app.config import config_by_name
    assert "testing" in config_by_name


def test_create_app_testing_uses_memory_db_and_testing_flag():
    from app import create_app
    app = create_app("testing")
    assert app.testing is True
    assert app.config["SQLALCHEMY_DATABASE_URI"] == "sqlite:///:memory:"


def test_create_app_testing_does_not_start_scheduler():
    from app import create_app
    app = create_app("testing")
    assert getattr(app, "scheduler_instance", None) is None
