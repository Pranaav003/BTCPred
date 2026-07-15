# tests/test_control_static.py
import pathlib


def test_control_static_files_exist():
    assert pathlib.Path("app/static/js/control.js").exists()
    assert pathlib.Path("app/static/css/control.css").exists()


def test_control_template_references_assets():
    html = pathlib.Path("app/templates/control.html").read_text()
    assert "js/control.js" in html


def test_control_js_calls_state_and_defaults_endpoints():
    js = pathlib.Path("app/static/js/control.js").read_text()
    assert "/api/control/state" in js
    assert "/api/control/apply-defaults" in js
