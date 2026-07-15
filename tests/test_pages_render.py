# tests/test_pages_render.py
import pytest


@pytest.mark.parametrize("path", ["/dashboard", "/monitor", "/analytics", "/settings"])
def test_page_renders_200(client, path):
    resp = client.get(path)
    assert resp.status_code == 200
    assert b"<html" in resp.data.lower()


def test_root_redirects(client):
    resp = client.get("/")
    assert resp.status_code in (301, 302)
