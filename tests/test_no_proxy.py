"""Tests sans réseau externe, sans authentification et sans ordre."""
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("explicit", [False, True])
def test_safe_startup_without_proxy(explicit):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("PO_", "TRADING_"))
           and key.lower() not in {"http_proxy", "https_proxy", "all_proxy"}}
    env.update(TRADING_MODE="demo", TRADING_ENABLED="false",
               PO_LIVE_ENABLED="false")
    if explicit:
        env["PO_PROXY_ENABLED"] = "false"
    subprocess.run([sys.executable, "-c", '''
from Config import Settings
from po_connector import PocketOptionConnector
from fastapi.testclient import TestClient
from server import app
settings = Settings(_env_file=None)
assert settings.PO_PROXY_ENABLED is False
assert settings.TRADING_MODE == "demo"
assert settings.TRADING_ENABLED is False
status = PocketOptionConnector().get_status()
assert status["proxy_enabled"] is False
assert status["mode"] == "demo"
assert status["trading_enabled"] is False
assert status["live_enabled"] is False
assert status["client_loaded"] is False
with TestClient(app) as client:
    for path in ("/", "/health", "/config"):
        assert client.get(path).status_code == 200
'''], env=env, check=True, timeout=30)