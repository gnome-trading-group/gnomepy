import pytest

from gnomepy.registry import api as registry_api
from gnomepy.registry.api import RegistryClient


class _Response:
    def raise_for_status(self):
        pass

    def json(self):
        return {"sessionId": "s1", "status": "STOPPED"}


@pytest.fixture
def captured(monkeypatch):
    calls = []

    def fake_post(url, json, headers):
        calls.append({"url": url, "json": json, "headers": headers})
        return _Response()

    monkeypatch.setattr(registry_api.requests, "post", fake_post)
    monkeypatch.setattr(registry_api, "get_id_token", lambda: "id-token-123")
    return calls


def test_stop_strategy_session_uses_cognito_route_and_id_token(captured):
    client = RegistryClient(base_url="https://registry.example.com", api_key="service-key")

    client.stop_strategy_session("s1")

    assert captured == [{
        "url": "https://registry.example.com/api/cognito/strategy-sessions/stop",
        "json": {"sessionId": "s1"},
        "headers": {"Authorization": "id-token-123", "Content-Type": "application/json"},
    }]


def test_stop_strategy_session_never_sends_the_api_key(captured):
    client = RegistryClient(base_url="https://registry.example.com", api_key="service-key")

    client.stop_strategy_session("s1")

    assert "x-api-key" not in {k.lower() for k in captured[0]["headers"]}


def test_stop_strategy_session_as_service_uses_api_key_route_and_names_the_actor(monkeypatch):
    calls = []

    def fake_post(url, json, headers):
        calls.append({"url": url, "json": json, "headers": headers})
        return _Response()

    monkeypatch.setattr(registry_api.requests, "post", fake_post)
    client = RegistryClient(base_url="https://registry.example.com", api_key="service-key")

    client.stop_strategy_session_as_service("s1", actor="launcher:cs2-prematch")

    assert calls == [{
        "url": "https://registry.example.com/api/strategy-sessions/stop",
        "json": {"sessionId": "s1", "actor": "launcher:cs2-prematch"},
        "headers": {"x-api-key": "service-key", "Content-Type": "application/json"},
    }]
