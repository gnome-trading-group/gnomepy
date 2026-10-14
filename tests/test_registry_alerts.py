import pytest

from gnomepy import config as gnome_config
from gnomepy.registry import api as registry_api
from gnomepy.registry.api import RegistryClient


class _Response:
    def raise_for_status(self):
        pass

    def json(self):
        return {"key": "LAUNCH_FAILED:-:7"}


@pytest.fixture
def captured(monkeypatch):
    calls = []

    def fake_post(url, json, headers):
        calls.append({"url": url, "json": json, "headers": headers})
        return _Response()

    monkeypatch.setattr(registry_api.requests, "post", fake_post)
    return calls


def test_send_alert_posts_to_the_alerts_api_with_the_api_key(captured, monkeypatch):
    monkeypatch.setenv("ALERTS_API_URL", "https://alerts.example.com/api/")
    client = RegistryClient(base_url="https://registry.example.com", api_key="service-key")

    client.send_alert("LAUNCH_FAILED", "Couldn't launch arb", body="422: missing MAX_POSITION", strategy_id=7,
                      actor="launcher:cs2-prematch", labels={"rule": "cs2-prematch"})

    assert captured == [{
        "url": "https://alerts.example.com/api/alerts",
        "json": {"type": "LAUNCH_FAILED", "title": "Couldn't launch arb", "status": "FIRING",
                 "body": "422: missing MAX_POSITION", "actor": "launcher:cs2-prematch", "strategyId": 7,
                 "labels": {"rule": "cs2-prematch"}},
        "headers": {"x-api-key": "service-key", "Content-Type": "application/json"},
    }]


def test_send_alert_resolves_by_key(captured):
    client = RegistryClient(base_url="https://registry.example.com", api_key="service-key")

    client.send_alert("CUSTOM", "fair value diverged", key="fv:7", status="RESOLVED",
                      alerts_url="https://alerts.example.com/api")

    assert captured[0]["url"] == "https://alerts.example.com/api/alerts"
    assert captured[0]["json"] == {"type": "CUSTOM", "title": "fair value diverged", "status": "RESOLVED", "key": "fv:7"}


def test_alerts_url_falls_back_to_the_alerting_stacks_parameter(monkeypatch):
    monkeypatch.delenv("ALERTS_API_URL", raising=False)

    class _Ssm:
        def get_parameter(self, Name):
            assert Name == "/gnome/alerting/api-url"
            return {"Parameter": {"Value": "https://from-ssm.example.com/api/"}}

    monkeypatch.setattr(gnome_config.boto3, "client", lambda service, region_name: _Ssm())

    assert gnome_config.resolve_alerts_api_url() == "https://from-ssm.example.com/api/"
