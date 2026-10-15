from gnomepy.registry import api as registry_api
from gnomepy.registry.api import REQUEST_TIMEOUT_SECONDS, RegistryClient


class _Response:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


def test_get_sends_bool_filters_as_lowercase_with_a_timeout(monkeypatch):
    calls = []

    def fake_get(url, params, headers, timeout):
        calls.append({"url": url, "params": params, "timeout": timeout})
        return _Response([])

    monkeypatch.setattr(registry_api.requests, "get", fake_get)
    client = RegistryClient(base_url="https://registry.example.com", api_key="service-key")

    client.get_event(resolved=True, exchange_id=2)
    client.get_event(resolved=False)

    assert calls == [
        {"url": "https://registry.example.com/api/events",
         "params": {"resolved": "true", "exchangeId": 2, "limit": 1000, "offset": 0}, "timeout": REQUEST_TIMEOUT_SECONDS},
        {"url": "https://registry.example.com/api/events",
         "params": {"resolved": "false", "limit": 1000, "offset": 0}, "timeout": REQUEST_TIMEOUT_SECONDS},
    ]
