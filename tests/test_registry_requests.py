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


def test_get_fee_timelines_sends_only_the_given_bounds_and_keys_by_listing(monkeypatch):
    calls = []

    def fake_get(url, params, headers, timeout):
        calls.append({"url": url, "params": params})
        return _Response([{"listingId": 7, "timeline": [{"effectiveFrom": "2026-01-01T00:00:00.000Z"}]},
                          {"listingId": 9, "timeline": []}])

    monkeypatch.setattr(registry_api.requests, "get", fake_get)
    client = RegistryClient(base_url="https://registry.example.com", api_key="service-key")

    timelines = client.get_fee_timelines([7, 9], start="2026-01-01T00:00:00Z")

    assert calls == [{"url": "https://registry.example.com/api/fees",
                      "params": {"listingIds": "7,9", "from": "2026-01-01T00:00:00Z"}}]
    assert timelines == {7: [{"effectiveFrom": "2026-01-01T00:00:00.000Z"}], 9: []}
