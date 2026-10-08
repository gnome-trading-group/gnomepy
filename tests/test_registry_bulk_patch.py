from gnomepy.registry import api as registry_api
from gnomepy.registry.api import RegistryClient


class _Response:
    ok = True

    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


def test_bulk_patch_event_contracts_sends_camel_case_batches(monkeypatch):
    calls = []

    def fake_patch(url, json, headers):
        calls.append({"url": url, "json": json, "headers": headers})
        return _Response([{"event_contract_id": item["eventContractId"]} for item in json])

    monkeypatch.setattr(registry_api.requests, "patch", fake_patch)
    client = RegistryClient(base_url="https://registry.example.com", api_key="service-key")

    result = client.bulk_patch_event_contracts([
        {"event_contract_id": 1, "settlement_price": 1_000_000_000},
        {"event_contract_id": 2, "settlement_price": 0},
    ])

    assert calls == [{
        "url": "https://registry.example.com/api/event-contracts",
        "json": [{"eventContractId": 1, "settlementPrice": 1_000_000_000}, {"eventContractId": 2, "settlementPrice": 0}],
        "headers": {"x-api-key": "service-key", "Content-Type": "application/json"},
    }]
    assert result == [{"event_contract_id": 1}, {"event_contract_id": 2}]
