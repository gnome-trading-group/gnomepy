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


def test_get_event_sends_list_filters_as_one_comma_separated_value_and_reads_volume(monkeypatch):
    calls = []

    def fake_get(url, params, headers, timeout):
        calls.append(params)
        return _Response([{"event_id": 7, "title": "A", "description": None, "category": "Sports", "tags": ["nfl"],
                           "resolved": False, "resolved_at": None, "expiry": None, "date_modified": "x",
                           "date_created": "x", "volume_24h": 1250.5, "volume_updated_at": "y"}])

    monkeypatch.setattr(registry_api.requests, "get", fake_get)
    client = RegistryClient(base_url="https://registry.example.com", api_key="service-key")

    events = client.get_event(exchange_ids=[1, 2], categories=["Sports"], tags=["nfl", "nba"],
                              min_volume_24h=1000, sort_by="volume_24h", sort_order="desc")

    assert calls == [{"exchangeIds": "1,2", "categories": "Sports", "tags": "nfl,nba", "minVolume24h": 1000,
                      "sortBy": "volume_24h", "sortOrder": "desc", "limit": 1000, "offset": 0}]
    assert events[0].volume_24h == 1250.5


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


def test_get_streams_joins_id_lists_and_keeps_known_fields(monkeypatch):
    calls = []
    row = {
        "stream_id": 31, "source_id": 2, "source_code": "KALSHI", "source_kind": "VENUE", "subject_kind": "LISTING",
        "listing_id": 789, "native_subject_id": "MKT:yes", "family": "book", "region": None,
        "resolved_region": "us-east-1", "event_id": None, "starts_at": None, "ends_at": None, "active": True,
        "date_created": "2026-10-25T17:00:00.000Z", "unexpected": "dropped",
    }

    def fake_get(url, params, headers, timeout):
        calls.append({"url": url, "params": params})
        return _Response([row])

    monkeypatch.setattr(registry_api.requests, "get", fake_get)
    client = RegistryClient(base_url="https://registry.example.com", api_key="service-key")

    streams = client.get_streams(listing_ids=[789, 790], family="book", active=True)

    assert calls == [{
        "url": "https://registry.example.com/api/streams",
        "params": {"listingIds": "789,790", "family": "book", "active": "true", "limit": 1000, "offset": 0},
    }]
    assert streams[0].stream_id == 31
    assert streams[0].listing_id == 789
    assert streams[0].resolved_region == "us-east-1"


def test_get_sources_filters_by_code(monkeypatch):
    calls = []

    def fake_get(url, params, headers, timeout):
        calls.append({"url": url, "params": params})
        return _Response([{
            "source_id": 9, "code": "PANDASCORE", "kind": "VENDOR", "exchange_id": None, "home_region": "eu-west-1",
            "adapter": "PANDASCORE", "families": ["esports"], "active": True,
            "date_modified": "2026-10-25T17:00:00.000Z", "date_created": "2026-10-25T17:00:00.000Z",
        }])

    monkeypatch.setattr(registry_api.requests, "get", fake_get)
    client = RegistryClient(base_url="https://registry.example.com", api_key="service-key")

    [source] = client.get_sources(code="PANDASCORE")

    assert calls[0]["params"] == {"code": "PANDASCORE", "limit": 1000, "offset": 0}
    assert source.families == ["esports"]
    assert source.home_region == "eu-west-1"


def test_get_source_accounts_filters_by_source_and_label(monkeypatch):
    calls = []

    def fake_get(url, params, headers, timeout):
        calls.append({"url": url, "params": params})
        return _Response([{
            "account_id": 3, "source_id": 4, "source_code": "KALSHI", "label": "md-1",
            "secret_name": "gnome/market-data/kalshi/md-1", "connection_budget": None, "status": "ACTIVE",
            "note": None, "created_by": "ops@example.com",
            "date_modified": "2026-10-27T15:00:00.000Z", "date_created": "2026-10-27T15:00:00.000Z",
        }])

    monkeypatch.setattr(registry_api.requests, "get", fake_get)
    client = RegistryClient(base_url="https://registry.example.com", api_key="service-key")

    [account] = client.get_source_accounts(source_code="KALSHI", label="md-1")

    assert calls[0]["url"].endswith("/api/sources/accounts")
    assert calls[0]["params"] == {"sourceCode": "KALSHI", "label": "md-1", "limit": 1000, "offset": 0}
    assert account.secret_name == "gnome/market-data/kalshi/md-1"
    assert account.connection_budget is None


def _relationship_row(**extra):
    return {"relationship_id": 1, "security_id_a": 10, "security_id_b": 11, "relationship_type": "IMPLIES",
            "method": "derived", "date_created": "x", **extra}


def test_contract_relationships_parse_with_or_without_confidence_and_flags(monkeypatch):
    pages = [[_relationship_row(confidence=0.9)],
             [_relationship_row(flags=["RESOLUTION_SOURCE_DIFFERS"], evidence={"rule": "threshold"}, derivation_version="d1")]]
    monkeypatch.setattr(registry_api.requests, "get", lambda url, params, headers, timeout: _Response(pages.pop(0)))
    client = RegistryClient(base_url="https://registry.example.com", api_key="service-key")

    old, = client.get_contract_relationships()
    new, = client.get_contract_relationships()

    assert old.flags == [] and old.evidence is None
    assert new.flags == ["RESOLUTION_SOURCE_DIFFERS"] and new.derivation_version == "d1"


def test_proposition_reads_send_id_lists_as_one_value(monkeypatch):
    calls = []

    def fake_get(url, params, headers, timeout):
        calls.append((url, params))
        if url.endswith("/proposition-subjects"):
            return _Response([{"subject_id": 3, "key": "btc-usd", "description": "", "date_created": "x",
                               "date_modified": "x", "proposition_count": 2}])
        return _Response([{"security_id": 10, "subject_id": 3, "extracted_subject_id": 3, "fields": {"kind": "THRESHOLD"},
                           "method": "VENUE", "extractor_version": "v1", "source_hash": "h", "date_created": "x",
                           "date_modified": "x", "symbol": "ignored"}])

    monkeypatch.setattr(registry_api.requests, "get", fake_get)
    client = RegistryClient(base_url="https://registry.example.com", api_key="service-key")

    subject, = client.get_proposition_subjects(subject_ids=[3, 4], dirty=True)
    proposition, = client.get_contract_propositions(security_ids=[10, 11])

    assert calls[0][1] == {"subjectIds": "3,4", "dirty": "true", "limit": 1000, "offset": 0}
    assert calls[1][1] == {"securityIds": "10,11", "limit": 1000, "offset": 0}
    assert subject.proposition_count == 2 and subject.canonical_subject_id is None
    assert proposition.fields == {"kind": "THRESHOLD"}


def test_reconcile_posts_camel_case_relationships(monkeypatch):
    posted = []

    def fake_post(url, json, headers, timeout):
        posted.append((url, json))
        return _Response({"inserted": 1})

    monkeypatch.setattr(registry_api.requests, "post", fake_post)
    client = RegistryClient(base_url="https://registry.example.com", api_key="service-key")

    client.reconcile_contract_relationships(
        security_ids=[10, 11], method="derived", derivation_version="d1", subject_id=3,
        relationships=[{"security_id_a": 10, "security_id_b": 11, "relationship_type": "IMPLIES", "flags": []}],
    )

    assert posted == [("https://registry.example.com/api/contract-relationships/reconcile", {
        "securityIds": [10, 11], "method": "derived", "derivationVersion": "d1", "subjectId": 3,
        "relationships": [{"securityIdA": 10, "securityIdB": 11, "relationshipType": "IMPLIES", "flags": []}],
    })]
