__all__ = ["RegistryClient"]

import dataclasses
import logging
import os
from typing import Optional

import requests

from gnomepy.auth import get_id_token
from gnomepy.config import config, resolve_alerts_api_url, resolve_registry_api_key
from gnomepy.registry.types import (
    ContractProposition,
    ContractRelationship,
    ContractSource,
    Currency,
    Event,
    EventContract,
    Exchange,
    Listing,
    ListingSpec,
    PropositionSubject,
    Security,
    Source,
    SourceAccount,
    Stream,
    Strategy,
    StrategySession,
)

logger = logging.getLogger(__name__)

# API Gateway gives up on an integration after 29s, so a request still open well past that is a dropped connection.
REQUEST_TIMEOUT_SECONDS = 60


def _to_camel_case(snake_str: str) -> str:
    camel = "".join(x.capitalize() for x in snake_str.lower().split("_"))
    return snake_str[0].lower() + camel[1:]


def _parse_kwarg_params(items) -> dict:
    params = dict(items)
    # The registry compares query flags with the string "true"; requests would send Python's "True".
    return {
        _to_camel_case(k): str(v).lower() if isinstance(v, bool) else v
        for k, v in params.items()
        if v is not None and k != "self"
    }


class RegistryClient:

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        page_size: int = 1000,
    ):
        if base_url:
            self.base_url = base_url.rstrip("/") + "/api"
        else:
            self.base_url = f"https://{config.REGISTRY_API_HOST}/api"

        if api_key is None:
            api_key = resolve_registry_api_key()

        if not api_key or not isinstance(api_key, str) or api_key.isspace():
            raise ValueError(
                "No registry API key found. Set GNOME_REGISTRY_API_KEY or "
                "REGISTRY_API_KEY_ID (requires AWS credentials)."
            )
        self.api_key = api_key
        self._page_size = page_size

    def get_exchange(self, *, exchange_id: Optional[int] = None, exchange_name: Optional[str] = None) -> list[Exchange]:
        return self._get("/exchanges", _parse_kwarg_params(locals()), Exchange)

    def get_currency(self) -> list[Currency]:
        return self._get("/currencies", {}, Currency)

    def get_security(self, *, security_id: Optional[int] = None, symbol: Optional[str] = None) -> list[Security]:
        return self._get("/securities", _parse_kwarg_params(locals()), Security)

    def get_listing(
        self,
        *,
        listing_id: Optional[int] = None,
        exchange_id: Optional[int] = None,
        security_id: Optional[int] = None,
        exchange_security_id: Optional[str] = None,
        exchange_security_symbol: Optional[str] = None,
    ) -> list[Listing]:
        return self._get("/listings", _parse_kwarg_params(locals()), Listing)

    def get_listing_spec(self, *, listing_id: Optional[int] = None) -> list[ListingSpec]:
        return self._get("/listing-specs", _parse_kwarg_params(locals()), ListingSpec)

    def get_event(
        self,
        *,
        event_id: Optional[int] = None,
        category: Optional[str] = None,
        resolved: Optional[bool] = None,
        exchange_id: Optional[int] = None,
        native_event_id: Optional[str] = None,
        exchange_ids: Optional[list[int]] = None,
        categories: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        expiry_from: Optional[str] = None,
        expiry_to: Optional[str] = None,
        min_volume_24h: Optional[float] = None,
        sort_by: Optional[str] = None,
        sort_order: Optional[str] = None,
    ) -> list[Event]:
        """Events matching every given filter. List filters match any of their values; ``sort_by="volume_24h"`` puts
        events with no volume reading last."""
        params = _parse_kwarg_params(locals())
        # The registry reads list filters as one comma-separated value; requests would repeat the key per item.
        for key in ("exchangeIds", "categories", "tags"):
            if key in params:
                params[key] = ",".join(str(v) for v in params[key])
        return self._get("/events", params, Event)

    def get_event_contracts(
        self,
        *,
        event_contract_id: Optional[int] = None,
        event_id: Optional[int] = None,
        security_id: Optional[int] = None,
    ) -> list[EventContract]:
        return self._get("/event-contracts", _parse_kwarg_params(locals()), EventContract)

    def get_contract_relationships(
        self,
        *,
        relationship_id: Optional[int] = None,
        security_id: Optional[int] = None,
        reviewed: Optional[bool] = None,
        method: Optional[str] = None,
        relationship_type: Optional[str] = None,
    ) -> list[ContractRelationship]:
        return self._get("/contract-relationships", _parse_kwarg_params(locals()), ContractRelationship)

    def get_connected_relationships(self, *, security_ids: list[int]) -> list[ContractRelationship]:
        """Every relationship in the connected sets of these securities (reached through relationships and
        complements), with each side's complement: the rows a consumer needs to rebuild the complete relation, which
        the registry stores only the fewest rows of (see ``relations.close``)."""
        params = {"securityIds": ",".join(map(str, security_ids)), "connected": "true"}
        return self._get("/contract-relationships", params, ContractRelationship)

    def get_proposition_subjects(
        self,
        *,
        subject_ids: Optional[list[int]] = None,
        key: Optional[str] = None,
        search: Optional[str] = None,
        canonical: Optional[bool] = None,
        dirty: Optional[bool] = None,
    ) -> list[PropositionSubject]:
        params = _parse_kwarg_params(locals())
        if subject_ids is not None:
            params["subjectIds"] = ",".join(map(str, subject_ids))
        return self._get("/proposition-subjects", params, PropositionSubject)

    def get_contract_propositions(
        self,
        *,
        security_ids: Optional[list[int]] = None,
        subject_id: Optional[int] = None,
    ) -> list[ContractProposition]:
        params = _parse_kwarg_params(locals())
        if security_ids is not None:
            params["securityIds"] = ",".join(map(str, security_ids))
        return self._get("/contract-propositions", params, ContractProposition)

    def get_contract_sources(
        self,
        *,
        security_ids: Optional[list[int]] = None,
        exchange_id: Optional[int] = None,
    ) -> list[ContractSource]:
        params = _parse_kwarg_params(locals())
        if security_ids is not None:
            params["securityIds"] = ",".join(map(str, security_ids))
        return self._get("/contract-sources", params, ContractSource)

    def get_sources(
        self,
        *,
        source_id: Optional[int] = None,
        code: Optional[str] = None,
        kind: Optional[str] = None,
        exchange_id: Optional[int] = None,
        active: Optional[bool] = None,
    ) -> list[Source]:
        return self._get("/sources", _parse_kwarg_params(locals()), Source)

    def get_source_accounts(
        self,
        *,
        source_id: Optional[int] = None,
        source_code: Optional[str] = None,
        label: Optional[str] = None,
    ) -> list[SourceAccount]:
        return self._get("/sources/accounts", _parse_kwarg_params(locals()), SourceAccount)

    def get_streams(
        self,
        *,
        stream_id: Optional[int] = None,
        stream_ids: Optional[list[int]] = None,
        listing_id: Optional[int] = None,
        listing_ids: Optional[list[int]] = None,
        source_id: Optional[int] = None,
        family: Optional[str] = None,
        event_id: Optional[int] = None,
        active: Optional[bool] = None,
    ) -> list[Stream]:
        params = _parse_kwarg_params(locals())
        for key in ("streamIds", "listingIds"):
            if key in params:
                params[key] = ",".join(str(i) for i in params[key])
        return self._get("/streams", params, Stream)

    def create_vendor_source(self, code: str, home_region: str, adapter: str, families: list[str]) -> dict:
        """Registers a data vendor. Venue sources are created with their exchange. Needs `gnomepy login`."""
        body = {"code": code, "homeRegion": home_region, "adapter": adapter, "families": families}
        return self._post_as_operator("/sources", body)

    def update_source(
        self,
        source_id: int,
        *,
        home_region: Optional[str] = None,
        adapter: Optional[str] = None,
        families: Optional[list[str]] = None,
        active: Optional[bool] = None,
    ) -> dict:
        """Edits a source; a venue's region is its exchange's and can't be set here. Needs `gnomepy login`."""
        body = {
            _to_camel_case(k): v
            for k, v in {"home_region": home_region, "adapter": adapter, "families": families, "active": active}.items()
            if v is not None
        }
        return self._patch_as_operator("/sources", {"sourceId": str(source_id)}, body)

    def create_currency(self, **kwargs) -> dict:
        return self._post("/currencies", {_to_camel_case(k): v for k, v in kwargs.items()})

    def create_security(self, **kwargs) -> dict:
        return self._post("/securities", {_to_camel_case(k): v for k, v in kwargs.items()})

    def create_listing(self, **kwargs) -> dict:
        return self._post("/listings", {_to_camel_case(k): v for k, v in kwargs.items()})

    def create_listing_spec(self, **kwargs) -> dict:
        return self._post("/listing-specs", {_to_camel_case(k): v for k, v in kwargs.items()})

    def create_event(self, **kwargs) -> dict:
        return self._post("/events", {_to_camel_case(k): v for k, v in kwargs.items()})

    def create_event_contract(self, **kwargs) -> dict:
        return self._post("/event-contracts", {_to_camel_case(k): v for k, v in kwargs.items()})

    def create_contract_relationship(self, **kwargs) -> dict:
        return self._post("/contract-relationships", {_to_camel_case(k): v for k, v in kwargs.items()})

    def bulk_create_securities(self, items: list[dict]) -> list[dict]:
        return self._post_bulk("/securities", items)

    def bulk_create_currencies(self, items: list[dict]) -> list[dict]:
        return self._post_bulk("/currencies", items)

    def bulk_create_listings(self, items: list[dict]) -> list[dict]:
        return self._post_bulk("/listings", items)

    def bulk_create_listing_specs(self, items: list[dict]) -> list[dict]:
        return self._post_bulk("/listing-specs", items)

    def bulk_create_fee_schedules(self, items: list[dict]) -> list[dict]:
        return self._post_bulk("/fees", items)

    def get_fee_timelines(
        self,
        listing_ids: list[int],
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        known_as_of: Optional[str] = None,
    ) -> dict[int, list[dict]]:
        """Each listing's fee schedules over time, as the registry resolves them; times are ISO-8601."""
        params = {"listingIds": ",".join(str(i) for i in listing_ids), "from": start, "to": end, "knownAsOf": known_as_of}
        res = requests.get(
            self.base_url + "/fees", params={k: v for k, v in params.items() if v is not None},
            headers={"x-api-key": self.api_key}, timeout=REQUEST_TIMEOUT_SECONDS,
        )
        res.raise_for_status()
        return {item["listingId"]: item["timeline"] for item in res.json()}

    def get_simulation_parameters(
        self, *, exchange_id: Optional[int] = None, known_as_of: Optional[str] = None, history: bool = False,
    ) -> list[dict]:
        """Each exchange's simulation parameters in force at ``known_as_of`` (default now), as stored rows; with
        ``history``, every recorded version, newest first."""
        params = {"exchangeId": exchange_id, "knownAsOf": known_as_of, "history": "true" if history else None}
        res = requests.get(
            self.base_url + "/simulation-parameters", params={k: v for k, v in params.items() if v is not None},
            headers={"x-api-key": self.api_key}, timeout=REQUEST_TIMEOUT_SECONDS,
        )
        res.raise_for_status()
        return res.json()

    def bulk_create_events(self, items: list[dict]) -> list[dict]:
        return self._post_bulk("/events", items)

    def bulk_create_event_contracts(self, items: list[dict]) -> list[dict]:
        return self._post_bulk("/event-contracts", items)

    def bulk_create_contract_relationships(self, items: list[dict]) -> list[dict]:
        return self._post_bulk("/contract-relationships", items)

    def create_proposition_subjects(self, items: list[dict]) -> list[dict]:
        """Creates subjects by ``key`` (with an optional ``description``), or returns the ones that exist."""
        return self._post_bulk("/proposition-subjects", items)

    def upsert_contract_sources(self, items: list[dict]) -> list[dict]:
        """Writes each security's venue source (``security_id``, ``exchange_id``, ``source``, ``source_hash``)."""
        return self._post_bulk("/contract-sources", items)

    def upsert_contract_propositions(self, items: list[dict]) -> list[dict]:
        """Writes one proposition per security; returns ``security_id``, resolved ``subject_id`` and ``changed`` each."""
        return self._post_bulk("/contract-propositions", items)

    def reconcile_contract_relationships(
        self,
        *,
        security_ids: list[int],
        relationships: list[dict],
        method: str,
        derivation_version: str,
        subject_id: Optional[int] = None,
        derived_at: Optional[str] = None,
    ) -> dict:
        """Replaces the derived relationships touching ``security_ids`` with ``relationships`` in one transaction.

        Manual rows are kept, and a relationship on a pair a person already related is skipped. With ``subject_id``,
        also records the subject as derived as of ``derived_at`` (when its propositions were read; default now).
        """
        body = {
            "securityIds": security_ids,
            "relationships": [{_to_camel_case(k): v for k, v in r.items()} for r in relationships],
            "method": method,
            "derivationVersion": derivation_version,
        }
        if subject_id is not None:
            body["subjectId"] = subject_id
        if derived_at is not None:
            body["derivedAt"] = derived_at
        return self._post("/contract-relationships/reconcile", body)

    def patch_event_contract(self, event_contract_id: int, **kwargs) -> dict:
        body = {_to_camel_case(k): v for k, v in kwargs.items()}
        return self._patch("/event-contracts", {"eventContractId": str(event_contract_id)}, body)

    def patch_event(self, event_id: int, **kwargs) -> dict:
        body = {_to_camel_case(k): v for k, v in kwargs.items()}
        return self._patch("/events", {"eventId": str(event_id)}, body)

    def patch_security(self, security_id: int, **kwargs) -> dict:
        body = {_to_camel_case(k): v for k, v in kwargs.items()}
        return self._patch("/securities", {"securityId": str(security_id)}, body)

    def patch_listing(self, listing_id: int, **kwargs) -> dict:
        body = {_to_camel_case(k): v for k, v in kwargs.items()}
        return self._patch("/listings", {"listingId": str(listing_id)}, body)

    def bulk_patch_events(self, items: list[dict]) -> list[dict]:
        return self._patch_bulk("/events", items)

    def bulk_patch_securities(self, items: list[dict]) -> list[dict]:
        return self._patch_bulk("/securities", items)

    def bulk_patch_listings(self, items: list[dict]) -> list[dict]:
        return self._patch_bulk("/listings", items)

    def bulk_patch_event_contracts(self, items: list[dict]) -> list[dict]:
        return self._patch_bulk("/event-contracts", items)

    def _get(self, path: str, params: dict, output_type) -> list:
        all_items = []
        offset = 0
        page_num = 0
        logger.debug("GET %s params=%s page_size=%d", path, params, self._page_size)
        while True:
            page_params = {**params, "limit": self._page_size, "offset": offset}
            res = requests.get(
                self.base_url + path, params=page_params, headers={"x-api-key": self.api_key},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            res.raise_for_status()
            page = res.json()
            logger.debug("GET %s page=%d offset=%d got=%d total=%d", path, page_num, offset, len(page), len(all_items) + len(page))
            if dataclasses.is_dataclass(output_type):
                known = {f.name for f in dataclasses.fields(output_type)}
                all_items.extend(output_type(**{k: v for k, v in item.items() if k in known}) for item in page)
            else:
                all_items.extend(output_type(**item) for item in page)
            if len(page) < self._page_size:
                break
            offset += self._page_size
            page_num += 1
        logger.debug("GET %s done total=%d pages=%d", path, len(all_items), page_num + 1)
        return all_items

    def _post_bulk(self, path: str, items: list[dict], batch_size: int = 500) -> list[dict]:
        n_batches = (len(items) + batch_size - 1) // batch_size
        logger.debug("POST bulk %s total=%d batch_size=%d n_batches=%d", path, len(items), batch_size, n_batches)
        results: list[dict] = []
        for i in range(0, len(items), batch_size):
            batch = [{_to_camel_case(k): v for k, v in item.items()} for item in items[i : i + batch_size]]
            logger.debug("POST bulk %s batch=%d/%d items=%d", path, i // batch_size + 1, n_batches, len(batch))
            res = requests.post(
                self.base_url + path,
                json=batch,
                headers={"x-api-key": self.api_key, "Content-Type": "application/json"},

                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if not res.ok:
                logger.error("POST %s failed (%d): %s", path, res.status_code, res.text)
            res.raise_for_status()
            results.extend(res.json())
        return results

    def _patch_bulk(self, path: str, items: list[dict], batch_size: int = 500) -> list[dict]:
        n_batches = (len(items) + batch_size - 1) // batch_size
        logger.debug("PATCH bulk %s total=%d batch_size=%d n_batches=%d", path, len(items), batch_size, n_batches)
        results: list[dict] = []
        for i in range(0, len(items), batch_size):
            batch = [{_to_camel_case(k): v for k, v in item.items()} for item in items[i : i + batch_size]]
            logger.debug("PATCH bulk %s batch=%d/%d items=%d", path, i // batch_size + 1, n_batches, len(batch))
            res = requests.patch(
                self.base_url + path,
                json=batch,
                headers={"x-api-key": self.api_key, "Content-Type": "application/json"},

                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if not res.ok:
                logger.error("PATCH %s failed (%d): %s", path, res.status_code, res.text)
            res.raise_for_status()
            results.extend(res.json())
        return results

    def _post(self, path: str, body: dict) -> dict:
        res = requests.post(
            self.base_url + path,
            json=body,
            headers={"x-api-key": self.api_key, "Content-Type": "application/json"},

            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        res.raise_for_status()
        return res.json()

    def _post_as_operator(self, path: str, body: dict) -> dict:
        # Routes that must name a person in the audit log admit only the `gnomepy login` ID token, never the API key.
        res = requests.post(
            self.base_url + path,
            json=body,
            headers={"Authorization": get_id_token(), "Content-Type": "application/json"},

            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        res.raise_for_status()
        return res.json()

    def _patch_as_operator(self, path: str, params: dict, body: dict) -> dict:
        res = requests.patch(
            self.base_url + path,
            params=params,
            json=body,
            headers={"Authorization": get_id_token(), "Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        res.raise_for_status()
        return res.json()

    def _patch(self, path: str, params: dict, body: dict) -> dict:
        res = requests.patch(
            self.base_url + path,
            params=params,
            json=body,
            headers={"x-api-key": self.api_key, "Content-Type": "application/json"},

            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        res.raise_for_status()
        return res.json()

    def _delete(self, path: str, body: dict) -> dict:
        res = requests.delete(
            self.base_url + path,
            json=body,
            headers={"x-api-key": self.api_key, "Content-Type": "application/json"},

            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        res.raise_for_status()
        return res.json()

    def get_strategies(
        self,
        *,
        strategy_id: int | None = None,
        name: str | None = None,
        archived: bool | None = None,
    ) -> list[Strategy]:
        return self._get("/strategies", _parse_kwarg_params(locals()), Strategy)

    def create_strategy(
        self,
        name: str,
        description: str | None = None,
        parameters: dict | None = None,
    ) -> dict:
        body: dict = {"name": name}
        if description is not None:
            body["description"] = description
        if parameters is not None:
            body["parameters"] = parameters
        return self._post("/strategies", body)

    def get_strategy_sessions(
        self,
        *,
        session_id: str | None = None,
        strategy_id: int | None = None,
        status: str | None = None,
    ) -> list[StrategySession]:
        return self._get("/strategy-sessions", _parse_kwarg_params(locals()), StrategySession)

    def create_strategy_session(
        self,
        session_id: str,
        strategy_id: int,
        mode: str,
        config: dict,
        research_commit: str | None = None,
        region: str | None = None,
        instance_type: str | None = None,
        orchestrator_version: str | None = None,
        gnomepy_version: str | None = None,
        availability_zone: str | None = None,
    ) -> dict:
        """Launch a session (``SessionConfig.to_session_config()`` builds ``config``).

        ``config`` keys the registry reads: ``listings`` (traded listing ids), ``streams`` (market-data stream ids the
        session reads without trading, any region; paper only; the registry leases them with each
        traded listing's book stream from the collector shards), ``strategy.type``/``strategy.class``,
        ``strategy.args.<name>``, ``simulation`` (paper), ``latency.profile``, ``accounts`` (live) and
        ``overrides.<orchestrator property>``. Other keys reach the instance as environment variables.
        """
        body = {
            "sessionId": session_id,
            "strategyId": strategy_id,
            "mode": mode,
            "config": config,
        }
        optional = {
            "researchCommit": research_commit,
            "region": region,
            "instanceType": instance_type,
            "orchestratorVersion": orchestrator_version,
            "gnomepyVersion": gnomepy_version,
            "availabilityZone": availability_zone,
        }
        body.update({key: value for key, value in optional.items() if value is not None})
        return self._post("/strategy-sessions/launch", body)

    def stop_strategy_session(self, session_id: str, reason: Optional[str] = None) -> dict:
        body = {"sessionId": session_id}
        if reason:
            body["reason"] = reason
        return self._post_as_operator("/strategy-sessions/stop", body)

    def send_alert(
        self,
        type: str,
        title: str,
        *,
        status: str = "FIRING",
        key: Optional[str] = None,
        severity: Optional[str] = None,
        body: Optional[str] = None,
        actor: Optional[str] = None,
        strategy_id: Optional[int] = None,
        session_id: Optional[str] = None,
        listing_id: Optional[int] = None,
        exchange_id: Optional[int] = None,
        account_id: Optional[int] = None,
        mode: Optional[str] = None,
        labels: Optional[dict[str, str]] = None,
        links: Optional[list[dict[str, str]]] = None,
        occurred_at: Optional[str] = None,
        alerts_url: Optional[str] = None,
    ) -> dict:
        """Report an alert (POST /alerts on the alerts API, with this client's API key).

        `type` is a catalog type (e.g. LAUNCH_FAILED) or CUSTOM, whose `severity` you choose (capped below paging).
        Repeats of the same `key` while it fires are one alert; send `status="RESOLVED"` with that key to close it.
        Without a key, one alert per type, session and strategy (and title, for CUSTOM).
        """
        payload = {
            "type": type, "title": title, "status": status, "key": key, "severity": severity, "body": body,
            "actor": actor, "strategyId": strategy_id, "sessionId": session_id, "listingId": listing_id,
            "exchangeId": exchange_id, "accountId": account_id, "mode": mode, "labels": labels, "links": links,
            "occurredAt": occurred_at,
        }
        url = (alerts_url or resolve_alerts_api_url()).rstrip("/") + "/alerts"
        res = requests.post(
            url,
            json={k: v for k, v in payload.items() if v is not None},
            headers={"x-api-key": self.api_key, "Content-Type": "application/json"},

            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        res.raise_for_status()
        return res.json()

    def stop_strategy_session_as_service(self, session_id: str, actor: str, reason: Optional[str] = None) -> dict:
        """Stop a session from an automated caller (no `gnomepy login`), e.g. the launcher.

        `actor` is recorded as who stopped it, on the session and in the kill-switch audit log, e.g.
        "launcher:cs2-prematch"; `reason`, if given, is shown on the session.
        """
        body = {"sessionId": session_id, "actor": actor}
        if reason:
            body["reason"] = reason
        return self._post("/strategy-sessions/stop", body)
