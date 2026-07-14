__all__ = ["RegistryClient"]

import dataclasses
import logging
import os
from typing import Optional

import requests

from gnomepy.config import config
from gnomepy.registry.types import (
    ContractRelationship,
    Currency,
    Event,
    EventContract,
    Exchange,
    ExchangeEvent,
    Listing,
    ListingSpec,
    Security,
)

logger = logging.getLogger(__name__)


def _to_camel_case(snake_str: str) -> str:
    camel = "".join(x.capitalize() for x in snake_str.lower().split("_"))
    return snake_str[0].lower() + camel[1:]


def _parse_kwarg_params(items) -> dict:
    params = dict(items)
    return {_to_camel_case(k): v for k, v in params.items() if v is not None and k != "self"}


class RegistryClient:

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_environ_key: str = "GNOME_REGISTRY_API_KEY",
        page_size: int = 1000,
    ):
        if base_url:
            self.base_url = base_url.rstrip("/") + "/api"
        else:
            self.base_url = f"https://{config.REGISTRY_API_HOST}/api"

        if api_key is None:
            api_key = os.environ.get(api_key_environ_key)

        if api_key is None or not isinstance(api_key, str) or api_key.isspace():
            raise ValueError(f"Invalid API key: {api_key}")
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
    ) -> list[Event]:
        return self._get("/events", _parse_kwarg_params(locals()), Event)

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

    def get_exchange_events(self, *, exchange_id: Optional[int] = None, event_id: Optional[int] = None) -> list[ExchangeEvent]:
        return self._get("/exchange-events", _parse_kwarg_params(locals()), ExchangeEvent)

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

    def bulk_create_events(self, items: list[dict]) -> list[dict]:
        return self._post_bulk("/events", items)

    def bulk_create_event_contracts(self, items: list[dict]) -> list[dict]:
        return self._post_bulk("/event-contracts", items)

    def bulk_create_contract_relationships(self, items: list[dict]) -> list[dict]:
        return self._post_bulk("/contract-relationships", items)

    def bulk_create_exchange_events(self, items: list[dict]) -> list[dict]:
        return self._post_bulk("/exchange-events", items)

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

    def _get(self, path: str, params: dict, output_type) -> list:
        all_items = []
        offset = 0
        page_num = 0
        logger.debug("GET %s params=%s page_size=%d", path, params, self._page_size)
        while True:
            page_params = {**params, "limit": self._page_size, "offset": offset}
            res = requests.get(self.base_url + path, params=page_params, headers={"x-api-key": self.api_key})
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
            )
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
            )
            res.raise_for_status()
            results.extend(res.json())
        return results

    def _post(self, path: str, body: dict) -> dict:
        res = requests.post(
            self.base_url + path,
            json=body,
            headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
        )
        res.raise_for_status()
        return res.json()

    def _patch(self, path: str, params: dict, body: dict) -> dict:
        res = requests.patch(
            self.base_url + path,
            params=params,
            json=body,
            headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
        )
        res.raise_for_status()
        return res.json()
