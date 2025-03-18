__all__ = ["RegistryClient"]

import os
from typing import Optional

import requests

from gnomepy.config import config

def _to_camel_case(snake_str):
    camel_string = "".join(x.capitalize() for x in snake_str.lower().split("_"))
    return snake_str[0].lower() + camel_string[1:]

def _parse_kwarg_params(items):
    params = dict(items)
    return {_to_camel_case(k): v for k, v in params.items() if v is not None and k != 'self'}


class RegistryClient:

    def __init__(
            self,
            api_key: Optional[str] = None,
            api_key_environ_key: str = "GNOME_REGISTRY_API_KEY",
    ):
        self.base_url = config.REGISTRY_API_URL

        if api_key is None:
            api_key = os.environ.get(api_key_environ_key)

        if api_key is None or not isinstance(api_key, str) or api_key.isspace():
            raise ValueError(f"Invalid API key: {api_key}")
        self.api_key = api_key

    def get_security(
            self,
            *,
            security_id: Optional[int] = None,
            symbol: Optional[str] = None,
    ):
        return self._get("/securities", _parse_kwarg_params(locals()))

    def get_exchange(
            self,
            *,
            exchange_id: Optional[int] = None,
            exchange_name: Optional[str] = None,
    ):
        return self._get("/exchanges", _parse_kwarg_params(locals()))

    def get_listing(
            self,
            *,
            listing_id: Optional[int] = None,
            exchange_id: Optional[int] = None,
            security_id: Optional[int] = None,
            exchange_security_id: Optional[str] = None,
            exchange_security_symbol: Optional[str] = None,
    ):
        return self._get("/listings", _parse_kwarg_params(locals()))

    def _get(self, path, params):
        res = requests.get(self.base_url + path, params=params, headers={
            'x-api-key': self.api_key,
        })
        res.raise_for_status()
        return res.json()
