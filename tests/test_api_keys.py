"""Where gnomepy finds the APIs' keys: the environment first, then their Secrets Manager mirrors."""
from __future__ import annotations

import pytest
from botocore.exceptions import NoCredentialsError

import gnomepy.config as gnome_config


class _Secrets:
    def __init__(self, values: dict[str, str] | None = None, error: Exception | None = None):
        self.values = values or {}
        self.error = error
        self.reads: list[str] = []

    def get_secret_value(self, SecretId):
        self.reads.append(SecretId)
        if self.error:
            raise self.error
        return {"SecretString": self.values[SecretId]}


@pytest.fixture
def secrets(monkeypatch):
    def install(store: _Secrets) -> _Secrets:
        monkeypatch.setattr(gnome_config.boto3, "client", lambda service, **kwargs: store)
        return store

    for name in ("GNOME_REGISTRY_API_KEY", "REGISTRY_API_KEY_ID", "MARKET_DATA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    gnome_config._read_secret.cache_clear()
    yield install
    gnome_config._read_secret.cache_clear()


def test_the_environment_wins_over_the_secret(secrets, monkeypatch):
    store = secrets(_Secrets({gnome_config.MARKET_DATA_API_KEY_SECRET: "from-secret"}))
    monkeypatch.setenv("MARKET_DATA_API_KEY", "from-env")
    monkeypatch.setenv("GNOME_REGISTRY_API_KEY", "registry-env")

    assert gnome_config.resolve_market_data_api_key() == "from-env"
    assert gnome_config.resolve_registry_api_key() == "registry-env"
    assert store.reads == []


def test_without_the_environment_each_key_comes_from_its_secret(secrets):
    store = secrets(_Secrets({
        gnome_config.MARKET_DATA_API_KEY_SECRET: "md-secret",
        gnome_config.REGISTRY_API_KEY_SECRET: "registry-secret",
    }))

    assert gnome_config.resolve_market_data_api_key() == "md-secret"
    assert gnome_config.resolve_registry_api_key() == "registry-secret"
    # Read once per process.
    gnome_config.resolve_market_data_api_key()
    assert store.reads.count(gnome_config.MARKET_DATA_API_KEY_SECRET) == 1


def test_no_readable_secret_is_an_empty_key(secrets):
    secrets(_Secrets(error=NoCredentialsError()))

    assert gnome_config.resolve_market_data_api_key() == ""
    assert gnome_config.resolve_registry_api_key() == ""
