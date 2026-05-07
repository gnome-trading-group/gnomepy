"""HTTP client for the gnome-controller backtest API (API Gateway + Cognito auth)."""
from __future__ import annotations

import os

import requests

from gnomepy.auth import get_id_token
from gnomepy.config import config


def _api_base_url() -> str:
    return os.environ.get("GNOME_CONTROLLER_API_URL", config.CONTROLLER_API_URL).rstrip("/")


def _headers() -> dict[str, str]:
    return {
        "Authorization": get_id_token(),
        "Content-Type": "application/json",
    }


def _request(method: str, path: str, **kwargs) -> dict:
    url = f"{_api_base_url()}{path}"
    resp = requests.request(method, url, headers=_headers(), **kwargs)
    if resp.status_code >= 400:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise RuntimeError(f"API error {resp.status_code}: {detail}")
    return resp.json()


def submit_backtest(config_yaml: str, research_commit: str = "main") -> dict:
    return _request("POST", "/backtests", json={
        "config": config_yaml,
        "research_commit": research_commit,
    })


def get_backtest(run_id: str) -> dict:
    return _request("GET", f"/backtests/{run_id}")


def list_backtests(status: str | None = None, limit: int = 20) -> dict:
    params: dict = {"limit": str(limit)}
    if status:
        params["status"] = status
    return _request("GET", "/backtests", params=params)


def cancel_backtest(run_id: str) -> dict:
    return _request("DELETE", f"/backtests/{run_id}")
