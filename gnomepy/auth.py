"""OAuth2 PKCE authentication for gnomepy CLI."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

from gnomepy.config import config

_CREDENTIALS_DIR = Path.home() / ".gnomepy"
_CREDENTIALS_FILE = _CREDENTIALS_DIR / "credentials.json"
_CALLBACK_PORT = 9876
_REDIRECT_URI = f"http://localhost:{_CALLBACK_PORT}/callback"
_EXPIRY_BUFFER_SECS = 300  # refresh 5 minutes before actual expiry


def _cognito_domain() -> str:
    return os.environ.get("GNOME_COGNITO_DOMAIN", config.COGNITO_DOMAIN)


def _client_id() -> str:
    return os.environ.get("GNOME_COGNITO_CLIENT_ID", config.COGNITO_CLIENT_ID)


def _generate_pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(96)[:128]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _save_credentials(tokens: dict) -> None:
    _CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    _CREDENTIALS_FILE.write_text(json.dumps(tokens, indent=2))
    os.chmod(_CREDENTIALS_FILE, 0o600)


def _load_credentials() -> dict | None:
    if not _CREDENTIALS_FILE.exists():
        return None
    return json.loads(_CREDENTIALS_FILE.read_text())


def _token_expired(credentials: dict) -> bool:
    return time.time() >= (credentials.get("expires_at", 0) - _EXPIRY_BUFFER_SECS)


def _post_token(data: dict) -> dict:
    body = urlencode(data).encode("ascii")
    req = Request(
        f"https://{_cognito_domain()}/oauth2/token",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urlopen(req) as resp:
        return json.loads(resp.read())


def _exchange_code(code: str, code_verifier: str) -> dict:
    tokens = _post_token({
        "grant_type": "authorization_code",
        "client_id": _client_id(),
        "code": code,
        "redirect_uri": _REDIRECT_URI,
        "code_verifier": code_verifier,
    })
    tokens["expires_at"] = time.time() + tokens.get("expires_in", 3600)
    return tokens


def _refresh(refresh_token: str) -> dict:
    tokens = _post_token({
        "grant_type": "refresh_token",
        "client_id": _client_id(),
        "refresh_token": refresh_token,
    })
    tokens["refresh_token"] = refresh_token  # Cognito does not return it on refresh
    tokens["expires_at"] = time.time() + tokens.get("expires_in", 3600)
    return tokens


def login() -> dict:
    """Run PKCE login flow: open browser, wait for callback, cache tokens."""
    code_verifier, code_challenge = _generate_pkce()
    state = secrets.token_urlsafe(32)

    authorize_url = f"https://{_cognito_domain()}/oauth2/authorize?" + urlencode({
        "response_type": "code",
        "client_id": _client_id(),
        "redirect_uri": _REDIRECT_URI,
        "scope": "openid email profile",
        "code_challenge_method": "S256",
        "code_challenge": code_challenge,
        "state": state,
    })

    received: dict = {}

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            qs = parse_qs(urlparse(self.path).query)
            received["code"] = qs.get("code", [None])[0]
            received["state"] = qs.get("state", [None])[0]
            received["error"] = qs.get("error", [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h2>Login successful!</h2>"
                b"<p>You can close this tab.</p></body></html>"
            )

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", _CALLBACK_PORT), _Handler)
    server.timeout = 120

    webbrowser.open(authorize_url)
    server.handle_request()
    server.server_close()

    if received.get("error"):
        raise RuntimeError(f"OAuth error: {received['error']}")
    if received.get("state") != state:
        raise RuntimeError("OAuth state mismatch — possible CSRF")
    if not received.get("code"):
        raise RuntimeError("No authorization code received — login timed out or was cancelled")

    tokens = _exchange_code(received["code"], code_verifier)
    _save_credentials(tokens)
    return tokens


def get_id_token() -> str:
    """Return a valid id_token, auto-refreshing if expired."""
    credentials = _load_credentials()
    if credentials is None:
        raise RuntimeError("Not authenticated. Run `gnomepy login` first.")

    if not _token_expired(credentials):
        return credentials["id_token"]

    refresh_token = credentials.get("refresh_token")
    if not refresh_token:
        raise RuntimeError("Session expired. Run `gnomepy login` again.")

    try:
        credentials = _refresh(refresh_token)
        _save_credentials(credentials)
        return credentials["id_token"]
    except HTTPError as e:
        raise RuntimeError(f"Token refresh failed ({e.code}). Run `gnomepy login` again.") from e


def logout() -> None:
    """Remove cached credentials."""
    if _CREDENTIALS_FILE.exists():
        _CREDENTIALS_FILE.unlink()
