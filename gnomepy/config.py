import functools
import os

import boto3
from botocore.exceptions import BotoCoreError, ClientError

# Secrets Manager mirrors of the APIs' keys (gnome-shared-cdk ApiKeySecret), readable with AWS credentials alone: a
# Batch job or a laptop with a profile needs no key in its environment.
REGISTRY_API_KEY_SECRET = "gnome/api-keys/registry"
MARKET_DATA_API_KEY_SECRET = "gnome/api-keys/market-data"


def resolve_registry_api_key() -> str:
    if key := os.environ.get("GNOME_REGISTRY_API_KEY"):
        return key
    key_id = os.environ.get("REGISTRY_API_KEY_ID", "")
    if key_id:
        return _fetch_api_key(key_id)
    return _read_secret(REGISTRY_API_KEY_SECRET)


def resolve_market_data_api_key() -> str:
    """The market-data API's key: MARKET_DATA_API_KEY, else its secret; empty when neither is readable."""
    return os.environ.get("MARKET_DATA_API_KEY") or _read_secret(MARKET_DATA_API_KEY_SECRET)


@functools.cache
def _read_secret(name: str) -> str:
    try:
        return boto3.client("secretsmanager", region_name="us-east-1").get_secret_value(SecretId=name)["SecretString"]
    except (BotoCoreError, ClientError):
        return ""


# GetApiKey is an account-wide, tightly throttled control-plane call, and a warm Lambda builds a RegistryClient on
# every invocation.
@functools.cache
def _fetch_api_key(key_id: str) -> str:
    return boto3.client("apigateway", region_name="us-east-1").get_api_key(apiKey=key_id, includeValue=True)["value"]


ALERTS_API_URL_PARAMETER = "/gnome/alerting/api-url"


def resolve_alerts_api_url() -> str:
    """Where alerts are reported: ALERTS_API_URL (a session has it), else the alerting stack's SSM parameter."""
    if url := os.environ.get("ALERTS_API_URL"):
        return url
    response = boto3.client("ssm", region_name="us-east-1").get_parameter(Name=ALERTS_API_URL_PARAMETER)
    return response["Parameter"]["Value"]


class Config:
    REGISTRY_API_HOST: str
    MARKET_DATA_API_HOST: str
    CONTROLLER_API_URL: str
    CONTROLLER_UI_URL: str
    COGNITO_DOMAIN: str
    COGNITO_CLIENT_ID: str

class DevConfig(Config):
    REGISTRY_API_HOST: str = 'i3116oczxe.execute-api.us-east-1.amazonaws.com'
    MARKET_DATA_API_HOST: str = '9imu88ibol.execute-api.us-east-1.amazonaws.com'
    CONTROLLER_API_URL: str = 'https://uwfjao7rtf.execute-api.us-east-1.amazonaws.com/api'
    CONTROLLER_UI_URL: str = 'https://controller-dev.gnometrading.group'
    COGNITO_DOMAIN: str = 'gnome-controller-dev.auth.us-east-1.amazoncognito.com'
    COGNITO_CLIENT_ID: str = '6mtnlmpgc1ojs612r304p7n33v'

class StagingConfig(Config):
    ...

class ProdConfig(Config):
    REGISTRY_API_HOST: str = 'n5dxpwnij0.execute-api.us-east-1.amazonaws.com'
    MARKET_DATA_API_HOST: str = 'icj8z6151a.execute-api.us-east-1.amazonaws.com'
    CONTROLLER_API_URL: str = 'https://5yoy6t6fba.execute-api.us-east-1.amazonaws.com/api'
    CONTROLLER_UI_URL: str = 'https://controller.gnometrading.group'
    COGNITO_DOMAIN: str = 'gnome-controller-prod.auth.us-east-1.amazoncognito.com'
    COGNITO_CLIENT_ID: str = '1o8bmieukcgs674hib0cmfj3gi'

_STAGE = os.getenv("STAGE", "prod").lower()

_CONFIG_MAP = {
    "dev": DevConfig,
    "staging": StagingConfig,
    "prod": ProdConfig
}

config = _CONFIG_MAP.get(_STAGE, ProdConfig)
