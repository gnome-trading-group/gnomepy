import os

class Config:
    REGISTRY_API_URL = 'https://i3116oczxe.execute-api.us-east-1.amazonaws.com/api'

class DevConfig(Config):
    ...

class StagingConfig(Config):
    ...

class ProdConfig(Config):
    ...

_ENV = os.getenv("ENVIRONMENT", "prod").lower()

_CONFIG_MAP = {
    "dev": DevConfig,
    "staging": StagingConfig,
    "prod": ProdConfig
}

config = _CONFIG_MAP.get(_ENV, DevConfig)
