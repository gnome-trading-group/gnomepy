import os

class Config:
    REGISTRY_API_URL: str

class DevConfig(Config):
    REGISTRY_API_URL: str = 'https://i3116oczxe.execute-api.us-east-1.amazonaws.com/api'

class StagingConfig(Config):
    ...

class ProdConfig(Config):
    REGISTRY_API_URL: str = 'https://n5dxpwnij0.execute-api.us-east-1.amazonaws.com/api'

_ENV = os.getenv("ENVIRONMENT", "prod").lower()

_CONFIG_MAP = {
    "dev": DevConfig,
    "staging": StagingConfig,
    "prod": ProdConfig
}

config = _CONFIG_MAP.get(_ENV, DevConfig)
