import os

class Config:
    REGISTRY_API_HOST: str

class DevConfig(Config):
    REGISTRY_API_HOST: str = 'i3116oczxe.execute-api.us-east-1.amazonaws.com'

class StagingConfig(Config):
    ...

class ProdConfig(Config):
    REGISTRY_API_HOST: str = 'n5dxpwnij0.execute-api.us-east-1.amazonaws.com'

_STAGE = os.getenv("STAGE", "prod").lower()

_CONFIG_MAP = {
    "dev": DevConfig,
    "staging": StagingConfig,
    "prod": ProdConfig
}

config = _CONFIG_MAP.get(_STAGE, ProdConfig)
