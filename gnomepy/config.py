import os

class Config:
    REGISTRY_API_HOST: str
    REGISTRY_API_KEY: str
    CONTROLLER_API_URL: str
    COGNITO_DOMAIN: str
    COGNITO_CLIENT_ID: str

class DevConfig(Config):
    REGISTRY_API_HOST: str = 'i3116oczxe.execute-api.us-east-1.amazonaws.com'
    REGISTRY_API_KEY: str = '9WPV7CfeqXa578yVYlxdG3kCPFzACr7YaMU0UVma'
    CONTROLLER_API_URL: str = 'https://uwfjao7rtf.execute-api.us-east-1.amazonaws.com/api'
    COGNITO_DOMAIN: str = 'gnome-controller-dev.auth.us-east-1.amazoncognito.com'
    COGNITO_CLIENT_ID: str = '6mtnlmpgc1ojs612r304p7n33v'

class StagingConfig(Config):
    ...

class ProdConfig(Config):
    REGISTRY_API_HOST: str = 'n5dxpwnij0.execute-api.us-east-1.amazonaws.com'
    REGISTRY_API_KEY: str = 'Smr7Jrmr8j93MMymhYdebaoRbV2T6TkX7phGPnrd'
    CONTROLLER_API_URL: str = 'https://5yoy6t6fba.execute-api.us-east-1.amazonaws.com/api'
    COGNITO_DOMAIN: str = 'gnome-controller-prod.auth.us-east-1.amazoncognito.com'
    COGNITO_CLIENT_ID: str = '1o8bmieukcgs674hib0cmfj3gi'

_STAGE = os.getenv("STAGE", "prod").lower()

_CONFIG_MAP = {
    "dev": DevConfig,
    "staging": StagingConfig,
    "prod": ProdConfig
}

config = _CONFIG_MAP.get(_STAGE, ProdConfig)
