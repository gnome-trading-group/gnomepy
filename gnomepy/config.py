import os

class Config:
    REGISTRY_API_HOST: str
    CONTROLLER_API_URL: str
    COGNITO_DOMAIN: str
    COGNITO_CLIENT_ID: str

class DevConfig(Config):
    REGISTRY_API_HOST: str = 'i3116oczxe.execute-api.us-east-1.amazonaws.com'
    CONTROLLER_API_URL: str = 'da3lbhvdrrwj4.cloudfront.net'
    COGNITO_DOMAIN: str = 'gnome-controller-dev.auth.us-east-1.amazoncognito.com'
    COGNITO_CLIENT_ID: str = '6mtnlmpgc1ojs612r304p7n33v'

class StagingConfig(Config):
    ...

class ProdConfig(Config):
    REGISTRY_API_HOST: str = 'n5dxpwnij0.execute-api.us-east-1.amazonaws.com'
    CONTROLLER_API_URL: str = 'd18yvlldysyqjm.cloudfront.net'
    COGNITO_DOMAIN: str = 'gnome-controller-prod.auth.us-east-1.amazoncognito.com'
    COGNITO_CLIENT_ID: str = '1o8bmieukcgs674hib0cmfj3gi'

_STAGE = os.getenv("STAGE", "prod").lower()

_CONFIG_MAP = {
    "dev": DevConfig,
    "staging": StagingConfig,
    "prod": ProdConfig
}

config = _CONFIG_MAP.get(_STAGE, ProdConfig)
