"""
OC Serve App Configs
"""
from pydantic_settings import SettingsConfigDict, BaseSettings


class OCServeConfigs(BaseSettings):
    """Ray Server Type Dataclass"""
    orchestrator_type: str = "ray"

    model_config = SettingsConfigDict(
        env_prefix="OC_SERVE_",
        extra="ignore",
    )
