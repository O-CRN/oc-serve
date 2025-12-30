"""
Ray Orchestrator Configurations
"""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from pydantic_settings import SettingsConfigDict, BaseSettings

from configs.orchestrators_configs import OrchestratorConfigs


class RayDeploymentSettings(BaseSettings):
    """Ray Deployment Settings Dataclass"""
    name: Optional[str] = None
    version: Optional[str] = None
    num_replicas: Optional[Union[int, str]] = None
    route_prefix: Optional[str] = None

    ray_actor_options: Optional[Dict[str, Any]] = None
    placement_group_bundles: Optional[List[Dict[str, float]]] = None
    placement_group_strategy: Optional[str] = None
    max_replicas_per_node: Optional[int] = None

    user_config: Optional[Any] = None
    max_ongoing_requests: Optional[int] = None
    max_queued_requests: Optional[int] = None
    autoscaling_config: Optional[Dict[str, Any]] = None

    graceful_shutdown_wait_loop_s: Optional[float] = None
    graceful_shutdown_timeout_s: Optional[float] = None
    health_check_period_s: Optional[float] = None
    health_check_timeout_s: Optional[float] = None

    logging_config: Optional[Dict[str, Any]] = None
    request_router_config: Optional[Dict[str, Any]] = None

    model_config = SettingsConfigDict(
        env_prefix="RAY_",
        extra="ignore",
    )


    @staticmethod
    def _json_fields():
        return {
            "ray_actor_options",
            "placement_group_bundles",
            "user_config",
            "autoscaling_config",
            "logging_config",
            "request_router_config",
        }


    def _json_or_none(self, v):
        if v is None or v == "":
            return None
        if isinstance(v, (dict, list)):
            return v
        return json.loads(v)

    def model_post_init(self, __context):
        for field in self._json_fields():
            value = getattr(self, field)
            if value is not None:
                setattr(self, field, self._json_or_none(value))

class RayBackendServerSettings(BaseSettings):
    """Ray Server Type Dataclass"""
    backend_server_type: str = "vllm"

    model_config = SettingsConfigDict(
        env_prefix="RAY_",
        extra="ignore",
    )


@OrchestratorConfigs.register("ray")
@dataclass
class RayConfigs(OrchestratorConfigs):
    """Ray Orchestrator Configurations Dataclass"""
    backend_server_settings: RayBackendServerSettings = field(default_factory=RayBackendServerSettings)
    deployment_settings: RayDeploymentSettings = field(default_factory=RayDeploymentSettings)
