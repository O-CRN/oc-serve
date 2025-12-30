"""Ray Orchestrator Class
"""
from typing import Annotated

from ray import serve

from oc_serve.orchestrators import Orchestrator
from oc_serve.servers import Server
from oc_serve.api import root_api_app
from oc_serve.api.models import (
    Form,
    Request,
    Response,
    ChatCompletionRequest,
    CompletionRequest,
    DetokenizeRequest,
    TokenizeRequest,
    TranscriptionRequest,
)
from configs import OrchestratorConfigs

@Orchestrator.register("ray")
class Ray(Orchestrator):
    """Ray Orchestrator Class"""
    def __init__(self, orchestrator_configs: OrchestratorConfigs):
        self.orchestrator_configs = orchestrator_configs
        self.server = Server.get(
            self.orchestrator_configs.backend_server_settings.backend_server_type
            )

    @classmethod
    def build(cls, orchestrator_configs: OrchestratorConfigs):
        """Factory method to build a Ray Serve deployment."""
        deployment_settings = orchestrator_configs.deployment_settings.model_dump(exclude_none=True)
        ingressed_cls = serve.ingress(root_api_app)(cls)
        deployment_cls = serve.deployment(**deployment_settings)(ingressed_cls)
        return deployment_cls.bind(orchestrator_configs)


    @root_api_app.post(f"/api-health")
    async def check_api_health(self, raw_request: Request = None):
        return Response(status_code=200, content="API is Healthy!")


    @root_api_app.post(f"/model-health")
    async def check_model_health(self, raw_request: Request = None):
        return await self.server.check_model_health(raw_request)


    @root_api_app.post(f"/model-info")
    async def get_model_info(self, raw_request: Request = None):
        return await self.server.get_model_info(raw_request)


    @root_api_app.post(f"/instruct")
    async def instruct(self, request: ChatCompletionRequest, raw_request: Request):
        return await self.server.instruct(request, raw_request)


    @root_api_app.post(f"/complete")
    async def complete(self, request: CompletionRequest, raw_request: Request):
        return await self.server.complete(request, raw_request)


    @root_api_app.post(f"/transcribe")
    async def transcribe(self, request: Annotated[TranscriptionRequest, Form()],
                       raw_request: Request):
        return await self.server.transcribe(request, raw_request)


    @root_api_app.post(f"/tokenize")
    async def tokenize(self, request: TokenizeRequest, raw_request: Request):
        return await self.server.tokenize(request, raw_request)


    @root_api_app.post(f"/detokenize")
    async def detokenize(self, request: DetokenizeRequest, raw_request: Request):
        return await self.server.detokenize(request, raw_request)


    @root_api_app.get(f"/metrics")
    async def get_metrics(self, raw_request: Request):
        return self.server.metrics(request=raw_request)
