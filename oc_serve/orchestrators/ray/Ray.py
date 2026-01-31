"""Ray Orchestrator Class
"""
from typing import Annotated

from ray import serve

from oc_serve.orchestrators import Orchestrator
from oc_serve.servers import Server
from oc_serve.api import root_api_app
from oc_serve.api.models import (
    Form,
    OCChatCompletionRequest,
    OCCompletionRequest,
    OCDetokenizeRequest,
    OCPoolingRequest,
    OCRequest,
    OCResponse,
    OCScoringRequest,
    OCTokenizeRequest,
    OCTranscriptionRequest,
    OCRerankRequest,
    OCEmbeddingRequest,
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

    @root_api_app.post("/api-health")
    @root_api_app.post("/v1/api-health")
    async def check_api_health(self, raw_request: OCRequest = None) -> OCResponse:
        return OCResponse(status_code=200, content="API is Healthy!")

    @root_api_app.post("/model-health")
    @root_api_app.post("/v1/model-health")
    async def check_model_health(self, raw_request: OCRequest) -> OCResponse:
        return await self.server.check_model_health(raw_request)

    @root_api_app.post("/model-info")
    @root_api_app.get("/v1/models")
    async def get_model_info(self, raw_request: OCRequest) -> OCResponse:
        return await self.server.get_model_info(raw_request)

    @root_api_app.post("/v1/chat/completions")
    async def instruct(self, request: OCChatCompletionRequest, raw_request: OCRequest) -> OCResponse:
        return await self.server.instruct(request, raw_request)

    @root_api_app.post("/v1/completions")
    async def complete(self, request: OCCompletionRequest, raw_request: OCRequest) -> OCResponse:
        return await self.server.complete(request, raw_request)

    @root_api_app.post("/v1/transcriptions")
    @root_api_app.post("/v1/audio/transcriptions")
    async def transcribe(self, request: Annotated[OCTranscriptionRequest, Form()],
                       raw_request: OCRequest) -> OCResponse:
        return await self.server.transcribe(request, raw_request)

    @root_api_app.post("/v1/tokenize")
    async def tokenize(self, request: OCTokenizeRequest, raw_request: OCRequest) -> OCResponse:
        return await self.server.tokenize(request, raw_request)

    @root_api_app.post("/v1/detokenize")
    async def detokenize(self, request: OCDetokenizeRequest, raw_request: OCRequest) -> OCResponse:
        return await self.server.detokenize(request, raw_request)

    @root_api_app.post("/v1/score")
    async def score(self, request: OCScoringRequest, raw_request: OCRequest) -> OCResponse:
        """Scoring Endpoint"""
        return await self.server.score(request, raw_request)

    @root_api_app.post("/v1/pool")
    async def pool(self, request: OCPoolingRequest, raw_request: OCRequest) -> OCResponse:
        """Pooling Endpoint"""
        return await self.server.pool(request, raw_request)

    @root_api_app.post("/v1/embeddings")
    async def embed(self, request: OCEmbeddingRequest, raw_request: OCRequest) -> OCResponse:
        """Embedding Endpoint"""
        return await self.server.embed(request, raw_request)

    @root_api_app.post("/v1/rerank")
    async def rerank(self, request: OCRerankRequest, raw_request: OCRequest) -> OCResponse:
        """Rerank Endpoint"""
        return await self.server.rerank(request, raw_request)

    @root_api_app.get("/metrics")
    @root_api_app.get("/v1/metrics")
    async def get_metrics(self, request: OCRequest) -> OCResponse:
        return await self.server.get_metrics(request=request)
