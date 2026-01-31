"""
Abstract Model Server Class
"""
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Dict, Type, TypeVar, Annotated

from configs import ServerConfigs
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


S = TypeVar("S", bound="Server")

class Server(ABC):
    """Abstract Model Server Class"""

    _REGISTRY: ClassVar[Dict[str, Type["Server"]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[S]], Type[S]]:
        """
        Register a concrete Server subclass under `name`.
        """
        def deco(server_cls: Type[S]) -> Type[S]:
            if not issubclass(server_cls, Server):
                raise TypeError(f"{server_cls.__name__} must inherit from Server")

            sig = inspect.signature(server_cls.__init__)
            params = list(sig.parameters.values())

            if len(params) < 2:
                raise TypeError(f"{server_cls.__name__}.__init__ must accept (self, server_configs)")

            if params[1].name not in ("server_configs", "configs", "cfg"):
                raise TypeError(f"{server_cls.__name__}.__init__ second arg should be \
                                'server_configs' " f"(or configs/cfg). Got '{params[1].name}'.")

            if name in cls._REGISTRY:
                existing = cls._REGISTRY[name]
                raise KeyError(f"Server name '{name}' already registered to " \
                               f"{existing.__module__}.{existing.__name__}")

            cls._REGISTRY[name] = server_cls
            return server_cls
        return deco

    @classmethod
    def _get_class(cls, server_type: str) -> Type["Server"]:
        try:
            return cls._REGISTRY[server_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported server_type: {server_type}. " \
                             f"Available: {sorted(cls._REGISTRY.keys())}") from exc

    @classmethod
    def get(cls, server_type: str) -> "Server":
        """Factory method to build a Server instance based on server_type."""
        server_cls = cls._get_class(server_type)
        server_configs = ServerConfigs.get(server_type)
        return server_cls(server_configs=server_configs)

    @abstractmethod
    def __init__(self, server_configs: ServerConfigs):
        pass

    @abstractmethod
    async def check_model_health(self, raw_request: OCRequest) -> OCResponse:
        """Check Model Health Endpoint"""
        pass

    @abstractmethod
    async def get_model_info(self, raw_request: OCRequest) -> OCResponse:
        """Get Model Info Endpoint"""
        pass

    @abstractmethod
    async def instruct(self, request: OCChatCompletionRequest, raw_request: OCRequest) -> OCResponse:
        """Instruct Endpoint"""
        pass

    @abstractmethod
    async def complete(self, request: OCCompletionRequest, raw_request: OCRequest) -> OCResponse:
        """Complete Endpoint"""
        pass

    @abstractmethod
    async def transcribe(self, request: Annotated[OCTranscriptionRequest, Form()],
                       raw_request: OCRequest) -> OCResponse:
        """Transcribe Endpoint"""
        pass

    @abstractmethod
    async def tokenize(self, request: OCTokenizeRequest, raw_request: OCRequest) -> OCResponse:
        """Tokenize Endpoint"""
        pass

    @abstractmethod
    async def detokenize(self, request: OCDetokenizeRequest, raw_request: OCRequest) -> OCResponse:
        """Detokenize Endpoint"""
        pass

    @abstractmethod
    async def score(self, request: OCScoringRequest, raw_request: OCRequest) -> OCResponse:
        """Scoring Endpoint"""
        pass

    @abstractmethod
    async def pool(self, request: OCPoolingRequest, raw_request: OCRequest) -> OCResponse:
        """Pooling Endpoint"""
        pass

    @abstractmethod
    async def rerank(self, request: OCRerankRequest, raw_request: OCRequest) -> OCResponse:
        """Rerank Endpoint"""
        pass

    @abstractmethod
    async def embed(self, request: OCEmbeddingRequest, raw_request: OCRequest) -> OCResponse:
        """Embedding Endpoint"""
        pass

    @abstractmethod
    async def get_metrics(self, request: OCRequest) -> OCResponse:
        """Get Metrics Endpoint"""
        pass
