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
    ChatCompletionRequest,
    CompletionRequest,
    DetokenizeRequest,
    PoolingRequest,
    Request,
    Response,
    ScoreRequest,
    TokenizeRequest,
    TranscriptionRequest,
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
    async def check_model_health(self, raw_request: Request):
        """Check Model Health Endpoint"""
        pass

    @abstractmethod
    async def get_model_info(self, raw_request: Request):
        """Get Model Info Endpoint"""
        pass

    @abstractmethod
    async def instruct(self, request: ChatCompletionRequest, raw_request: Request):
        """Instruct Endpoint"""
        pass

    @abstractmethod
    async def complete(self, request: CompletionRequest, raw_request: Request):
        """Complete Endpoint"""
        pass

    @abstractmethod
    async def transcribe(self, request: Annotated[TranscriptionRequest, Form()],
                       raw_request: Request) -> Response:
        """Transcribe Endpoint"""
        pass

    @abstractmethod
    async def metrics(self, request: Request) -> Response:
        """Get Metrics Endpoint"""
        pass

    @abstractmethod
    async def tokenize(self, request: TokenizeRequest, raw_request: Request):
        """Tokenize Endpoint"""
        pass

    @abstractmethod
    async def scoring(self, request: ScoreRequest, raw_request: Request):
        """Scoring Endpoint"""
        pass

    @abstractmethod
    async def pooling(self, request: PoolingRequest, raw_request: Request):
        """Pooling Endpoint"""
        pass

    @abstractmethod
    async def detokenize(self, request: DetokenizeRequest, raw_request: Request):
        """Detokenize Endpoint"""
        pass
