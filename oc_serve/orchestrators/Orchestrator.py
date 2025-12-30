"""Abstract Orchestrator Class"""
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Dict, Type, TypeVar, Annotated

from configs import OrchestratorConfigs
from oc_serve.api.models import (
    Form,
    ChatCompletionRequest,
    CompletionRequest,
    DetokenizeRequest,
    Request,
    Response,
    TokenizeRequest,
    TranscriptionRequest,
)

O = TypeVar("O", bound="Orchestrator")


class Orchestrator(ABC):
    """Base Orchestrator Class"""

    _REGISTRY: ClassVar[Dict[str, Type["Orchestrator"]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[O]], Type[O]]:
        """Decorator to register a concrete Orchestrator subclass."""
        def deco(orch_cls: Type[O]) -> Type[O]:
            if not issubclass(orch_cls, Orchestrator):
                raise TypeError(f"{orch_cls.__name__} must inherit from Orchestrator")

            sig = inspect.signature(orch_cls.__init__)
            params = list(sig.parameters.values())
            if len(params) < 2:
                raise TypeError(
                    f"{orch_cls.__name__}.__init__ must accept (self, orchestrator_configs)"
                )

            if params[1].name not in ("orchestrator_configs", "configs", "cfg"):
                raise TypeError(
                    f"{orch_cls.__name__}.__init__ second arg must be 'orchestrator_configs'"
                )

            if name in cls._REGISTRY:
                raise KeyError(f"Orchestrator '{name}' already registered")

            cls._REGISTRY[name] = orch_cls
            return orch_cls

        return deco

    @classmethod
    def _get_class(cls, orchestrator_type: str) -> Type["Orchestrator"]:
        try:
            return cls._REGISTRY[orchestrator_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported orchestrator_type: {orchestrator_type}. " \
                             f"Available: {sorted(cls._REGISTRY.keys())}") from exc

    @classmethod
    def get(cls, orchestrator_type: str):
        """Return an instance of the registered orchestrator class."""
        orch_cls = cls._get_class(orchestrator_type)
        orchestrator_configs = OrchestratorConfigs.get(orchestrator_type)
        return orch_cls.build(orchestrator_configs)

    @classmethod
    @abstractmethod
    def build(cls, orchestrator_configs: OrchestratorConfigs):
        """Factory method to build an Orchestrator instance."""
        pass

    @abstractmethod
    def __init__(self, orchestrator_configs: OrchestratorConfigs):
        pass

    @abstractmethod
    async def check_api_health(self, raw_request: Request) -> Response:
        """Check API Health Endpoint"""
        pass

    @abstractmethod
    async def check_model_health(self, raw_request: Request) -> Response:
        """Check Model Health Endpoint"""
        pass

    @abstractmethod
    async def get_model_info(self, raw_request: Request) -> Response:
        """Get Model Info Endpoint"""
        pass

    @abstractmethod
    async def instruct(self, request: ChatCompletionRequest,
                       raw_request: Request) -> Response:
        """Instruct Endpoint"""
        pass

    @abstractmethod
    async def complete(self, request: CompletionRequest,
                       raw_request: Request) -> Response:
        """Complete Endpoint"""
        pass

    @abstractmethod
    async def transcribe(self, request: Annotated[TranscriptionRequest, Form()],
                       raw_request: Request) -> Response:
        """Transcribe Endpoint"""
        pass

    @abstractmethod
    async def tokenize(self, request: TokenizeRequest,
                       raw_request: Request) -> Response:
        """Tokenize Endpoint"""
        pass

    @abstractmethod
    async def detokenize(self, request: DetokenizeRequest,
                         raw_request: Request) -> Response:
        """Detokenize Endpoint"""
        pass

    @abstractmethod
    async def get_metrics(self, raw_request: Request) -> Response:
        """Get Metrics Endpoint"""
        pass
