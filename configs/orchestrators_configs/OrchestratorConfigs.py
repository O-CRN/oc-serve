"""
Abstract base class for orchestrator configuration dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from typing import Callable, ClassVar, Dict, Type, TypeVar
import inspect

O = TypeVar("O", bound="OrchestratorConfigs")


@dataclass
class OrchestratorConfigs:
    """
    Base class for orchestrator config dataclasses.

    Register subclasses via @OrchestratorConfigs.register("name")
    Get instance via OrchestratorConfigs.get("name")
    """
    _REGISTRY: ClassVar[Dict[str, Type["OrchestratorConfigs"]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[O]], Type[O]]:
        """Decorator to register an orchestrator config subclass."""
        def deco(cfg_cls: Type[O]) -> Type[O]:
            if not issubclass(cfg_cls, OrchestratorConfigs):
                raise TypeError(f"{cfg_cls.__name__} must inherit from OrchestratorConfigs")

            if not is_dataclass(cfg_cls):
                raise TypeError(f"{cfg_cls.__name__} must be a dataclass")

            sig = inspect.signature(cfg_cls)
            for p in sig.parameters.values():
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is inspect._empty:
                    raise TypeError(f"{cfg_cls.__name__} must be instantiable with no args; " \
                                    f"field '{p.name}' has no default")

            if name in cls._REGISTRY:
                existing = cls._REGISTRY[name]
                raise KeyError(f"Orchestrator config name '{name}' already registered to " \
                               f"{existing.__module__}.{existing.__name__}")

            cls._REGISTRY[name] = cfg_cls
            return cfg_cls
        return deco

    @classmethod
    def _get_class(cls, orchestrator_type: str) -> Type["OrchestratorConfigs"]:
        try:
            return cls._REGISTRY[orchestrator_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported orchestrator_type: {orchestrator_type}. " \
                             f"Available: {sorted(cls._REGISTRY.keys())}") from exc
    @classmethod
    def get(cls, orchestrator_type: str) -> "OrchestratorConfigs":
        """Return an instance of the registered orchestrator config class."""
        return cls._get_class(orchestrator_type)()
