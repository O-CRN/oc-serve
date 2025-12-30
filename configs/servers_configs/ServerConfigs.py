"""
Abstract base class for server configuration dataclasses.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass
from typing import Callable, ClassVar, Dict, Type, TypeVar
import inspect

C = TypeVar("C", bound="ServerConfigs")


@dataclass
class ServerConfigs:
    """
    Base class for all server config dataclasses.
    Register subclasses via @ServerConfigs.register("name")
    Get instance via ServerConfigs.get("name")
    """
    _REGISTRY: ClassVar[Dict[str, Type["ServerConfigs"]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[C]], Type[C]]:
        """Decorator to register a server config subclass."""
        def deco(cfg_cls: Type[C]) -> Type[C]:
            if not issubclass(cfg_cls, ServerConfigs):
                raise TypeError(f"{cfg_cls.__name__} must inherit from ServerConfigs")

            if not is_dataclass(cfg_cls):
                raise TypeError(f"{cfg_cls.__name__} must be a dataclass")

            sig = inspect.signature(cfg_cls)
            for p in sig.parameters.values():
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is inspect._empty:
                    raise TypeError(f"{cfg_cls.__name__} must be instantiable with no args; " \
                                    f"field '{p.name}' has no default")

            if name in cls._REGISTRY:
                existing = cls._REGISTRY[name]
                raise KeyError(f"Config name '{name}' already registered to \
                               {existing.__module__}.{existing.__name__}")

            cls._REGISTRY[name] = cfg_cls
            return cfg_cls
        return deco

    @classmethod
    def _get_class(cls, server_type: str) -> Type["ServerConfigs"]:
        try:
            return cls._REGISTRY[server_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported server_type: {server_type}. " \
                             f"Available: {sorted(cls._REGISTRY.keys())}") from exc

    @classmethod
    def get(cls, server_type: str) -> "ServerConfigs":
        """Return an instance of the registered config class."""
        cfg_cls = cls._get_class(server_type)
        return cfg_cls.build()

    @classmethod
    def build(cls):
        """Factory method to build an Orchestrator instance."""
        pass