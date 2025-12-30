"""
VLLM Async Engine Args with environment variable support.
"""
from __future__ import annotations

import dataclasses
import json
import os
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Dict, Type, TypeVar, get_args, get_origin

from vllm.engine.arg_utils import AsyncEngineArgs

from .ServerConfigs import ServerConfigs

T = TypeVar("T", bound="VLLMAsyncEngineArgs")


_DEFAULT_EXTRA_ARGS: Dict[str, Any] = {
    "response_role": "assistant",
    "max_concurrent_calls": 1,
    "chat_template": None,
    "lora_modules": None,
    "skips": 0,
    "return_tokens_as_token_ids": False,
    "enable_auto_tools": False,
    "tool_parser": None,
    "chat_template_content_format": "auto",
    "vllm_use_v1": 1,
    "vllm_enable_pooling": False,
    "vllm_enable_scoring": False,
    "use_transcribe_server": False,
}

@ServerConfigs.register("vllm")
@dataclass
class VLLMAsyncEngineArgs(AsyncEngineArgs, ServerConfigs):
    """
    Extension of AsyncEngineArgs to support extra arguments via environment variables.
    """

    extra_args: Namespace = field(default_factory=lambda: Namespace(**_DEFAULT_EXTRA_ARGS))

    @classmethod
    def from_env_vars(cls: Type[T],
                      prefix: str = "VLLM_",
                      extra_prefix: str = "VLLM_EXTRA_") -> T:
        """
        Create an instance from environment variables.
        Args:
            prefix: Prefix for normal engine args.
            extra_prefix: Prefix for extra args.
        Returns:
            An instance of EngineArgs.
        """
        inst = cls()

        field_map = {
            f.name.lower(): f
            for f in dataclasses.fields(cls)
            if f.name != "extra_args"
        }

        for key, raw in os.environ.items():
            # EXTRA ARGS (Namespace)
            if key.startswith(extra_prefix):
                extra_key = cls._env_key_to_py_key(key[len(extra_prefix):])
                value = cls._parse_extra_value(raw)
                setattr(inst.extra_args, extra_key, value)
                continue

            # NORMAL ENGINE ARGS
            if not key.startswith(prefix):
                continue

            # prevent double-processing EXTRA
            if key.startswith(extra_prefix):
                continue

            field_key = cls._env_key_to_py_key(key[len(prefix):])
            f = field_map.get(field_key)
            if f is None:
                continue

            try:
                value = cls._parse_env_value(raw, f.type)
                setattr(inst, f.name, value)
            except Exception:
                continue

        return inst

    @staticmethod
    def _env_key_to_py_key(s: str) -> str:

        s = s.strip().lstrip("_")
        s = s.replace("__", "_").replace("-", "_").replace(".", "_")
        return s.lower()

    @staticmethod
    def _parse_extra_value(raw: str) -> Any:

        v = raw.strip()
        try:
            return json.loads(v)
        except Exception:
            return raw

    @classmethod
    def _parse_env_value(cls, raw: str, target_type: Any) -> Any:

        raw_str = raw.strip()
        origin = get_origin(target_type)

        # Optional / Union
        if origin is not None:
            if origin in (list, dict, tuple, set):
                val = json.loads(raw_str)
                return origin(val) if origin is not tuple else tuple(val)

            if origin is getattr(__import__("typing"), "Union"):
                for t in get_args(target_type):
                    if t is type(None) and raw_str == "":
                        return None
                    try:
                        return cls._parse_env_value(raw_str, t)
                    except Exception:
                        pass
                raise ValueError

        # primitives
        if target_type is bool:
            v = raw_str.lower()
            if v in {"1", "true", "yes", "on"}:
                return True
            if v in {"0", "false", "no", "off"}:
                return False
            raise ValueError

        if target_type in (int, float, str):
            return target_type(raw_str)

        if target_type is Any:
            return cls._parse_extra_value(raw)

        # enums
        try:
            import enum
            if isinstance(target_type, type) and issubclass(target_type, enum.Enum):
                return target_type[raw_str] if raw_str in target_type.__members__ else target_type(raw_str)
        except Exception:
            pass

        # dict → ctor
        if raw_str.startswith("{"):
            obj = json.loads(raw_str)
            if isinstance(obj, dict):
                try:
                    return target_type(**obj)
                except Exception:
                    return obj

        # fallback
        try:
            return target_type(raw_str)
        except Exception:
            return raw

    @classmethod
    def build(cls):
        return cls.from_env_vars()