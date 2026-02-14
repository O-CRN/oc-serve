"""
SGLang ServerArgs with environment variable support
"""
from __future__ import annotations

import dataclasses
import json
import os
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple, Type, TypeVar, Union, get_args, get_origin

from sglang.srt.server_args import ServerArgs
from sglang.srt.managers.template_manager import TemplateManager
from sglang.srt.managers.tokenizer_manager import TokenizerManager

from configs.servers_configs.ServerConfigs import ServerConfigs


T = TypeVar("T", bound="SGLangServerArgs")

_DEFAULT_SGLANG_EXTRA_ARGS: Dict[str, Any] = {}

@ServerConfigs.register("sglang")
@dataclass
class SGLangServerArgs(ServerArgs, ServerConfigs):
    """
    Extension of ServerArgs to support extra arguments via environment variables.
    """

    model_path: str = "Qwen/Qwen3-0.6B"

    extra_args: Namespace = field(default_factory=lambda: Namespace(**_DEFAULT_SGLANG_EXTRA_ARGS))

    @classmethod
    def from_env_vars(cls: Type[T],
                      prefix: str = "SGLANG_",
                      extra_prefix: str = "SGLANG_EXTRA_") -> T:
        """
        Create an instance from environment variables.
        Args:
            prefix: Prefix for normal engine args.
            extra_prefix: Prefix for extra args.
        Returns:
            An instance of ServerArgs.
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
                extra_key = cls._env_key_to_py_key(key[len(extra_prefix) :])
                value = cls._parse_extra_value(raw)
                setattr(inst.extra_args, extra_key, value)
                continue

            # NORMAL ARGS
            if not key.startswith(prefix):
                continue

            # prevent double-processing EXTRA (paranoia)
            if key.startswith(extra_prefix):
                continue

            field_key = cls._env_key_to_py_key(key[len(prefix) :])
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

        # Literal
        if origin is not None and str(origin).endswith("Literal"):
            literal_vals = get_args(target_type)
            for v in literal_vals:
                if isinstance(v, str) and raw_str == v:
                    return v
            for v in literal_vals:
                try:
                    if isinstance(v, bool):
                        return cls._parse_env_value(raw_str, bool)
                    if isinstance(v, int):
                        return int(raw_str)
                    if isinstance(v, float):
                        return float(raw_str)
                except Exception:
                    pass
            # Fallback to raw string
            return raw_str

        # Optional / Union
        if origin is not None:
            if origin in (list, dict, tuple, set, List, Dict, Tuple, Set):
                val = json.loads(raw_str)
                if origin in (tuple, Tuple):
                    return tuple(val)
                if origin in (set, Set):
                    return set(val)
                if origin in (dict, Dict):
                    return dict(val)
                return list(val)

            if origin is Union or str(origin).endswith("Union"):
                for t in get_args(target_type):
                    if t is type(None) and raw_str == "":
                        return None
                    try:
                        return cls._parse_env_value(raw_str, t)
                    except Exception:
                        pass
                raise ValueError(f"Cannot parse '{raw_str}' as {target_type}")

        # primitives
        if target_type is bool:
            v = raw_str.lower()
            if v in {"1", "true", "yes", "on"}:
                return True
            if v in {"0", "false", "no", "off"}:
                return False
            raise ValueError(f"Invalid bool: {raw_str}")

        if target_type in (int, float, str):
            return target_type(raw_str)

        if target_type is Any:
            return cls._parse_extra_value(raw)

        # enums
        try:
            import enum

            if isinstance(target_type, type) and issubclass(target_type, enum.Enum):
                return (
                    target_type[raw_str]
                    if raw_str in target_type.__members__
                    else target_type(raw_str)
                )
        except Exception:
            pass

        # dict JSON
        if raw_str.startswith("{"):
            obj = json.loads(raw_str)
            if isinstance(obj, dict):
                try:
                    return target_type(**obj)
                except Exception:
                    return obj

        # list JSON
        if raw_str.startswith("["):
            obj = json.loads(raw_str)
            return obj

        # fallback
        try:
            return target_type(raw_str)
        except Exception:
            return raw

    @classmethod
    def build(cls) -> "SGLangServerArgs":
        return cls.from_env_vars()


@dataclass
class _GlobalState:
    tokenizer_manager: TokenizerManager
    template_manager: TemplateManager
    scheduler_info: Dict
