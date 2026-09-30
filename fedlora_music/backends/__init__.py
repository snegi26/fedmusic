"""Model backends and their registry.

Backends are looked up by name only: the built-ins below, plus packages installed
on this machine that register a ``fedlora_music.backends`` entry point (value
``module:Class``). A name can never be a module path, so a run config cannot make a
client import arbitrary code, and a client additionally runs only the backends its
node config allows (``LocalPolicy.effective_model``).
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

from fedlora_music.backends.base import Batch, Generator, ModelBackend, PrepareResult
from fedlora_music.config import BUILTIN_BACKENDS

if TYPE_CHECKING:
    from fedlora_music.config import ModelSpec

__all__ = [
    "Batch",
    "Generator",
    "ModelBackend",
    "PrepareResult",
    "backend_class",
    "backend_names",
    "get_backend",
]

ENTRY_POINT_GROUP = "fedlora_music.backends"
_BUILTIN = {
    "acestep": "fedlora_music.backends.acestep:AceStepBackend",
    "toy": "fedlora_music.backends.toy:ToyBackend",
}
assert _BUILTIN.keys() == BUILTIN_BACKENDS


def _plugins() -> dict[str, str]:
    # Built-in names win, so a plugin cannot shadow them.
    return {
        ep.name: ep.value for ep in entry_points(group=ENTRY_POINT_GROUP) if ep.name not in _BUILTIN
    }


def backend_names() -> list[str]:
    return sorted({*_BUILTIN, *_plugins()})


def backend_class(name: str) -> type[ModelBackend]:
    target = _BUILTIN.get(name) or _plugins().get(name)
    if target is None:
        raise KeyError(f"unknown model backend {name!r}; installed: {', '.join(backend_names())}")
    module, _, attr = target.partition(":")
    cls = getattr(importlib.import_module(module), attr)
    if not (isinstance(cls, type) and issubclass(cls, ModelBackend)):
        raise TypeError(f"{target} is not a ModelBackend")
    return cls


@lru_cache(maxsize=4)
def get_backend(spec: ModelSpec) -> ModelBackend:
    """Process-wide backend instance per spec (backends cache loaded metadata)."""
    return backend_class(spec.backend)(spec)
