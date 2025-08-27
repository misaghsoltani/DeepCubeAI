from __future__ import annotations

__version__ = "0.2.1"
__author__ = "Misagh Soltani"

import importlib
from types import ModuleType

__all__ = ["__version__", "__author__"]


def _env_pkg() -> ModuleType:
    """Lazily import and return the environments package object.

    This avoids importing the environments package (and thereby executing
    any discovery) at top-level import time of `deepcubeai`. Callers that
    need to interact with the registry should do so via the thin wrappers
    below which import on demand.
    """
    return importlib.import_module("deepcubeai.utils.env_utils")


def register_env(key: str, cls: type) -> None:
    """Register an environment class under `key`.

    Usage from external packages:
                   import deepcubeai

                   deepcubeai.register_env("myenv", MyEnvClass)
    """
    pkg = _env_pkg()
    # Expose 'register_environment'
    pkg.register_environment(key, cls)


def register_lazy_env(key: str, module_name: str, attr: str | None = None) -> None:
    """Register a lazy mapping for an environment implemented elsewhere.

    This registers a module path and optional attribute name. The module is only imported
    when the environment is requested via :func:`deepcubeai.get_env`.
    """
    pkg = _env_pkg()
    pkg.register_lazy(key, module_name, attr)


def get_env(key: str) -> object:
    """Instantiate and return the environment registered under 'key'."""
    pkg = _env_pkg()
    return pkg.get_environment(key)


def list_environments() -> list[str]:
    """Return the list of known environment keys.

    This function does not import individual environment modules when creating the list.
    """
    pkg = _env_pkg()
    return pkg.list_environments()


__all__.extend(["register_env", "register_lazy_env", "get_env", "list_environments"])
