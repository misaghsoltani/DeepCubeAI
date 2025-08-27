"""Environment package public API.

Exposes a file-backed registry (via `deepcubeai.utils.env_utils`) and provides access to the base `Environment` class.
"""

from __future__ import annotations

from typing import Any

from deepcubeai.environments.environment_abstract import Environment
from deepcubeai.utils import env_utils


def _env_utils() -> Any:
    """Return the runtime env_utils module."""
    return env_utils


def add_env(key: str, module: str, attr: str | None = None, *, env_type: str = "user") -> None:
    """Add a user environment to the registry.

    This is a thin wrapper around `deepcubeai.utils.env_utils.add_env`.
    """
    _env_utils().add_env(key, module, attr, env_type=env_type)


def get_environment(key: str) -> Environment:
    """Instantiate and return the Environment registered under `key`.

    See `deepcubeai.utils.env_utils.get_environment`.
    """
    return _env_utils().get_environment(key)


def list_environments() -> list[str]:
    """Return sorted list of known environment keys."""
    return _env_utils().list_environments()


def list_environments_info() -> dict[str, dict[str, str | None]]:
    """Return detailed registry mapping: key -> {type,module,attr}."""
    return _env_utils().list_environments_info()


def remove_env(key: str) -> None:
    """Remove a user-registered environment from the registry."""
    _env_utils().remove_env(key)


def register_environment(key: str, cls: type[Environment]) -> None:
    """Register an Environment subclass under `key` (compatibility helper)."""
    _env_utils().register_environment(key, cls)


def register_lazy(key: str, module_name: str, attr: str | None = None) -> None:
    """Register a lazily-imported environment (module path + optional attr)."""
    _env_utils().register_lazy(key, module_name, attr)


__all__ = [
    "Environment",
    "add_env",
    "get_environment",
    "list_environments",
    "list_environments_info",
    "remove_env",
    "register_environment",
    "register_lazy",
]
