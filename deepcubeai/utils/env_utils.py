from __future__ import annotations

from collections.abc import Mapping
import importlib
import json
import os
from pathlib import Path
from threading import RLock

from deepcubeai.environments.environment_abstract import Environment

# File-backed registry: deepcubeai/environments/envs.json
_LOCK = RLock()
_REGISTRY_FILE: Path = Path(__file__).resolve().parents[1] / "environments" / "envs.json"

# Builtin environment metadata:
# These entries are always available even if the file-backed registry is missing or deleted.
BUILTINS: dict[str, dict[str, str | None]] = {
    "cube3": {"type": "builtin", "module": "deepcubeai.environments.cube3", "attr": "Cube3"},
    "cube3_triples": {"type": "builtin", "module": "deepcubeai.environments.cube3", "attr": "Cube3Triples"},
    "digitjump": {"type": "builtin", "module": "deepcubeai.environments.digit_jump", "attr": "DigitJumpEnvironment"},
    "iceslider": {"type": "builtin", "module": "deepcubeai.environments.ice_slider", "attr": "IceSliderEnvironment"},
    "sokoban": {"type": "builtin", "module": "deepcubeai.environments.sokoban", "attr": "Sokoban"},
}


def _load_registry() -> dict[str, dict[str, str | None]]:
    """Read the registry.

    Callers will merge BUILTINS on demand so builtins remain available.
    """
    try:
        if not _REGISTRY_FILE.exists():
            return {}
        raw = _REGISTRY_FILE.read_bytes()
        data = json.loads(raw)
        out: dict[str, dict[str, str | None]] = {}
        if isinstance(data, Mapping):
            for k, v in data.items():
                # Normalize entries into expected shape: {type, module, attr}
                if not isinstance(v, Mapping):
                    continue

                out[str(k)] = {
                    "type": (
                        v.get("type") if v.get("type") is None or isinstance(v.get("type"), str) else str(v.get("type"))
                    ),
                    "module": (
                        v.get("module")
                        if v.get("module") is None or isinstance(v.get("module"), str)
                        else str(v.get("module"))
                    ),
                    "attr": (
                        v.get("attr") if v.get("attr") is None or isinstance(v.get("attr"), str) else str(v.get("attr"))
                    ),
                }
            return out

        return {}

    except Exception:
        return {}


def _save_registry(reg: dict[str, dict[str, str | None]]) -> None:
    # atomic write
    tmp = _REGISTRY_FILE.with_suffix(".json.tmp")
    tmp.write_bytes(json.dumps(reg, indent=2, ensure_ascii=False).encode("utf-8"))
    os.replace(tmp, _REGISTRY_FILE)


def list_environments() -> list[str]:
    """Return the list of registered environment keys (sorted)."""
    reg = _load_registry()
    # merge builtin keys
    keys = set(BUILTINS.keys()) | set(reg.keys())

    return sorted(keys)


def list_environments_info() -> dict[str, dict[str, str | None]]:
    """Return detailed info for each known environment key.

    Returns mapping: key -> {type,module,attr}
    """
    user = _load_registry()
    out: dict[str, dict[str, str | None]] = {}
    # Start with builtins (authoritative)
    for k, v in BUILTINS.items():
        out[k] = {"type": v.get("type"), "module": v.get("module"), "attr": v.get("attr")}

    # Overlay user entries (do not overwrite builtin metadata)
    for k, v in user.items():
        if k in out and out[k].get("type") == "builtin":
            continue

        out[k] = {"type": v.get("type"), "module": v.get("module"), "attr": v.get("attr")}

    return out


def get_environment(key: str) -> Environment:
    """Instantiate and return the Environment registered under `key`.

    Raises ValueError if not found or cannot be instantiated.
    """
    if not isinstance(key, str) or not key:
        raise ValueError("key must be a non-empty string")
    k = key.lower()
    # prefer builtin metadata
    info: dict[str, str | None] | None
    if k in BUILTINS:
        info = BUILTINS[k]
    else:
        reg = _load_registry()
        info = reg.get(k)
        if not info:
            raise ValueError(f"No registered environment with key: {key}")
    module_name = info.get("module")
    attr = info.get("attr")
    if not module_name:
        raise ValueError(f"Registry entry for {key} missing module")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise ValueError(f"Failed importing environment module {module_name}: {exc}") from exc

    try:
        if attr:
            candidate = getattr(module, attr)
        else:
            # fallback: find first Environment subclass in module
            candidate = None
            for _n, obj in vars(module).items():
                if isinstance(obj, type) and issubclass(obj, Environment):
                    candidate = obj
                    break
            if candidate is None:
                raise AttributeError("no Environment subclass found in module")
    except AttributeError as exc:
        raise ValueError(f"Module {module_name} has no attribute {attr}: {exc}") from exc

    if not isinstance(candidate, type) or not issubclass(candidate, Environment):
        raise TypeError(f"Registered attribute for {key} is not an Environment subclass")

    try:
        return candidate()
    except Exception as exc:
        raise ValueError(f"Failed instantiating environment {key}: {exc}") from exc


def add_env(key: str, module: str, attr: str | None = None, *, env_type: str = "user") -> None:
    """Add a registry entry (user-provided). Builtins should be marked 'builtin'."""
    if not isinstance(key, str) or not key:
        raise ValueError("key must be a non-empty string")
    k = key.lower()
    with _LOCK:
        # If user attempts to overwrite a builtin, reject
        if k in BUILTINS:
            raise ValueError("Cannot overwrite builtin environment")
        reg = _load_registry()
        reg[k] = {"type": env_type, "module": module, "attr": attr}
        # Ensure the registry file exists (create if missing) and persist user entries only
        parent = _REGISTRY_FILE.parent
        parent.mkdir(parents=True, exist_ok=True)
        if not _REGISTRY_FILE.exists():
            # create an empty dict
            _REGISTRY_FILE.write_bytes(json.dumps({}).encode("utf-8"))
        _save_registry(reg)


def remove_env(key: str) -> None:
    """Remove a user-registered environment from the registry.

    Args:
        key (str): The key of the environment to remove.
    """
    if not isinstance(key, str) or not key:
        raise ValueError("key must be a non-empty string")
    k = key.lower()
    with _LOCK:
        # Cannot remove builtin entries
        if k in BUILTINS:
            raise ValueError("Cannot remove builtin environment")
        reg = _load_registry()
        if k not in reg:
            raise ValueError("No such environment registered")
        reg.pop(k, None)
        _save_registry(reg)


def register_environment(key: str, cls: type[Environment]) -> None:
    """Register a class object (marks builtin if in-package)."""
    if not isinstance(key, str) or not key:
        raise ValueError("key must be a non-empty string")
    module = getattr(cls, "__module__", None)
    name = getattr(cls, "__name__", None)
    env_type = "builtin" if module and module.startswith("deepcubeai.environments") else "user"
    add_env(key, module or "", name, env_type=env_type)


def register_lazy(key: str, module_name: str, attr: str | None = None) -> None:
    """Add an entry pointing to a module/attr (lazy import on get)."""
    add_env(key, module_name, attr, env_type="user")
