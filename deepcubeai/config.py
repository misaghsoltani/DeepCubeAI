from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

PKG_ROOT: Path = Path(__file__).resolve().parent  # .../DeepCubeAI/deepcubeai


@dataclass(frozen=True, slots=True)
class OfflineDataPaths:
    """Resolved paths for offline datasets and encoded variants."""

    offline_dir: Path
    offline_enc_dir: Path
    env_test_dir: Path
    search_test_dir: Path
    data_sample_img_dir: Path
    train: Path
    val: Path
    env_test: Path
    search_test: Path
    train_enc: Path
    val_enc: Path


@dataclass(frozen=True, slots=True)
class EnvModelPaths:
    """Locations where environment models are saved and read from."""

    save_dir: Path
    model_dir: Path


@dataclass(frozen=True, slots=True)
class HeurModelPaths:
    """Locations for heuristic models and the current checkpoint directory."""

    save_dir: Path
    current_dir: Path


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Configuration for the pipeline.

    This aggregates all knobs needed across stages. It also includes helpers to
    derive default paths and counts. Values may be None to indicate using
    defaults derived at runtime.
    """

    # core
    stage: str = ""
    env: str = ""

    # data/pathing
    data_dir: str | None = None
    data_file_name: str | None = None
    num_offline_steps: int | None = None
    num_cpus: int = 1

    # offline episode counts
    num_train_eps: int | None = None
    num_val_eps: int | None = None
    num_test_eps: int = 100

    # seeding across levels
    start_level: int | None = None
    num_levels: int | None = None

    # training/test knobs
    env_model_name: str | None = None
    env_batch_size: int = 100
    print_interval: int = 1

    # heur training
    heur_nnet_name: str | None = None
    heur_batch_size: int | None = None
    states_per_update: int | None = None
    max_solve_steps: int | None = None
    start_steps: int | None = None
    goal_steps: int | None = None
    num_test: int | None = 1000
    per_eq_tol: float | None = None
    use_dist: bool = False
    # distributed heuristic training
    lr: float | None = None
    lr_d: float | None = None
    max_itrs: int | None = None
    update_nnet_batch_size: int | None = None
    ring: int | None = None
    seed: int | None = None
    debug: bool = False
    amp: bool = False
    compile: bool = False
    compile_all_models: bool = False

    # search
    qstar_batch_size: int | None = None
    qstar_weight: float | None = None
    qstar_h_weight: float | None = None
    qstar_results_dir: str | None = None
    ucs_batch_size: int | None = None
    ucs_results_dir: str | None = None
    gbfs_results_dir: str | None = None
    search_itrs: int | None = None
    save_imgs: str | bool | None = None
    search_test_data: str | None = None
    reverse: bool = False  # cube3-only flag

    # disc vs cont plotting
    model_test_data_dir: str | None = None
    env_model_dir_disc: str | None = None
    env_model_dir_cont: str | None = None
    num_episodes: int | None = None
    num_steps: int | None = None
    save_dir: str | None = None

    # data viz
    num_train_trajs_viz: int | None = 8
    num_train_steps_viz: int | None = 2
    num_val_trajs_viz: int | None = 8
    num_val_steps_viz: int | None = 2

    # compare solutions
    soln1: str | None = None
    soln2: str | None = None

    # Factory methods for unified configuration
    @staticmethod
    def from_json(path: str | Path) -> PipelineConfig:
        """Load configuration from a JSON file, ignoring unknown keys."""
        raw = json.loads(Path(path).read_bytes())
        if not isinstance(raw, dict):
            raise TypeError("Config JSON must be an object")
        allowed: set[str] = set(PipelineConfig.__dataclass_fields__.keys())
        data: dict[str, Any] = {k: v for k, v in raw.items() if k in allowed}
        return PipelineConfig(**data)

    def to_json_bytes(self) -> bytes:
        """Serialize the config to compact JSON bytes."""
        # Build a plain dict to avoid non-serializable types (Paths become str)
        d: dict[str, Any] = {
            k: (str(v) if isinstance(v := getattr(self, k), Path) else v)
            for k in self.__slots__
            if not k.startswith("_")
        }
        # Convert any non-str keys to str
        return json.dumps({str(k): v for k, v in d.items()}, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    @staticmethod
    def from_env(prefix: str = "DCAI_") -> PipelineConfig:
        """Build a config from environment variables.

        Mapping rule: DCAI_FIELD_NAME (upper) -> field name. Values are parsed by
        the annotated type where possible (int, float, bool, Path-like, str/None).
        Missing values fall back to dataclass defaults.
        """
        overrides: dict[str, Any] = {}
        for fname, f in PipelineConfig.__dataclass_fields__.items():
            if fname.startswith("_"):
                continue
            raw = os.getenv(prefix + fname.upper())
            if raw is None:
                continue

            val: int | float | str | None = raw
            try:
                if f.type in {int, "int"} or isinstance(getattr(PipelineConfig(), fname), int):
                    val = int(raw)
                elif f.type in {float, "float"} or isinstance(getattr(PipelineConfig(), fname), float):
                    val = float(raw)
                elif f.type in {bool, "bool"} or isinstance(getattr(PipelineConfig(), fname), bool):
                    val = raw.strip().lower() in {"1", "true", "yes", "y", "on"}
                elif f.type in {Path, "Path"}:
                    val = str(Path(raw))
                elif not raw:
                    val = None
                else:
                    val = raw
            except Exception:
                val = raw

            overrides[fname] = val

        return PipelineConfig(**overrides)

    def offline_paths(self) -> OfflineDataPaths:
        """Resolve and return offline data paths."""
        # Resolve data directory
        cand: Path = Path(self.data_dir or self.env).expanduser()
        base: Path
        if cand.is_absolute():
            base = cand
        else:
            parts = cand.parts
            base = (
                PKG_ROOT.parent / cand
                if len(parts) >= 2 and parts[0] == "deepcubeai" and parts[1] == "data"
                else (PKG_ROOT / "data") / cand
            )

        offline_dir: Path = base / "offline"
        offline_enc_dir: Path = base / "offline_enc"
        env_test_dir: Path = base / "model_test"
        search_test_dir: Path = base / "search_test"
        data_sample_img_dir: Path = base / "sample_images"

        def name_or(suffix: str, default: str) -> str:
            if not self.data_file_name:
                return default
            return self.data_file_name if suffix in self.data_file_name else f"{self.data_file_name}_{suffix}"

        train_name: str = name_or("train_data", "train_data")
        val_name: str = name_or("val_data", "val_data")
        env_test_name: str = name_or("env_test_data", "env_test_data")
        search_test_name: str = name_or("search_test_data", "search_test_data")
        train_enc_name: str = name_or("train_data_enc", "train_data_enc")
        val_enc_name: str = name_or("val_data_enc", "val_data_enc")

        return OfflineDataPaths(
            offline_dir=offline_dir,
            offline_enc_dir=offline_enc_dir,
            env_test_dir=env_test_dir,
            search_test_dir=search_test_dir,
            data_sample_img_dir=data_sample_img_dir,
            train=offline_dir / f"{train_name}.pkl",
            val=offline_dir / f"{val_name}.pkl",
            env_test=env_test_dir / f"{env_test_name}.pkl",
            search_test=search_test_dir / f"{search_test_name}.pkl",
            train_enc=offline_enc_dir / f"{train_enc_name}.pkl",
            val_enc=offline_enc_dir / f"{val_enc_name}.pkl",
        )

    def env_model_paths(self) -> EnvModelPaths:
        """Return env model save and model directory paths."""
        name = self.env_model_name or "env_model"
        save_dir: Path = PKG_ROOT / "saved_env_models"
        model_dir: Path = save_dir / name
        return EnvModelPaths(save_dir=save_dir, model_dir=model_dir)

    def heur_model_paths(self) -> HeurModelPaths:
        """Return heuristic model save and current checkpoint directories."""
        name = self.heur_nnet_name or "heur_model"
        save_dir: Path = PKG_ROOT / "saved_heur_models"
        current: Path = save_dir / name / "current"
        return HeurModelPaths(save_dir=save_dir, current_dir=current)

    def derive_offline_counts(self) -> tuple[int, int, int]:
        """Return train/val/test episode counts using sensible defaults."""
        if self.num_train_eps is None and self.num_val_eps is None:
            train, val = 9000, 1000
        elif self.num_train_eps is not None and self.num_val_eps is None:
            val = max(1, self.num_train_eps // 9)
            train = self.num_train_eps
        elif self.num_train_eps is None and self.num_val_eps is not None:
            train = 9 * self.num_val_eps
            val = self.num_val_eps
        else:
            train, val = int(self.num_train_eps or 0), int(self.num_val_eps or 0)

        test = self.num_test_eps or 100

        return int(train), int(val), int(test)

    def derive_seeds(self, train_eps: int) -> tuple[int, int, int, int, int, int]:
        """Return seed settings."""
        s_level, n_levels = self.start_level, self.num_levels
        if s_level is not None and n_levels is not None:
            s_train = s_level
            n_train = n_levels
            s_val = s_train + n_train
            n_val = n_levels
        elif s_level is not None:
            s_train = s_level
            n_train = n_levels if n_levels is not None else train_eps
            s_val = s_train + train_eps
            n_val = n_levels if n_levels is not None else train_eps
        elif n_levels is not None:
            s_train = -1
            n_train = n_levels
            s_val = -1
            n_val = n_levels
        else:
            s_train = -1
            n_train = -1
            s_val = -1
            n_val = -1

        s_test = s_train
        n_test = n_train

        return s_train, n_train, s_val, n_val, s_test, n_test


def merge_config(*candidates: PipelineConfig) -> PipelineConfig:
    """Merge multiple PipelineConfig instances with left-to-right precedence.

    Later configs override earlier ones when a field in the later config is
    considered "set" (i.e., not None/empty). Booleans use logical OR.
    """
    if not candidates:
        return PipelineConfig()

    base: PipelineConfig = candidates[0]
    new_data: dict[str, Any] = {k: getattr(base, k) for k in base.__slots__ if not k.startswith("_")}

    for c in candidates[1:]:
        for fname in base.__slots__:
            if fname.startswith("_"):
                continue
            bv = getattr(base, fname)
            cv = getattr(c, fname)
            if isinstance(bv, bool) and isinstance(cv, bool):
                new_data[fname] = bv or cv
            elif cv not in {None, "", Path("")}:
                new_data[fname] = cv

    return PipelineConfig(**new_data)
