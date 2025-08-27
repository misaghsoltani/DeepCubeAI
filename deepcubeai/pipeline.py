"""DeepCubeAI pipeline runner.

This constructs defaults, resolves file paths, and invokes existing module entry points.
"""

from __future__ import annotations

from argparse import (
    SUPPRESS,
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    Namespace,
    RawTextHelpFormatter,
    _ActionsContainer,
    _ArgumentGroup,
    _SubParsersAction,
)
from collections.abc import Callable
import json
from pathlib import Path
import sys
import time
from typing import Any, cast

from deepcubeai.config import EnvModelPaths, HeurModelPaths, OfflineDataPaths, PipelineConfig, merge_config
from deepcubeai.exceptions import ConfigError
from deepcubeai.utils import data_utils, dist_utils, env_utils


def _to_bool(val: str | bool | None) -> bool:
    """Consistently coerce CLI/env style truthy values to bool."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


# stage runners


def run_gen_offline(cfg: PipelineConfig) -> None:
    """Generate offline train/val datasets."""
    train_eps, val_eps, _ = cfg.derive_offline_counts()
    if cfg.num_offline_steps is None:
        raise ConfigError("--num_offline_steps is required for gen_offline")
    s_tr, n_tr, s_val, n_val, _, _ = cfg.derive_seeds(train_eps)

    offline: OfflineDataPaths = cfg.offline_paths()
    offline.offline_dir.mkdir(parents=True, exist_ok=True)

    # Use programmatic API instead of emulating CLI
    from deepcubeai.scripts.generate_offline_data import GenerateOfflineConfig, run_generate_offline  # noqa: PLC0415

    print("Generating offline training data")
    cfg_train = GenerateOfflineConfig(
        env=cfg.env,
        num_episodes=train_eps,
        num_steps=cfg.num_offline_steps,
        data_file=str(offline.train),
        num_procs=cfg.num_cpus,
        start_level=s_tr,
        num_levels=n_tr,
    )
    run_generate_offline(cfg_train)

    print("\nGenerating offline validation data")
    cfg_val = GenerateOfflineConfig(
        env=cfg.env,
        num_episodes=val_eps,
        num_steps=cfg.num_offline_steps,
        data_file=str(offline.val),
        num_procs=cfg.num_cpus,
        start_level=s_val,
        num_levels=n_val,
    )
    run_generate_offline(cfg_val)


def run_gen_env_test(cfg: PipelineConfig) -> None:
    """Generate offline test data for environment model evaluation."""
    _, _, test_eps = cfg.derive_offline_counts()
    if cfg.num_offline_steps is None:
        raise ConfigError("--num_offline_steps is required for gen_env_test")
    _, _, _, _, s_test, n_test = cfg.derive_seeds(test_eps)

    offline: OfflineDataPaths = cfg.offline_paths()
    offline.env_test_dir.mkdir(parents=True, exist_ok=True)

    from deepcubeai.scripts.generate_offline_data import GenerateOfflineConfig, run_generate_offline  # noqa: PLC0415

    print("Generating environment model offline test data")
    cfg_test = GenerateOfflineConfig(
        env=cfg.env,
        num_episodes=test_eps,
        num_steps=cfg.num_offline_steps,
        data_file=str(offline.env_test),
        num_procs=cfg.num_cpus,
        start_level=s_test,
        num_levels=n_test,
    )
    run_generate_offline(cfg_test)


def run_gen_search_test(cfg: PipelineConfig) -> None:
    """Generate start/goal states for search evaluation."""
    _, _, test_eps = cfg.derive_offline_counts()
    num_steps: int = cfg.num_offline_steps if cfg.num_offline_steps is not None else -1
    s_tr, _, _, _, s_test, _ = cfg.derive_seeds(test_eps)

    offline: OfflineDataPaths = cfg.offline_paths()
    offline.search_test_dir.mkdir(parents=True, exist_ok=True)

    from deepcubeai.scripts.generate_search_test_data import (  # noqa: PLC0415
        GenerateSearchTestConfig,
        run_generate_search_test,
    )

    print("Generating search test data")
    cfg_search = GenerateSearchTestConfig(
        env=cfg.env,
        num_episodes=test_eps,
        data_file=str(offline.search_test),
        num_steps=num_steps,
        start_level=s_tr if s_test == -1 else s_test,
        reverse=bool(getattr(cfg, "reverse", False) and cfg.env.lower() == "cube3"),
    )
    run_generate_search_test(cfg_search)


def run_train_model_disc(cfg: PipelineConfig) -> None:
    """Train the discrete environment model."""
    offline: OfflineDataPaths = cfg.offline_paths()
    env_paths: EnvModelPaths = cfg.env_model_paths()
    env_paths.model_dir.mkdir(parents=True, exist_ok=True)

    schedules: list[dict[str, float | int]] = [
        {"env_coeff": 0.0001, "max_itrs": 40000, "lr": 0.001},
        {"env_coeff": 0.001, "max_itrs": 60000, "lr": 0.001},
        {"env_coeff": 0.01, "max_itrs": 80000, "lr": 0.001},
        {"env_coeff": 0.1, "max_itrs": 100000, "lr": 0.001},
        {"env_coeff": 0.5, "max_itrs": 120000, "lr": 0.001},
        {"env_coeff": 0.5, "max_itrs": 140000, "lr": 0.0001},
        {"env_coeff": 0.5, "max_itrs": 160000, "lr": 0.00001},
        {"env_coeff": 0.5, "max_itrs": 180000, "lr": 0.000001},
    ]
    print("Training model (discrete)")

    from deepcubeai.training.train_env_disc import TrainEnvDiscConfig, run_train_env_disc  # noqa: PLC0415

    for sched in schedules:
        disc_cfg = TrainEnvDiscConfig(
            env=cfg.env,
            train_data=str(offline.train),
            val_data=str(offline.val),
            nnet_name=str(cfg.env_model_name or "env_model"),
            save_dir=str(env_paths.save_dir),
            env_coeff=float(sched["env_coeff"]),
            lr=float(sched["lr"]),
            max_itrs=int(sched["max_itrs"]),
            batch_size=int(cfg.env_batch_size),
        )
        run_train_env_disc(disc_cfg)


def run_train_model_cont(cfg: PipelineConfig) -> None:
    """Train the continuous environment model."""
    offline: OfflineDataPaths = cfg.offline_paths()
    env_paths: EnvModelPaths = cfg.env_model_paths()
    env_paths.model_dir.mkdir(parents=True, exist_ok=True)

    schedules: list[dict[str, float | int]] = [
        {"max_itrs": 120000, "lr": 0.001},
        {"max_itrs": 140000, "lr": 0.0001},
        {"max_itrs": 160000, "lr": 0.00001},
        {"max_itrs": 180000, "lr": 0.000001},
    ]
    print("Training model (continuous)")
    from deepcubeai.training.train_env_cont import TrainEnvContConfig, run_train_env_cont  # noqa: PLC0415

    for sched in schedules:
        cont_cfg = TrainEnvContConfig(
            env=cfg.env,
            train_data=str(offline.train),
            val_data=str(offline.val),
            nnet_name=str(cfg.env_model_name or "env_model"),
            save_dir=str(env_paths.save_dir),
            lr=float(sched["lr"]),
            max_itrs=int(sched["max_itrs"]),
            batch_size=int(cfg.env_batch_size),
            num_steps=1,
        )
        run_train_env_cont(cont_cfg)


def run_test_model(cfg: PipelineConfig, continuous: bool = False) -> None:
    """Evaluate an environment model on offline test data."""
    offline: OfflineDataPaths = cfg.offline_paths()
    env_paths: EnvModelPaths = cfg.env_model_paths()
    env_dir: Path = env_paths.model_dir
    if continuous:
        print("Testing model (continuous)")
        from deepcubeai.scripts.test_model_cont import TestModelContConfig, run_test_model_cont  # noqa: PLC0415

        tcfg_cont = TestModelContConfig(
            env=cfg.env, data=str(offline.env_test), env_dir=str(env_dir), print_interval=int(cfg.print_interval)
        )
        run_test_model_cont(tcfg_cont)
    else:
        print("Testing model (discrete)")
        from deepcubeai.scripts.test_model_disc import TestModelDiscConfig, run_test_model_disc  # noqa: PLC0415

        tcfg_disc = TestModelDiscConfig(
            env=cfg.env, data=str(offline.env_test), env_dir=str(env_dir), print_interval=int(cfg.print_interval)
        )
        run_test_model_disc(tcfg_disc)


def run_encode_offline(cfg: PipelineConfig) -> None:
    """Encode offline train/val datasets using the trained env model."""
    offline: OfflineDataPaths = cfg.offline_paths()
    env_dir: Path = cfg.env_model_paths().model_dir
    print("Encoding offline training data")

    from deepcubeai.scripts.encode_offline_data import EncodeOfflineConfig, run_encode_offline  # noqa: PLC0415

    cfg_train = EncodeOfflineConfig(
        env=cfg.env, env_dir=str(env_dir), data=str(offline.train), data_enc=str(offline.train_enc)
    )
    run_encode_offline(cfg_train)

    print("\nEncoding offline validation data")
    cfg_val = EncodeOfflineConfig(
        env=cfg.env, env_dir=str(env_dir), data=str(offline.val), data_enc=str(offline.val_enc)
    )
    run_encode_offline(cfg_val)


def run_train_heur(cfg: PipelineConfig) -> None:
    """Train the heuristic network (optionally single-process distributed)."""
    offline: OfflineDataPaths = cfg.offline_paths()
    env_dir: Path = cfg.env_model_paths().model_dir
    heur_paths: HeurModelPaths = cfg.heur_model_paths()

    required: dict[str, str | int | float | None] = {
        "heur_nnet_name": cfg.heur_nnet_name,
        "per_eq_tol": cfg.per_eq_tol,
        "heur_batch_size": cfg.heur_batch_size,
        "states_per_update": cfg.states_per_update,
        "max_solve_steps": cfg.max_solve_steps,
        "start_steps": cfg.start_steps,
        "goal_steps": cfg.goal_steps,
        "num_test": cfg.num_test,
    }
    missing: list[str] = [k for k, v in required.items() if v is None]
    if missing:
        raise ConfigError(f"Missing required arguments for train_heur: {', '.join(missing)}")

    # Narrow types after validation
    per_eq_tol_v: float = cast(float, cfg.per_eq_tol)
    heur_batch_size_v: int = cast(int, cfg.heur_batch_size)
    states_per_update_v: int = cast(int, cfg.states_per_update)
    max_solve_steps_v: int = cast(int, cfg.max_solve_steps)
    start_steps_v: int = cast(int, cfg.start_steps)
    goal_steps_v: int = cast(int, cfg.goal_steps)
    num_test_v: int = cast(int, cfg.num_test)

    if cfg.use_dist:  # distributed GPU pipeline
        print("Training heuristic function (distributed GPU-only pipeline)")
        from deepcubeai.training.qlearning_dist import QLearningDistConfig, run_qlearning_dist  # noqa: PLC0415

        qcfg_dist = QLearningDistConfig(
            env=cfg.env,
            env_model=str(env_dir),
            train=str(offline.train_enc),
            val=str(offline.val_enc),
            per_eq_tol=per_eq_tol_v,
            batch_size=heur_batch_size_v,
            states_per_update=states_per_update_v,
            max_solve_steps=max_solve_steps_v,
            start_steps=start_steps_v,
            goal_steps=goal_steps_v,
            num_test=num_test_v,
            nnet_name=str(cfg.heur_nnet_name),
            save_dir=str(heur_paths.save_dir),
            amp=cfg.amp,
            compile=cfg.compile,
            compile_all_models=cfg.compile_all_models,
            lr=(cfg.lr or 1e-3),
            lr_d=(cfg.lr_d or 0.9999993),
            max_itrs=(cfg.max_itrs or 1_000_000),
            update_nnet_batch_size=(cfg.update_nnet_batch_size or heur_batch_size_v),
            ring=(cfg.ring or 3),
            seed=(cfg.seed or 0),
            debug=cfg.debug,
        )
        run_qlearning_dist(qcfg_dist)
    else:
        print("Training heuristic function")
        from deepcubeai.training.qlearning import QLearningConfig, run_qlearning  # noqa: PLC0415

        qcfg_single = QLearningConfig(
            env=cfg.env,
            env_model=str(env_dir),
            train=str(offline.train_enc),
            val=str(offline.val_enc),
            per_eq_tol=per_eq_tol_v,
            batch_size=heur_batch_size_v,
            states_per_update=states_per_update_v,
            max_solve_steps=max_solve_steps_v,
            start_steps=start_steps_v,
            goal_steps=goal_steps_v,
            num_test=num_test_v,
            nnet_name=str(cfg.heur_nnet_name),
            save_dir=str(heur_paths.save_dir),
        )
        run_qlearning(qcfg_single)


def run_qstar(cfg: PipelineConfig) -> None:
    """Run Q* search on the generated search test states."""
    offline: OfflineDataPaths = cfg.offline_paths()
    env_dir: Path = cfg.env_model_paths().model_dir
    heur_dir: Path = cfg.heur_model_paths().current_dir

    default_results: str = (
        f"model={cfg.env_model_name}__heur={cfg.heur_nnet_name}_QStar_results/path_cost_weight={cfg.qstar_weight or 1}"
    )
    results_dir = Path("deepcubeai/results") / cfg.env / (cfg.qstar_results_dir or default_results)
    results_dir.mkdir(parents=True, exist_ok=True)

    states_path: str = cfg.search_test_data or str(offline.search_test)
    print("Doing Q* search")
    from deepcubeai.search_methods.qstar_imag import QStarImagConfig, run_qstar_imag  # noqa: PLC0415

    qcfg = QStarImagConfig(
        env=cfg.env,
        states=states_path,
        env_model=str(env_dir),
        results_dir=str(results_dir),
        per_eq_tol=float(cfg.per_eq_tol or 0.0),
        weight=float(cfg.qstar_weight or 1.0),
        h_weight=float(cfg.qstar_h_weight or 1.0),
        heur=str(heur_dir) if (cfg.qstar_h_weight or 1.0) != 0.0 else None,
        batch_size=int(cfg.qstar_batch_size or 1),
        save_imgs=_to_bool(cfg.save_imgs),
    )
    run_qstar_imag(qcfg)


def run_ucs(cfg: PipelineConfig) -> None:
    """Run Uniform-Cost Search (Q* with weight=1, h_weight=0)."""
    offline: OfflineDataPaths = cfg.offline_paths()
    env_dir: Path = cfg.env_model_paths().model_dir

    default_results: str = f"model={cfg.env_model_name}_UCS_results"
    results_dir = Path("deepcubeai/results") / cfg.env / (cfg.ucs_results_dir or default_results)
    results_dir.mkdir(parents=True, exist_ok=True)

    states_path: str = cfg.search_test_data or str(offline.search_test)
    print("Doing uniform-cost search")
    from deepcubeai.search_methods.ucs_imag import UCSImagConfig, run_ucs_imag  # noqa: PLC0415

    ucfg = UCSImagConfig(
        env=cfg.env,
        states=states_path,
        env_model=str(env_dir),
        results_dir=str(results_dir),
        per_eq_tol=float(cfg.per_eq_tol or 0.0),
        batch_size=int(cfg.ucs_batch_size or 1),
        save_imgs=_to_bool(cfg.save_imgs),
    )
    run_ucs_imag(ucfg)


def run_gbfs(cfg: PipelineConfig) -> None:
    """Run Greedy Best-First Search on the search test states."""
    offline: OfflineDataPaths = cfg.offline_paths()
    env_dir: Path = cfg.env_model_paths().model_dir
    heur_dir: Path = cfg.heur_model_paths().current_dir

    default_results: str = f"model={cfg.env_model_name}__heur={cfg.heur_nnet_name}_GBFS_results"
    results_dir = Path("deepcubeai/results") / cfg.env / (cfg.gbfs_results_dir or default_results)
    results_dir.mkdir(parents=True, exist_ok=True)
    states_path: str = cfg.search_test_data or str(offline.search_test)

    print("Doing Greedy Best First Search")
    from deepcubeai.search_methods.gbfs_imag import GBFSImagConfig, run_gbfs_imag  # noqa: PLC0415

    gbcfg = GBFSImagConfig(
        env=cfg.env,
        states=states_path,
        heur=str(heur_dir),
        env_model=str(env_dir),
        search_itrs=int(cfg.search_itrs or 100),
        per_eq_tol=float(cfg.per_eq_tol or 0.0),
        results_dir=str(results_dir),
        nnet_batch_size=None,
        debug=False,
    )
    run_gbfs_imag(gbcfg)


def run_disc_vs_cont(cfg: PipelineConfig) -> None:
    """Compare discrete vs. continuous env models' MSE and save plots."""
    offline: OfflineDataPaths = cfg.offline_paths()
    model_test_data: str = cfg.model_test_data_dir or str(offline.env_test)

    # env model dirs must be provided or derivable
    env_dir_disc: str | None = cfg.env_model_dir_disc
    env_dir_cont: str | None = cfg.env_model_dir_cont
    if env_dir_disc is None and cfg.env_model_name is not None:
        env_dir_disc = str(cfg.env_model_paths().model_dir)
    if env_dir_cont is None:
        raise ConfigError("--env_model_dir_cont is required for disc_vs_cont stage")

    print("MSE Comparison (Discrete vs Continuous)")
    from deepcubeai.extra.plot_disc_vs_cont import PlotDiscVsContConfig, run_plot_disc_vs_cont  # noqa: PLC0415

    pdccfg = PlotDiscVsContConfig(
        env=cfg.env,
        model_test_data=model_test_data,
        env_model_dir_disc=str(env_dir_disc),
        env_model_dir_cont=str(env_dir_cont),
        num_episodes=int(cfg.num_episodes) if cfg.num_episodes is not None else -1,
        num_steps=int(cfg.num_steps) if cfg.num_steps is not None else -1,
        save_dir=str(cfg.save_dir) if cfg.save_dir is not None else None,
        print_interval=int(cfg.print_interval),
    )
    run_plot_disc_vs_cont(pdccfg)


def run_visualize_data(cfg: PipelineConfig) -> None:
    """Render sample images from offline datasets for inspection."""
    offline: OfflineDataPaths = cfg.offline_paths()
    base_save: Path = offline.data_sample_img_dir
    base_save.mkdir(parents=True, exist_ok=True)

    print("Saving offline data as images")
    from deepcubeai.extra.offline_data_viz import OfflineDataVizConfig, run_offline_data_viz  # noqa: PLC0415

    vcfg = OfflineDataVizConfig(
        env=cfg.env,
        train_data=str(offline.train),
        val_data=str(offline.val),
        num_train_trajs=int(cfg.num_train_trajs_viz or 8),
        num_train_steps=int(cfg.num_train_steps_viz or 2),
        num_val_trajs=int(cfg.num_val_trajs_viz or 8),
        num_val_steps=int(cfg.num_val_steps_viz or 2),
        save_imgs=str(base_save),
    )
    run_offline_data_viz(vcfg)


def run_compare_solutions_stage(cfg: PipelineConfig) -> None:
    """Compare two solution pickles and print stats."""
    from deepcubeai.scripts.compare_solutions import CompareSolutionsConfig, run_compare_solutions  # noqa: PLC0415

    # Expect soln1/soln2 to be provided via CLI or config file
    if not hasattr(cfg, "soln1") or not hasattr(cfg, "soln2"):
        raise ConfigError("--soln1 and --soln2 are required for compare_solutions stage")

    cfa: Any = cfg
    soln1 = cast(str, cfa.soln1)
    soln2 = cast(str, cfa.soln2)

    ccfg = CompareSolutionsConfig(soln1=soln1, soln2=soln2)
    run_compare_solutions(ccfg)


# CLI


def build_parser() -> ArgumentParser:
    """Create a hierarchical, stage-based CLI."""

    class Formatter(ArgumentDefaultsHelpFormatter, RawTextHelpFormatter):
        pass

    # Discover available environments from the environment registry
    try:
        env_choices: list[str] = env_utils.list_environments()
        if not env_choices:
            raise RuntimeError("no envs discovered")
    except Exception:
        env_choices = ["cube3", "sokoban", "digitjump", "iceslider"]

    parser: ArgumentParser = ArgumentParser(
        prog="`python -m deepcubeai` (or `deepcubeai`)",
        description=(
            "DeepCubeAI pipeline.\n\n"
            "Pick a stage, then pass the stage-specific options. Typical flow:\n"
            "  1) gen_offline → 2) train_model_disc (Discrete) or train_model_cont (Continuous) →\n"
            "  3) encode_offline → 4) train_heur → 5) gen_search_test → 6) qstar/ucs/gbfs\n\n"
            "Use `python -m deepcubeai <stage> -h` or `deepcubeai -h` to see focused help for that stage."
        ),
        formatter_class=Formatter,
    )
    parser.add_argument("--config-file", help="Path to JSON config file", default=None)

    subparsers: _SubParsersAction[ArgumentParser] = parser.add_subparsers(
        title="stages", description="Choose one of the pipeline stages", metavar="<stage>", dest="stage", required=True
    )

    # small helper to create optional args
    def opt(sp: _ActionsContainer, *flags: str, **kwargs: Any) -> None:
        kwargs.setdefault("default", SUPPRESS)
        sp.add_argument(*flags, **kwargs)

    def add_env(sp: _ActionsContainer) -> None:
        sp.add_argument("--env", required=True, choices=env_choices, help="Which environment to use")

    def add_data_common(sp: _ActionsContainer) -> None:
        group: _ArgumentGroup = sp.add_argument_group("data paths")
        opt(group, "--data_dir", help="Base data directory name under deepcubeai/data (defaults to ENV)")
        opt(group, "--data_file_name", help="Prefix/name used when composing data file names")

    def add_seeding(sp: _ActionsContainer) -> None:
        group: _ArgumentGroup = sp.add_argument_group("seeding across levels")
        opt(group, "--start_level", type=int, help="Starting level/seed for data generation (-1=random)")
        opt(group, "--num_levels", type=int, help="How many levels/seeds to generate")

    def add_env_model_common(sp: _ActionsContainer) -> None:
        group: _ArgumentGroup = sp.add_argument_group("environment model")
        opt(group, "--env_model_name", help="Saved model name (under deepcubeai/saved_env_models)")
        opt(group, "--env_batch_size", type=int, help="Batch size for env model training", default=SUPPRESS)
        opt(group, "--print_interval", type=int, help="Status print interval", default=SUPPRESS)

    # gen_offline
    gen_offline: ArgumentParser = subparsers.add_parser(
        "gen_offline",
        help="Generate offline train/val datasets",
        description=(
            "Generate offline training and validation trajectories for a given environment.\n"
            "If neither --num_train_eps nor --num_val_eps is provided, defaults to 9000/1000."
        ),
        formatter_class=Formatter,
    )
    add_env(gen_offline)
    add_data_common(gen_offline)
    add_seeding(gen_offline)
    group_go: _ArgumentGroup = gen_offline.add_argument_group("generation")
    group_go.add_argument("--num_offline_steps", type=int, required=True, help="Steps per episode to generate")
    opt(group_go, "--num_cpus", type=int, help="Parallel workers for generation", default=1)
    group_split: _ArgumentGroup = gen_offline.add_argument_group("episode counts")
    opt(group_split, "--num_train_eps", type=int, help="Number of training episodes (default uses 90/10 split)")
    opt(group_split, "--num_val_eps", type=int, help="Number of validation episodes (default uses 90/10 split)")

    # gen_env_test
    gen_env_test: ArgumentParser = subparsers.add_parser(
        "gen_env_test",
        help="Generate offline test data for env model evaluation",
        description="Generate held-out environment trajectories for model evaluation.",
        formatter_class=Formatter,
    )
    add_env(gen_env_test)
    add_data_common(gen_env_test)
    add_seeding(gen_env_test)
    group_get: _ArgumentGroup = gen_env_test.add_argument_group("generation")
    group_get.add_argument("--num_offline_steps", type=int, required=True, help="Steps per episode to generate")
    opt(group_get, "--num_cpus", type=int, help="Parallel workers for generation", default=1)
    opt(gen_env_test, "--num_test_eps", type=int, help="Number of test episodes", default=100)

    # gen_search_test
    gen_search_test: ArgumentParser = subparsers.add_parser(
        "gen_search_test",
        help="Generate start/goal states for search evaluation",
        description="Create start-goal pairs used by qstar/ucs/gbfs stages.",
        formatter_class=Formatter,
    )
    add_env(gen_search_test)
    add_data_common(gen_search_test)
    add_seeding(gen_search_test)
    opt(gen_search_test, "--num_offline_steps", type=int, help="Steps per episode (optional; default -1)")
    opt(gen_search_test, "--num_test_eps", type=int, help="Number of search test pairs", default=100)
    # cube3-only helper
    gen_search_test.add_argument(
        "--reverse",
        action="store_true",
        help="cube3 only: start from canonical goal and set goals to scrambled states",
        default=False,
    )

    # train_model_disc (discrete)
    train_model_disc: ArgumentParser = subparsers.add_parser(
        "train_model_disc",
        help="Train the discrete environment model",
        description="Train the discrete env model.",
        formatter_class=Formatter,
    )
    add_env(train_model_disc)
    add_data_common(train_model_disc)
    add_env_model_common(train_model_disc)

    # train_model_cont (continuous)
    train_model_cont: ArgumentParser = subparsers.add_parser(
        "train_model_cont",
        help="Train the continuous environment model",
        description="Train the continuous env model.",
        formatter_class=Formatter,
    )
    add_env(train_model_cont)
    add_data_common(train_model_cont)
    add_env_model_common(train_model_cont)

    # test_model (discrete)
    test_model: ArgumentParser = subparsers.add_parser(
        "test_model",
        help="Evaluate a discrete env model",
        description="Evaluate a discrete env model on env test data (see gen_env_test).",
        formatter_class=Formatter,
    )
    add_env(test_model)
    add_data_common(test_model)
    add_env_model_common(test_model)

    # test_model_cont (continuous)
    test_model_cont: ArgumentParser = subparsers.add_parser(
        "test_model_cont",
        help="Evaluate a continuous env model",
        description="Evaluate a continuous env model on env test data (see gen_env_test).",
        formatter_class=Formatter,
    )
    add_env(test_model_cont)
    add_data_common(test_model_cont)
    add_env_model_common(test_model_cont)

    # encode_offline
    encode_offline: ArgumentParser = subparsers.add_parser(
        "encode_offline",
        help="Encode offline train/val with the trained env model",
        description="Produce encoded datasets used for heuristic training.",
        formatter_class=Formatter,
    )
    add_env(encode_offline)
    add_data_common(encode_offline)
    add_env_model_common(encode_offline)

    # train_heur
    train_heur: ArgumentParser = subparsers.add_parser(
        "train_heur",
        help="Train the heuristic network",
        description="Train heuristic network using encoded offline data (see encode_offline).",
        formatter_class=Formatter,
    )
    add_env(train_heur)
    add_data_common(train_heur)
    add_env_model_common(train_heur)
    group_hreq: _ArgumentGroup = train_heur.add_argument_group("required")
    group_hreq.add_argument("--heur_nnet_name", required=True, help="Name of neural network")

    group_hreq.add_argument("--heur_batch_size", required=True, type=int, help="Batch size")
    group_hreq.add_argument(
        "--states_per_update",
        required=True,
        type=int,
        help="How many states to train on before checking if target network should be updated",
    )
    group_hreq.add_argument(
        "--max_solve_steps",
        required=True,
        type=int,
        help="Number of steps to take when trying to solve training states with greedy best-first"
        " search (GBFS). Each state encountered when solving is added to the training set. "
        "Number of steps starts at 1 and is increased every update until the maximum number "
        "is reached. Increasing this number can make the cost-to-go function more robust by "
        "exploring more of the state space.",
    )
    group_hreq.add_argument(
        "--start_steps",
        required=True,
        type=int,
        help="Maximum number of steps to take from offline states to generate start states",
    )
    group_hreq.add_argument(
        "--goal_steps",
        required=True,
        type=int,
        help="Maximum number of steps to take from the start states to generate goal states",
    )
    group_hreq.add_argument("--num_test", type=int, default=1000, help="Number of test states.")
    group_hreq.add_argument(
        "--per_eq_tol",
        required=True,
        type=float,
        help="Percent of latent state elements that need to be equal to declare equal",
    )
    opt(train_heur, "--use_dist", action="store_true", help="Use distributed GPU-only pipeline (FSDP2)")
    opt(train_heur, "--amp", action="store_true", help="Enable automatic mixed precision")
    opt(train_heur, "--compile", action="store_true", help="Enable torch.compile for the DQN model")
    opt(
        train_heur,
        "--compile_all_models",
        action="store_true",
        help="If set, apply torch.compile to all local (unsharded) model replicas used for data generation "
        "and evaluation (env_model_gen, dqn_gen, dqn_targ_gen, env_model_eval, dqn_eval). For "
        "the sharded training DQN and env_model use --compile.",
    )
    opt(train_heur, "--lr", type=float, help="Initial learning rate")
    opt(
        train_heur,
        "--lr_d",
        type=float,
        help="Learning rate decay for every iteration. Learning rate is decayed according to: lr * (lr_d ^ itr)",
    )
    opt(train_heur, "--max_itrs", type=int, help="Maximum number of training iterations")
    opt(
        train_heur,
        "--update_nnet_batch_size",
        type=int,
        help="Batch size of each nnet used for each process update. Make smaller if running out of memory.",
    )
    opt(train_heur, "--ring", type=int, help="Ring buffer slots")
    opt(train_heur, "--seed", type=int, help="Random seed for reproducibility")
    opt(train_heur, "--debug", action="store_true", help="Enable debugging output")

    # qstar
    qstar: ArgumentParser = subparsers.add_parser(
        "qstar",
        help="Run Q* search",
        description="Run Q* on generated search test states (see gen_search_test).",
        formatter_class=Formatter,
    )
    add_env(qstar)
    add_data_common(qstar)
    add_env_model_common(qstar)
    group_qs: _ArgumentGroup = qstar.add_argument_group("search")
    opt(group_qs, "--heur_nnet_name", help="Heuristic model name (to load current checkpoint)")
    opt(group_qs, "--qstar_batch_size", type=int, help="Batch size for Q* search")
    opt(group_qs, "--qstar_weight", type=float, help="Path cost weight (w)")
    opt(group_qs, "--qstar_h_weight", type=float, help="Heuristic weight (h)")
    opt(group_qs, "--qstar_results_dir", help="Override default results dir name")
    opt(
        group_qs,
        "--per_eq_tol",
        type=float,
        help="Percent of latent state elements that need to be equal to declare equal",
    )
    opt(group_qs, "--search_test_data", help="Path to search test data (defaults to generated path)")
    opt(group_qs, "--save_imgs", dest="save_imgs", action="store_true", help="Save images for solutions")
    opt(group_qs, "--no_save_imgs", dest="save_imgs", action="store_false", help="Do not save images")

    # ucs
    ucs: ArgumentParser = subparsers.add_parser(
        "ucs",
        help="Run Uniform-Cost Search (Q* with w=1, h=0)",
        description="Run UCS on generated search test states (see gen_search_test).",
        formatter_class=Formatter,
    )
    add_env(ucs)
    add_data_common(ucs)
    add_env_model_common(ucs)
    group_ucs: _ArgumentGroup = ucs.add_argument_group("search")
    opt(group_ucs, "--ucs_batch_size", type=int, help="Batch size for UCS")
    opt(group_ucs, "--ucs_results_dir", help="Override default results dir name")
    opt(
        group_ucs,
        "--per_eq_tol",
        type=float,
        help="Percent of latent state elements that need to be equal to declare equal",
    )
    opt(group_ucs, "--search_test_data", help="Path to search test data (defaults to generated path)")
    opt(group_ucs, "--save_imgs", dest="save_imgs", action="store_true", help="Save images for solutions")
    opt(group_ucs, "--no_save_imgs", dest="save_imgs", action="store_false", help="Do not save images")

    # gbfs
    gbfs: ArgumentParser = subparsers.add_parser(
        "gbfs",
        help="Run Greedy Best-First Search",
        description="Run GBFS on generated search test states (see gen_search_test).",
        formatter_class=Formatter,
    )
    add_env(gbfs)
    add_data_common(gbfs)
    add_env_model_common(gbfs)
    group_gb: _ArgumentGroup = gbfs.add_argument_group("search")
    opt(group_gb, "--heur_nnet_name", help="Heuristic model name (to load current checkpoint)")
    opt(group_gb, "--search_itrs", type=int, help="Search iterations (per state)")
    opt(group_gb, "--gbfs_results_dir", help="Override default results dir name")
    opt(group_gb, "--search_test_data", help="Path to search test data (defaults to generated path)")
    opt(
        group_gb,
        "--per_eq_tol",
        type=float,
        help="Percent of latent state elements that need to be equal to declare equal",
    )

    # disc_vs_cont
    dvc: ArgumentParser = subparsers.add_parser(
        "disc_vs_cont",
        help="Compare discrete vs continuous env models (MSE plots)",
        description="Compare MSE of discrete vs continuous env models and save plots.",
        formatter_class=Formatter,
    )
    add_env(dvc)
    add_data_common(dvc)
    add_env_model_common(dvc)
    group_dvc: _ArgumentGroup = dvc.add_argument_group("compare")
    opt(group_dvc, "--model_test_data_dir", help="Path to model test data (defaults to generated path)")
    opt(group_dvc, "--env_model_dir_disc", help="Path to discrete env model dir (defaults from --env_model_name)")
    group_dvc.add_argument("--env_model_dir_cont", required=True, help="Path to continuous env model dir")
    opt(group_dvc, "--num_episodes", type=int, help="Limit episodes (-1=all)")
    opt(group_dvc, "--num_steps", type=int, help="Limit steps per episode (-1=all)")
    opt(group_dvc, "--save_dir", help="Directory to save plots (defaults to CWD)")

    # visualize_data
    viz: ArgumentParser = subparsers.add_parser(
        "visualize_data",
        help="Render and save sample offline data images",
        description="Quick visual sanity-check of offline datasets.",
        formatter_class=Formatter,
    )
    add_env(viz)
    add_data_common(viz)
    group_v: _ArgumentGroup = viz.add_argument_group("sampling")
    opt(group_v, "--num_train_trajs_viz", type=int, help="Train trajectories to render", default=8)
    opt(group_v, "--num_train_steps_viz", type=int, help="Steps per train trajectory", default=2)
    opt(group_v, "--num_val_trajs_viz", type=int, help="Validation trajectories to render", default=8)
    opt(group_v, "--num_val_steps_viz", type=int, help="Steps per validation trajectory", default=2)

    # compare_solutions
    cmp: ArgumentParser = subparsers.add_parser(
        "compare_solutions",
        help="Compare two solution result pickles",
        description="Print stats comparing two results.pkl files.",
        formatter_class=Formatter,
    )
    cmp.add_argument("--soln1", required=True, help="Path to the first solution file")
    cmp.add_argument("--soln2", required=True, help="Path to the second solution file")

    # envs: list available environments
    envs_p: ArgumentParser = subparsers.add_parser(
        "envs",
        help="List known environments",
        description="Print the known environment keys.",
        formatter_class=Formatter,
    )
    envs_p.add_argument("--json", dest="as_json", action="store_true", help="Print JSON output", default=False)
    envs_p.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Include module/attr metadata for each entry",
        default=False,
    )
    # envs-add: add a user environment entry to the file-backed registry
    envs_add: ArgumentParser = subparsers.add_parser(
        "envs-add",
        help="Add a user environment to the registry",
        description="Register a third-party or local environment by module path and class name.",
        formatter_class=Formatter,
    )
    envs_add.add_argument("--key", required=True, help="Registry key for the environment (e.g. 'myenv')")
    envs_add.add_argument("--module", required=True, help="Importable module path (e.g. mypkg.mymodule)")
    envs_add.add_argument("--attr", required=False, help="Attribute/class name in the module (optional)")
    # envs-remove: remove a user-added environment
    envs_remove: ArgumentParser = subparsers.add_parser(
        "envs-remove",
        help="Remove a user environment from the registry",
        description="Remove an environment previously added via envs-add (builtin entries cannot be removed).",
        formatter_class=Formatter,
    )
    envs_remove.add_argument("--key", required=True, help="Registry key to remove")

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for running the pipeline."""
    parser: ArgumentParser = build_parser()
    args: Namespace = parser.parse_args(argv)

    # Merge precedence: file/env < CLI
    cfg_file: PipelineConfig = (
        PipelineConfig.from_json(args.config_file) if getattr(args, "config_file", None) else PipelineConfig()
    )
    cfg_env: PipelineConfig = PipelineConfig.from_env()

    # Build CLI config with only provided args so dataclass defaults apply.
    cli_kwargs: dict[str, Any] = {}
    for name in PipelineConfig.__dataclass_fields__:
        if hasattr(args, name):
            cli_kwargs[name] = getattr(args, name)
    if "stage" not in cli_kwargs and getattr(args, "stage", None):
        cli_kwargs["stage"] = args.stage
    cfg_cli = PipelineConfig(**cli_kwargs)

    cfg: PipelineConfig = merge_config(cfg_file, cfg_env, cfg_cli)

    def format_elapsed(secs: float) -> str:
        """Format elapsed time as D-H:M:S:MS."""
        ms_total = int(round(secs * 1000.0))
        days, rem = divmod(ms_total, 86_400_000)
        hours, rem = divmod(rem, 3_600_000)
        minutes, rem = divmod(rem, 60_000)
        seconds, millis = divmod(rem, 1000)
        return f"{days}-{hours}:{minutes}:{seconds}:{millis}"

    def run_with_timing(stage_name: str, fn: Callable[[], None]) -> None:
        """Run a stage function with timing.

        Args:
            stage_name: Name of the stage for logging.
            fn: The function to run for the stage.
        """
        start: float = time.perf_counter()
        ok: bool = True
        try:
            fn()
        except Exception:
            ok = False
            raise
        finally:
            elapsed = time.perf_counter() - start
            print(
                f"{'-' * 72}\n{'-' * 72}\n"
                f"Stage '{stage_name}' finished with {'SUCCESS' if ok else 'FAILED'}.\n"
                f"Elapsed (D-H:M:S:MS): {format_elapsed(elapsed)}\n"
                f"{'-' * 72}\n{'-' * 72}\n"
            )

    timed_stages: dict[str, Callable[[], None]] = {
        "gen_offline": lambda: run_gen_offline(cfg),
        "gen_env_test": lambda: run_gen_env_test(cfg),
        "gen_search_test": lambda: run_gen_search_test(cfg),
        "train_model_disc": lambda: run_train_model_disc(cfg),
        "train_model_cont": lambda: run_train_model_cont(cfg),
        "test_model": lambda: run_test_model(cfg, continuous=False),
        "test_model_cont": lambda: run_test_model(cfg, continuous=True),
        "encode_offline": lambda: run_encode_offline(cfg),
        "train_heur": lambda: run_train_heur(cfg),
        "qstar": lambda: run_qstar(cfg),
        "ucs": lambda: run_ucs(cfg),
        "gbfs": lambda: run_gbfs(cfg),
        "disc_vs_cont": lambda: run_disc_vs_cont(cfg),
        "visualize_data": lambda: run_visualize_data(cfg),
        "compare_solutions": lambda: run_compare_solutions_stage(cfg),
    }

    if cfg.stage in timed_stages:
        # Attempt to map each stage to the same output path the stage's module
        # would have used when run standalone. Only create a pipeline-level
        # Logger when not debugging and when stdout isn't already a Logger.
        if not getattr(cfg, "debug", False) and not isinstance(sys.stdout, data_utils.Logger):
            output_file: Path | None = None

            # Training env models (discrete/continuous) use: <save_dir>/<nnet_name>/output.txt
            if cfg.stage in {"train_model_disc", "train_model_cont"}:
                env_paths = cfg.env_model_paths()
                nnet_name = cfg.env_model_name or "env_model"
                output_file = Path(env_paths.save_dir) / nnet_name / "output.txt"

            # Heuristic training: saved under heur_paths.save_dir/<heur_nnet_name>/output.txt
            elif cfg.stage == "train_heur":
                heur_paths = cfg.heur_model_paths()
                nnet_name = str(cfg.heur_nnet_name or "heur_model")
                # Distributed heuristic training should only log on rank 0
                if getattr(cfg, "use_dist", False):
                    world_rank = dist_utils.get_env_vars().get("world_rank", 0)
                    output_file = Path(heur_paths.save_dir) / str(nnet_name) / "output.txt" if world_rank == 0 else None
                else:
                    output_file = Path(heur_paths.save_dir) / str(nnet_name) / "output.txt"

            # Search stages write to results_dir/output.txt
            elif cfg.stage == "qstar":
                results_dir = (
                    Path("deepcubeai/results")
                    / cfg.env
                    / (
                        cfg.qstar_results_dir
                        or (
                            f"model={cfg.env_model_name}__heur={cfg.heur_nnet_name}_QStar_results/"
                            f"path_cost_weight={cfg.qstar_weight or 1}"
                        )
                    )
                )
                output_file = results_dir / "output.txt"
            elif cfg.stage == "ucs":
                results_dir = (
                    Path("deepcubeai/results")
                    / cfg.env
                    / (cfg.ucs_results_dir or f"model={cfg.env_model_name}_UCS_results")
                )
                output_file = results_dir / "output.txt"
            elif cfg.stage == "gbfs":
                results_dir = (
                    Path("deepcubeai/results")
                    / cfg.env
                    / (cfg.gbfs_results_dir or f"model={cfg.env_model_name}__heur={cfg.heur_nnet_name}_GBFS_results")
                )
                output_file = results_dir / "output.txt"

            # Fallback: a generic pipeline logs directory
            if output_file is None:
                logs_dir = Path("deepcubeai/logs")
                logs_dir.mkdir(parents=True, exist_ok=True)
                stage_name = cfg.stage or "pipeline"
                output_file = logs_dir / f"{stage_name}_output.txt"

            # Ensure parent exists and set the Logger (if we still have a target)
            if output_file is not None:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                sys.stdout = data_utils.Logger(str(output_file), "a")

        run_with_timing(cfg.stage, timed_stages[cfg.stage])
    elif cfg.stage == "envs":
        info = env_utils.list_environments_info()
        if getattr(args, "as_json", False):
            print(json.dumps(info, indent=2, ensure_ascii=False))
        else:
            print("Known environments:")
            for k, v in info.items():
                print(
                    f" - {k} ({v.get('type')}) -> module={v.get('module')} attr={v.get('attr')}"
                    if getattr(args, "verbose", False)
                    else f" - {k}"
                )
    elif cfg.stage == "envs-add":
        # Add a user environment to the registry. Expect --key and --module, --attr optional
        key = getattr(args, "key", None)
        module = getattr(args, "module", None)
        attr = getattr(args, "attr", None)
        if not key or not module:
            raise ConfigError("--key and --module are required for envs-add")
        try:
            env_utils.add_env(key, module, attr)
            print(f"Added environment '{key}' -> module={module} attr={attr}")
        except Exception as exc:
            raise ConfigError(f"Failed to add environment: {exc}") from exc
    elif cfg.stage == "envs-remove":
        # Remove a previously added user environment from the registry
        key = getattr(args, "key", None)
        if not key:
            raise ConfigError("--key is required for envs-remove")
        try:
            env_utils.remove_env(key)
            print(f"Removed environment '{key}'")
        except Exception as exc:
            raise ConfigError(f"Failed to remove environment: {exc}") from exc
    else:
        raise ConfigError(f"Unknown stage: {cfg.stage}")


if __name__ == "__main__":
    main()
