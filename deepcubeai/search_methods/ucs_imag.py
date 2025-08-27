from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from deepcubeai.search_methods.qstar_imag import main as qstar_main


def args_to_list(args_dict: dict[str, str]) -> list[str]:
    """Convert arguments dictionary to a list of command-line arguments.

    Args:
        args_dict: Dictionary mapping argument names to values.

    Returns:
        List of command-line arguments.
    """
    return [str(item) for pair in args_dict.items() for item in pair if pair[1] is not None]


def parse_arguments(parser: ArgumentParser) -> dict[str, Any]:
    """Parse command-line arguments for UCS search.

    Args:
        parser: Argument parser instance.

    Returns:
        Dictionary of parsed arguments.
    """
    parser.add_argument("--states", type=str, required=True, help="File containing states to solve")
    parser.add_argument("--env", type=str, required=True, help="Environment: cube3, iceslider, digitjump, sokoban")

    parser.add_argument("--env_model", type=str, required=True, help="Directory of env model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for BWAS")

    parser.add_argument("--results_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--start_idx", type=int, default=None, help="")
    parser.add_argument(
        "--nnet_batch_size",
        type=int,
        default=None,
        help="Set to control how many states per GPU are evaluated by the neural network at a time. "
        "Does not affect final results, but will help if nnet is running out of memory.",
    )
    parser.add_argument(
        "--per_eq_tol",
        type=float,
        required=True,
        help="Percent of latent state elements that need to be equal to declare equal",
    )
    parser.add_argument("--verbose", action="store_true", default=False, help="Set for verbose")
    parser.add_argument("--debug", action="store_true", default=False, help="Set when debugging")

    # If provided as --save_imgs 'True', then args.save_imgs will be 'True'
    # If provided as --save_imgs (without any value), then args.save_imgs will be True
    # If is not provided --save_imgs at all, then args.save_imgs will be False
    parser.add_argument(
        "--save_imgs",
        nargs="?",
        const=True,
        default=False,
        help="Save the images of the steps of solving each state to file",
    )

    # parse arguments
    args: Namespace = parser.parse_args()

    if (args.save_imgs.lower() not in {"true", "1"}) and (args.save_imgs.lower() not in {"false", "0"}):
        raise ValueError("Invalid value for '--save_imgs'. Expected values: 'true', '1', 'false', or '0'.")

    args_dict: dict[str, Any] = vars(args)

    return args_dict


@dataclass(frozen=True, slots=True)
class UCSImagConfig:
    """Programmatic config for UCS (Q* with weight=1, h_weight=0)."""

    states: str
    env: str
    env_model: str
    results_dir: str
    per_eq_tol: float
    batch_size: int = 1
    nnet_batch_size: int | None = None
    verbose: bool = False
    debug: bool = False
    save_imgs: bool = False

    @staticmethod
    def from_json(path: str) -> UCSImagConfig:
        """Load config from JSON, ignoring unknown keys."""
        raw: dict[str, Any] = json.loads(Path(path).read_bytes())
        return UCSImagConfig(**raw)


def run_ucs_imag(cfg: UCSImagConfig) -> None:
    """Programmatic entrypoint for UCS via qstar main(args_list)."""
    args_raw: dict[str, str | None] = {
        "--states": cfg.states,
        "--env": cfg.env,
        "--env_model": cfg.env_model,
        "--results_dir": cfg.results_dir,
        "--per_eq_tol": str(cfg.per_eq_tol),
        "--batch_size": str(cfg.batch_size),
        "--nnet_batch_size": None if cfg.nnet_batch_size is None else str(cfg.nnet_batch_size),
    }
    args: dict[str, str] = {k: v for k, v in args_raw.items() if v is not None}
    args_list: list[str] = args_to_list(args)
    # UCS baseline weights
    args_list = ["--weight", "1.0", "--h_weight", "0.0"] + args_list
    if cfg.verbose:
        args_list.append("--verbose")
    if cfg.debug:
        args_list.append("--debug")
    if cfg.save_imgs:
        args_list += ["--save_imgs", "true"]
    qstar_main(args_list)


def main() -> None:
    """Main function to run Uniform Cost Search (UCS)."""
    # arguments
    parser: ArgumentParser = ArgumentParser()
    args_dict: dict[str, Any] = parse_arguments(parser)

    verbose: bool = args_dict.pop("verbose")
    debug: bool = args_dict.pop("debug")

    args_list: list[str] = args_to_list({f"--{key}": value for key, value in args_dict.items()})
    args_qstar_main: list[str] = ["--weight", "1.0", "--h_weight", "0.0"]

    args_list = args_qstar_main + args_list

    if verbose:
        args_list.append("--verbose")

    if debug:
        args_list.append("--debug")

    qstar_main(args_list)


if __name__ == "__main__":
    main()
