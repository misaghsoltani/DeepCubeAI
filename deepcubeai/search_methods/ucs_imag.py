from argparse import ArgumentParser
from typing import Any, Dict, List

from deepcubeai.search_methods.qstar_imag import main as qstar_main


def args_to_list(args_dict: Dict[str, str]) -> List[str]:
    return [str(item) for pair in args_dict.items() for item in pair if pair[1] is not None]


def parse_arguments(parser: ArgumentParser) -> Dict[str, Any]:
    parser.add_argument("--states",
                        type=str,
                        required=True,
                        help="File containing states to solve")
    parser.add_argument("--env",
                        type=str,
                        required=True,
                        help="Environment: cube3, iceslider, digitjump, sokoban")

    parser.add_argument("--env_model", type=str, required=True, help="Directory of env model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for BWAS")

    parser.add_argument("--results_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--start_idx", type=int, default=None, help="")
    parser.add_argument("--nnet_batch_size",
                        type=int,
                        default=None,
                        help="Set to control how many states per GPU are "
                        "evaluated by the neural network at a time. "
                        "Does not affect final results, "
                        "but will help if nnet is running out of "
                        "memory.")
    parser.add_argument("--per_eq_tol",
                        type=float,
                        required=True,
                        help="Percent of latent state elements that need to "
                        "be equal to declare equal")
    parser.add_argument("--verbose", action="store_true", default=False, help="Set for verbose")
    parser.add_argument("--debug", action="store_true", default=False, help="Set when debugging")

    # If provided as --save_imgs 'True', then args.save_imgs will be 'True'
    # If provided as --save_imgs (without any value), then args.save_imgs will be True
    # If is not provided --save_imgs at all, then args.save_imgs will be False
    parser.add_argument("--save_imgs",
                        nargs="?",
                        const=True,
                        default=False,
                        help="Save the images of the steps of solving each state to file")

    # parse arguments
    args = parser.parse_args()

    if (args.save_imgs.lower() not in ("true", "1")) and (args.save_imgs.lower()
                                                          not in ("false", "0")):
        raise ValueError(
            "Invalid value for '--save_imgs'. Expected values: 'true', '1', 'false', or '0'.")

    args_dict: Dict[str, Any] = vars(args)

    return args_dict


def main():
    # arguments
    parser: ArgumentParser = ArgumentParser()
    args_dict: Dict[str, Any] = parse_arguments(parser)

    verbose = args_dict.pop("verbose")
    debug = args_dict.pop("debug")

    args_list: List[str] = args_to_list({f"--{key}": value for key, value in args_dict.items()})
    args_qstar_main: List[str] = ["--weight", "1.0", "--h_weight", "0.0"]

    args_lsit = args_qstar_main + args_list

    if verbose:
        args_lsit.append("--verbose")

    if debug:
        args_lsit.append("--debug")

    qstar_main(args_lsit)


if __name__ == "__main__":
    main()
