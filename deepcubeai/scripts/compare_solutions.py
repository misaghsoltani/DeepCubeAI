import pickle
from argparse import ArgumentParser, Namespace
from typing import Any, Dict

import numpy as np

from deepcubeai.utils.data_utils import print_args


def print_stats(data: np.ndarray, hist: bool = False) -> None:
    """Prints statistical information about the data.

    Args:
        data (np.ndarray): The data to analyze.
        hist (bool, optional): Whether to print histogram data. Defaults to False.
    """
    print(f"Min/Max/Median/Mean(Std) "
          f"{min(data)}/{max(data)}/{float(np.median(data))}"
          f"/{float(np.mean(data))}({float(np.std(data))})")
    if hist:
        hist1 = np.histogram(data)
        for x, y in zip(hist1[0], hist1[1]):
            print(f"{x} {y}")


def print_results(results: Dict[str, Any]) -> None:
    """Prints the results of the analysis.

    Args:
        results (Dict[str, Any]): The results dictionary containing times, solutions, and
            nodes generated.
    """
    times = np.array(results["times"])
    lens = np.array([len(x) for x in results["solutions"]])
    num_nodes_generated = np.array(results["num_nodes_generated"])

    print("-Times-")
    print_stats(times)
    print("-Lengths-")
    print_stats(lens)
    print("-Nodes Generated-")
    print_stats(num_nodes_generated)
    print("-Nodes/Sec-")
    print_stats(np.array(num_nodes_generated) / np.array(times))


def main():
    """Main function to parse arguments and compare solutions."""
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--soln1",
                        type=str,
                        required=True,
                        help="Path to the first solution file.")
    parser.add_argument("--soln2",
                        type=str,
                        required=True,
                        help="Path to the second solution file.")

    args: Namespace = parser.parse_args()
    print_args(args)

    with open(args.soln1, "rb") as file1, open(args.soln2, "rb") as file2:
        results1: Dict[str, Any] = pickle.load(file1)
        results2: Dict[str, Any] = pickle.load(file2)

    lens1 = np.array([len(x) for x in results1["solutions"]])
    lens2 = np.array([len(x) for x in results2["solutions"]])

    print(f"{len(results1['states'])} states")

    print("\n--SOLUTION 1---")
    print_results(results1)

    print("\n--SOLUTION 2---")
    print_results(results2)

    print("\n\n------Solution 2 - Solution 1 Lengths-----")
    print_stats(lens2 - lens1, hist=False)
    print(f"{100 * np.mean(lens2 == lens1):.2f}% soln2 equal to soln1")


if __name__ == "__main__":
    main()
