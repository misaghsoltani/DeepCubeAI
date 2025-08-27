from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy import float32
from numpy.typing import NDArray


def get_res(file_name: str, start: int | None = None, end: int | None = None) -> list[list[float]]:
    """Extracts results from a log file.

    Args:
        file_name (str): The name of the log file.
        start (Optional[int], optional): The starting iteration to consider. Defaults to None.
        end (Optional[int], optional): The ending iteration to consider. Defaults to None.

    Returns:
        list[list[float]]: A list containing two lists: iterations and corresponding losses.
    """
    losses: list[list[float]] = [[], []]
    curr_itr: int = 0
    with open(file_name, encoding="utf-8") as f:
        for line in f:
            m = re.search(r"Itr: (\d+)", line)
            if m is not None:
                curr_itr = int(m.group(1))

            m = re.search(r"Back Steps: 10.* %Solved: ([\d+\.]+),", line)
            if m is not None:
                if (end is not None) and (curr_itr > end):
                    continue
                if (start is not None) and (curr_itr < start):
                    continue

                loss = float(m.group(1))

                if (len(losses[0]) > 0) and losses[0][-1] == curr_itr:
                    losses[0][-1] = curr_itr
                    losses[1][-1] = loss
                else:
                    losses[0].append(curr_itr)
                    losses[1].append(loss)

    return losses


def moving_ave(x: list[float] | NDArray[float32], n: int) -> NDArray[float32]:
    """Calculates the moving average of a list of numbers.

    Args:
        x (list[float]): The list of numbers.
        n (int): The window size for the moving average.

    Returns:
        NDArray: The moving average of the input list.
    """
    res = np.convolve(x, np.ones((n,)) / n, mode="valid")
    return res


@dataclass(frozen=True, slots=True)
class PlotPerfConfig:
    """Config for plotting performance from training logs."""

    file_names: list[str]
    names: list[str]
    start: int | None = None
    end: int | None = None
    ave: int = 1

    @staticmethod
    def from_json(path: str) -> PlotPerfConfig:
        """Load config from JSON file, ignoring unknown keys."""
        raw: dict[str, Any] = json.loads(Path(path).read_bytes())
        return PlotPerfConfig(**raw)


def run_plot_perf(cfg: PlotPerfConfig) -> None:
    """Programmatic entrypoint for plotting performance."""
    plot_from_values(cfg.file_names, cfg.names, cfg.start, cfg.end, cfg.ave)


def plot_from_values(
    file_names: Sequence[str], names: Sequence[str], start: int | None, end: int | None, ave: int
) -> None:
    """Core plotting function used by both CLI and programmatic runner."""
    exp_names = list(names)

    name = exp_names[0]

    # Get results
    exp_to_res: dict[str, list[list[float]]] = {}
    for file_name, exp_name in zip(file_names, exp_names, strict=False):
        exp_to_res[exp_name] = get_res(file_name, start, end)

    # Plot results
    ave_num: int = ave
    exp_ls = ["-", "--"]

    for exp_idx, exp_name in enumerate(exp_names):
        x_seq = np.asarray(exp_to_res[exp_name][0], dtype=float32)
        y_seq = np.asarray(exp_to_res[exp_name][1], dtype=float32)

        x_ave: NDArray[float32] = moving_ave(x_seq, ave_num)
        if ave_num > 1:
            y_ave: NDArray[float32] = moving_ave(y_seq, ave_num)
        else:
            y_ave = y_seq

        plt.plot(x_ave, y_ave, label=f"{exp_name}", lw=2, linestyle=exp_ls[exp_idx])

    lgd = plt.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, fontsize="medium", title="Number of Scrambles"
    )
    plt.ylabel("Percent Solved with Greedy Best-First Search", fontsize="large")
    plt.xlabel("Iteration", fontsize="large")
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    plt.xticks(fontsize="medium")
    plt.yticks(fontsize="medium")

    plt.savefig(f"Loss{name}.eps", bbox_extra_artists=(lgd,), bbox_inches="tight")

    # pylab.show()


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_names", type=str, required=True, help="Comma-separated list of log file names.")
    parser.add_argument("--names", type=str, required=True, help="Comma-separated list of experiment names.")
    parser.add_argument("--start", type=int, default=None, help="Starting iteration to consider.")
    parser.add_argument("--end", type=int, default=None, help="Ending iteration to consider.")
    parser.add_argument("--ave", type=int, default=1, help="Window size for moving average.")
    args = parser.parse_args()
    file_names = args.file_names.split(",")
    names = args.names.split(",")
    plot_from_values(file_names, names, args.start, args.end, args.ave)


if __name__ == "__main__":
    main()
