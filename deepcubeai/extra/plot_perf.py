import argparse
import re
from typing import Dict, List, Optional

import numpy as np
import pylab


def get_res(file_name: str,
            start: Optional[int] = None,
            end: Optional[int] = None) -> List[List[float]]:
    """Extracts results from a log file.

    Args:
        file_name (str): The name of the log file.
        start (Optional[int], optional): The starting iteration to consider. Defaults to None.
        end (Optional[int], optional): The ending iteration to consider. Defaults to None.

    Returns:
        List[List[float]]: A list containing two lists: iterations and corresponding losses.
    """
    losses = [[], []]
    curr_itr = 0
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


def moving_ave(x: List[float], n: int) -> np.ndarray:
    """Calculates the moving average of a list of numbers.

    Args:
        x (List[float]): The list of numbers.
        n (int): The window size for the moving average.

    Returns:
        np.ndarray: The moving average of the input list.
    """
    res = np.convolve(x, np.ones((n, )) / n, mode="valid")
    return res


def main():
    """Main function to parse arguments and plot results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_names",
                        type=str,
                        required=True,
                        help="Comma-separated list of log file names.")
    parser.add_argument("--names",
                        type=str,
                        required=True,
                        help="Comma-separated list of experiment names.")
    parser.add_argument("--start", type=int, default=None, help="Starting iteration to consider.")
    parser.add_argument("--end", type=int, default=None, help="Ending iteration to consider.")
    parser.add_argument("--ave", type=int, default=1, help="Window size for moving average.")
    args = parser.parse_args()

    file_names = args.file_names.split(",")
    exp_names = args.names.split(",")

    name = exp_names[0]

    # Get results
    exp_to_res: Dict[str, List[List[float]]] = {}
    for file_name, exp_name in zip(file_names, exp_names):
        exp_to_res[exp_name] = get_res(file_name, args.start, args.end)

    # Plot results
    ave_num = args.ave
    exp_ls = ["-", "--"]

    for exp_idx, exp_name in enumerate(exp_names):
        x_ave = exp_to_res[exp_name][0]
        y_ave = exp_to_res[exp_name][1]

        if ave_num > 1:
            x_ave = moving_ave(x_ave, ave_num)
            y_ave = moving_ave(y_ave, ave_num)

        pylab.plot(x_ave, y_ave, label=f"{exp_name}", lw=2, linestyle=exp_ls[exp_idx])

    lgd = pylab.legend(loc="upper center",
                       bbox_to_anchor=(0.5, 1.22),
                       ncol=3,
                       fontsize="medium",
                       title="Number of Scrambles")
    pylab.ylabel("Percent Solved with Greedy Best-First Search", fontsize="large")
    pylab.xlabel("Iteration", fontsize="large")
    pylab.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    pylab.xticks(fontsize="medium")
    pylab.yticks(fontsize="medium")

    pylab.savefig(f"Loss{name}.eps", bbox_extra_artists=(lgd, ), bbox_inches="tight")

    # pylab.show()


if __name__ == "__main__":
    main()
