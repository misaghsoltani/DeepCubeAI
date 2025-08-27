# ========================================================================================== #
# Q* Search is a variant of A* search for DQNs. For more information, see the paper:         #
# Agostinelli, Forest, et al. "Q* Search: Heuristic Search with Deep Q-Networks." (2024).    #
# https://prl-theworkshop.github.io/prl2024-icaps/papers/9.pdf                               #
# ========================================================================================== #
from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Callable
from dataclasses import dataclass
from heapq import heappop, heappush
import json
import os
import pickle
import sys
import time
from typing import Any, cast

import numpy as np
from numpy import float32, intp
from numpy.typing import NDArray
import torch

from deepcubeai.environments.environment_abstract import State
from deepcubeai.environments.sokoban import SokobanState
from deepcubeai.utils import data_utils, env_utils, misc_utils, nnet_utils, search_utils
from deepcubeai.utils.data_utils import print_args


class Node:
    """Represents a node in the search tree.

    Attributes:
        state: The state representation as NDArray.
        path_cost: The cost of the path to this node.
        parent_move: The move that led to this node from its parent.
        parent: Reference to the parent node.
        _hash: Cached hash value for the node.
    """

    __slots__ = ("state", "path_cost", "parent_move", "parent", "_hash", "_state_bytes")

    def __init__(self, state: NDArray[float32], path_cost: float, parent_move: int | None, parent: Node | None) -> None:
        """Initializes a Node.

        Args:
            state (NDArray[float32]): The state representation.
            path_cost (float): The cost of the path to this node.
            parent_move (int | None): The move that led to this node from its parent.
            parent (Node | None): Reference to the parent node.
        """
        self.state: NDArray[float32] = state
        self.path_cost: float = path_cost
        self.parent_move: int | None = parent_move
        self.parent: Node | None = parent

        self._hash: int | None = None
        self._state_bytes: bytes | None = None

    def _bytes_key(self) -> bytes:
        """Lazily compute and cache the bytes representation of the state for hashing/closed set.

        Returns:
            bytes: Immutable byte representation of the state.
        """
        b: bytes | None = self._state_bytes
        if b is None:
            b = self.state.tobytes()
            self._state_bytes = b
        return b

    def __hash__(self) -> int:
        """Returns the hash of the node.

        Returns:
            int: The hash value.
        """
        if self._hash is None:
            # Derive hash from cached bytes to avoid repeated tobytes conversions
            self._hash = hash(self._bytes_key())
        return self._hash

    def __eq__(self, other: object) -> bool:
        """Checks if two nodes are equal.

        Args:
            other: The other object to compare.

        Returns:
            bool: True if nodes are equal, False otherwise.
        """
        if not isinstance(other, Node):
            return NotImplemented

        return bool(np.array_equal(self.state, other.state))


OpenSetElem = tuple[float, int, Node, int]


class Instance:
    """Represents a search instance with open/closed sets.

    Attributes:
        state: The initial state as NDArray.
        state_goal: The goal state as NDArray.
        cost: The cost associated with this instance.
        open_set: Priority queue of open nodes.
        closed_set: Set of closed nodes.
        root_node: The root node of the search tree.
    """

    __slots__ = (
        "state_goal",
        "open_set",
        "closed_dict",
        "heappush_count",
        "goal_nodes",
        "num_nodes_generated",
        "root_node",
        "weight",
    )

    def __init__(self, state: NDArray[float32], state_goal: NDArray[float32], cost: float, weight: float) -> None:
        """Initializes an Instance.

        Args:
            state (np.NDArray): The initial state.
            state_goal (np.NDArray): The goal state.
            cost (float): The initial cost.
            weight (float): The weight for path cost.
        """
        self.state_goal: NDArray[float32] = state_goal

        self.open_set: list[OpenSetElem] = []

        # CLOSED keyed by immutable bytes of state for faster equality checks and fewer collisions
        self.closed_dict: dict[bytes, float] = {}

        self.heappush_count: int = 0
        self.goal_nodes: list[Node] = []
        self.num_nodes_generated: int = 0

        self.root_node: Node = Node(state, 0.0, None, None)
        self.weight: float = float(weight)

        self.push_to_open([self.root_node], [[-1]], [[cost]])

    def push_to_open(self, nodes: list[Node], moves: list[list[int]], costs: list[list[float]]) -> None:
        """Pushes nodes to the open set.

        Args:
            nodes (list[Node]): The nodes to push.
            moves (list[list[int]]): The moves associated with the nodes.
            costs (list[list[float]]): The costs associated with the nodes.
        """
        heappush_count: int = self.heappush_count
        for node, moves_node, costs_node in zip(nodes, moves, costs, strict=False):
            for move, cost in zip(moves_node, costs_node, strict=False):
                heappush(self.open_set, (float(cost), heappush_count, node, int(move)))
                heappush_count += 1
        self.heappush_count = heappush_count

    def pop_from_open(self, num_nodes: int) -> tuple[list[Node], list[int]]:
        """Pops nodes from the open set.

        Args:
            num_nodes (int): The number of nodes to pop.

        Returns:
            tuple[list[Node], list[int]]: The popped nodes and their associated moves.
        """
        num_to_pop: int = min(num_nodes, len(self.open_set))
        if num_to_pop == 0:
            return [], []
        popped_elems: list[OpenSetElem] = [heappop(self.open_set) for _ in range(num_to_pop)]
        popped_nodes: list[Node] = [elem[2] for elem in popped_elems]
        moves: list[int] = [elem[3] for elem in popped_elems]
        return popped_nodes, moves

    def remove_in_closed(self, nodes: list[Node]) -> list[Node]:
        """Removes nodes that are in the closed set.

        Args:
            nodes (list[Node]): The nodes to check.

        Returns:
            list[Node]: The nodes not in the closed set.
        """
        nodes_not_in_closed: list[Node] = []
        closed_dict: dict[bytes, float] = self.closed_dict

        key: bytes
        prev: float | None
        for node in nodes:
            key = node._bytes_key()
            prev = closed_dict.get(key)
            if prev is None or prev > node.path_cost:
                nodes_not_in_closed.append(node)
                closed_dict[key] = node.path_cost

        return nodes_not_in_closed


@torch.inference_mode()
def pop_from_open(
    instances: list[Instance], batch_size: int, model_fn: Callable[..., Any], is_solved_fn: Callable[..., Any]
) -> list[list[Node]]:
    """Pops nodes from the open sets of instances.

    Args:
        instances (list[Instance]): The instances to pop nodes from.
        batch_size (int): The batch size.
        model_fn (Callable): The model function.
        is_solved_fn (Callable): The function to check if a state is solved.

    Returns:
        list[list[Node]]: The popped nodes for each instance.
    """
    popped_nodes_by_inst: list[list[Node]] = []
    moves_by_inst: list[list[int]] = []
    for instance in instances:
        popped_nodes_inst, moves_inst = instance.pop_from_open(batch_size)
        popped_nodes_by_inst.append(popped_nodes_inst)
        moves_by_inst.append(moves_inst)

    # make moves
    popped_nodes_flat, split_idxs = misc_utils.flatten(popped_nodes_by_inst)
    moves_flat_list, _ = misc_utils.flatten(moves_by_inst)

    if len(popped_nodes_flat) == 0:
        return [[] for _ in instances]  # nothing to do

    popped_nodes_next_flat: list[Node] = []

    # initial layer check for all popped nodes (fast path)
    if moves_flat_list[0] == -1:  # initial layer
        for popped_nodes_inst in popped_nodes_by_inst:
            assert len(popped_nodes_inst) == 1, "Initial condition should only happen at the first iteration"
        assert all(mv == -1 for mv in moves_flat_list), "Initial condition should happen for all at the first iteration"
        states_next_flat: NDArray[float32] = np.stack([n.state for n in popped_nodes_flat], axis=0)
        popped_nodes_next_flat = popped_nodes_flat
    else:
        states_flat: NDArray[float32] = np.stack([n.state for n in popped_nodes_flat], axis=0)
        actions: NDArray[float32] = np.asarray(moves_flat_list, dtype=float32)
        states_next_flat = np.ascontiguousarray(model_fn(states_flat, actions).round(), dtype=float32)

        # constant step cost (vectorized)
        parent_costs = np.fromiter(
            (n.path_cost for n in popped_nodes_flat), count=len(popped_nodes_flat), dtype=float32
        )
        child_costs = parent_costs + 1.0

        # build children
        for state, path_cost, move, parent in zip(
            states_next_flat, child_costs, moves_flat_list, popped_nodes_flat, strict=False
        ):
            popped_nodes_next_flat.append(Node(state, float(path_cost), int(move), parent))

    # state goals (flattened)
    # Build state_goals_flat with preallocation to reduce Python overhead
    total = sum(len(pn) for pn in popped_nodes_by_inst)
    if total:
        sg0 = instances[0].state_goal
        goal_shape = sg0.shape
        state_goals_flat = np.empty((total, *goal_shape), dtype=float32)
        idx = 0
        for instance, popped_nodes_inst in zip(instances, popped_nodes_by_inst, strict=False):
            k = len(popped_nodes_inst)
            if k:
                state_goals_flat[idx : idx + k] = instance.state_goal
                idx += k
    else:
        state_goals_flat = np.empty((0,), dtype=float32)

    # solved?
    is_solved_flat = list(is_solved_fn(states_next_flat, state_goals_flat))
    is_solved_by_inst: list[list[bool]] = misc_utils.unflatten(is_solved_flat, split_idxs)
    popped_nodes_next_by_inst: list[list[Node]] = misc_utils.unflatten(popped_nodes_next_flat, split_idxs)

    # update per instance
    for instance, next_nodes_inst, flags_inst in zip(
        instances, popped_nodes_next_by_inst, is_solved_by_inst, strict=False
    ):
        instance.goal_nodes.extend([n for n, ok in zip(next_nodes_inst, flags_inst, strict=False) if ok])
        instance.num_nodes_generated += len(next_nodes_inst)

    return popped_nodes_next_by_inst


@torch.inference_mode()
def add_heuristic_and_cost(
    nodes: list[list[Node]],
    state_goals_flat: NDArray[float32],
    heuristic_fn: Callable[..., NDArray[float32]] | None,
    weights: list[float],
    num_actions_max: int,
) -> tuple[list[list[list[int]]], list[list[list[float]]], NDArray[float32], NDArray[float32]]:
    """Adds heuristic and cost to nodes.

    Args:
        nodes (list[list[Node]]): The nodes to add heuristic and cost to.
        state_goals_flat (np.NDArray): The flattened goal states.
        heuristic_fn (Callable): The heuristic function.
        weights (list[float]): The weights for the path costs.
        num_actions_max (int): The maximum number of actions.

    Returns:
        tuple[list[list[list[int]]], list[list[list[float]]], NDArray, NDArray]: The moves,
            costs, parent path costs, and heuristics.
    """
    nodes_flat: list[Node]
    nodes_flat, split_idxs = misc_utils.flatten(nodes)

    if len(nodes_flat) == 0:
        return [], [], np.zeros(0, dtype=float32), np.zeros(0, dtype=float32)

    # get heuristic
    states_flat: NDArray[float32] = np.stack([n.state.astype(float32) for n in nodes_flat], axis=0)
    parent_costs: NDArray[float32] = np.fromiter(
        (n.path_cost for n in nodes_flat), count=len(nodes_flat), dtype=float32
    )

    # compute node cost
    heuristics_flat: NDArray[float32]

    # If performing Q* search
    if heuristic_fn is not None:
        heuristics_flat = heuristic_fn(states_flat, state_goals_flat).astype(float32, copy=False)

    # If performing Uniform Cost Search
    else:
        heuristics_flat = np.zeros((states_flat.shape[0], num_actions_max), dtype=float32)

    weights_arr = np.asarray(weights, dtype=float32)
    path_cost_weighted = (weights_arr * parent_costs)[:, None]  # (N,1)
    total_costs = heuristics_flat + path_cost_weighted  # (N,A)

    # reuse a single immutable template for moves; no mutation later
    moves_template = list(range(num_actions_max))
    moves_flat = [moves_template] * total_costs.shape[0]
    costs_flat = total_costs.tolist()

    moves = misc_utils.unflatten(moves_flat, split_idxs)
    costs = misc_utils.unflatten(costs_flat, split_idxs)

    # return the real parent costs (float) and per-node min heuristic for logging
    return moves, costs, parent_costs, heuristics_flat.min(axis=1).astype(float32)


def add_to_open(
    instances: list[Instance], nodes: list[list[Node]], moves: list[list[list[int]]], costs: list[list[list[float]]]
) -> None:
    """Adds nodes to the open sets of instances.

    Args:
        instances (list[Instance]): The instances to add nodes to.
        nodes (list[list[Node]]): The nodes to add.
        moves (list[list[list[int]]]): The moves associated with the nodes.
        costs (list[list[list[float]]]): The costs associated with the nodes.
    """
    for instance, nodes_inst, moves_inst, costs_inst in zip(instances, nodes, moves, costs, strict=False):
        instance.push_to_open(nodes_inst, moves_inst, costs_inst)


def get_path(node: Node) -> tuple[list[NDArray[float32]], list[int], float]:
    """Gets the path from the root to the given node.

    Args:
        node (Node): The node to trace back from.

    Returns:
        tuple[list[NDArray], list[int], float]: The path, moves, and path cost.
    """
    path: list[NDArray[float32]] = []
    moves: list[int] = []

    parent_node: Node = node
    while parent_node.parent is not None:
        path.append(parent_node.state)
        if parent_node.parent_move is not None:
            moves.append(parent_node.parent_move)
        parent_node = parent_node.parent

    path.append(parent_node.state)

    path = path[::-1]
    moves = moves[::-1]

    return path, moves, node.path_cost


def get_is_solved_fn(per_eq_tol: float) -> Callable[..., Any]:
    """Gets the function to check if states are solved.

    Args:
        per_eq_tol (float): The tolerance for equality.

    Returns:
        Callable: The function to check if states are solved.
    """

    def is_solved_fn(states: NDArray[float32], states_comp: NDArray[float32]) -> NDArray[np.bool_]:
        return (100 * np.equal(states, states_comp).mean(axis=1)) >= per_eq_tol

    return is_solved_fn


class QStarImag:
    """Q* search algorithm."""

    __slots__ = ("weights", "step_num", "num_actions_max", "timings", "instances", "last_node")

    weights: list[float]
    step_num: int
    num_actions_max: int
    timings: dict[str, float]
    instances: list[Instance]
    last_node: Node | None

    @torch.inference_mode()
    def __init__(
        self,
        states: NDArray[float32],
        state_goals: NDArray[float32],
        heuristic_fn: Callable[..., NDArray[float32]] | None,
        weights: list[float],
        num_actions_max: int,
    ) -> None:
        """Initializes an QStarImag instance.

        Args:
            states (np.NDArray): The initial states.
            state_goals (np.NDArray): The goal states.
            heuristic_fn (Callable): The heuristic function.
            weights (list[float]): The weights for the path costs.
            num_actions_max (int): The maximum number of actions.
        """
        self.weights: list[float] = weights
        self.step_num: int = 0
        self.num_actions_max: int = num_actions_max

        self.timings: dict[str, float] = {"pop": 0.0, "closed": 0.0, "heur": 0.0, "add": 0.0, "itr": 0.0}

        # compute starting costs
        # Heuristic values if performing Q* search, zero if performing Uniform Cost Search
        costs: NDArray[float32] = (
            heuristic_fn(states, state_goals).min(axis=1)
            if heuristic_fn is not None
            else np.zeros(len(states), dtype=float32)
        )

        # initialize instances
        self.instances: list[Instance] = [
            Instance(state, state_goal, cost, weights[i])
            for i, (state, state_goal, cost) in enumerate(zip(states, state_goals, costs, strict=False))
        ]

        self.last_node: Node | None = None

    # TODO make separate is_solved_fn and is_same_fn

    @torch.inference_mode()
    def step(
        self,
        heuristic_fn: Callable[..., NDArray[float32]] | None,
        model_fn: Callable[..., NDArray[float32]],
        is_solved_fn: Callable[..., NDArray[np.bool_]],
        batch_size: int,
        verbose: bool = False,
    ) -> bool:
        """Performs a step in the Q* search.

        Args:
            heuristic_fn (Callable): The heuristic function.
            model_fn (Callable): The model function.
            is_solved_fn (Callable): The function to check if a state is solved.
            batch_size (int): The batch size.
            verbose (bool): Whether to print verbose output.

        Returns:
            bool: True if the search continues, False if no more nodes to expand.
        """
        start_time_itr = time.time()
        instances = [
            instance for instance in self.instances if (len(instance.goal_nodes) == 0) and len(instance.open_set) > 0
        ]
        if len(instances) == 0:
            print("Open set is empty. Returning the result ...")
            return False

        # Pop from open
        start_time = time.time()
        popped_nodes: list[list[Node]] = pop_from_open(instances, batch_size, model_fn, is_solved_fn)
        pop_time = time.time() - start_time

        # Check if popped nodes are in closed
        start_time = time.time()
        for inst_idx, instance in enumerate(instances):
            popped_nodes[inst_idx] = instance.remove_in_closed(popped_nodes[inst_idx])
        closed_time = time.time() - start_time

        if len(popped_nodes) > 0:
            popped = popped_nodes[-1]
            if len(popped) > 0:
                self.last_node = popped[-1]

        # Get heuristic of children
        start_time = time.time()
        # Build state_goals_flat with preallocation
        total = sum(len(pn) for pn in popped_nodes)
        if total:
            goal_shape = instances[0].state_goal.shape
            state_goals_flat = np.empty((total, *goal_shape), dtype=float32)
            idx = 0
            for instance, popped_nodes_inst in zip(instances, popped_nodes, strict=False):
                k = len(popped_nodes_inst)
                if k:
                    state_goals_flat[idx : idx + k] = instance.state_goal
                    idx += k
        else:
            # No nodes remained after filtering (e.g., all were in CLOSED).
            state_goals_flat = np.empty((0,), dtype=float32)

        # Align weights to the active instances only using per-instance weight
        weights_list: list[float] = []
        if total:
            for instance, popped_nodes_inst in zip(instances, popped_nodes, strict=False):
                k = len(popped_nodes_inst)
                if k:
                    weights_list.extend([instance.weight] * k)
        moves, costs, path_costs, heuristics = add_heuristic_and_cost(
            popped_nodes, state_goals_flat, heuristic_fn, weights_list, self.num_actions_max
        )
        heur_time = time.time() - start_time

        # Add to open
        start_time = time.time()
        add_to_open(instances, popped_nodes, moves, costs)
        add_time = time.time() - start_time

        itr_time = time.time() - start_time_itr

        # Print to screen
        if verbose:
            if heuristics.shape[0] > 0:
                min_heur = np.min(heuristics)
                min_heur_pc = path_costs[np.argmin(heuristics)]
                max_heur = np.max(heuristics)
                max_heur_pc = path_costs[np.argmax(heuristics)]

                print(
                    f"Itr: {self.step_num}, Added to OPEN - Min/Max Heur(PathCost): "
                    f"{min_heur:.2f}({min_heur_pc:.2f})/{max_heur:.2f}({max_heur_pc:.2f}) "
                )

            print(
                f"Times - pop: {pop_time:.2f}, closed: {closed_time:.2f}, heur: {heur_time:.2f}, "
                f"add: {add_time:.2f}, itr: {itr_time:.2f}"
            )

        # Update timings
        self.timings["pop"] += pop_time
        self.timings["closed"] += closed_time
        self.timings["heur"] += heur_time
        self.timings["add"] += add_time
        self.timings["itr"] += itr_time

        self.step_num += 1

        return True

    def has_found_goal(self) -> list[bool]:
        """Checks if the goal has been found for each instance.

        Returns:
            list[bool]: List indicating if the goal has been found for each instance.
        """
        goal_found: list[bool] = [len(self.get_goal_nodes(idx)) > 0 for idx in range(len(self.instances))]
        return goal_found

    def get_goal_nodes(self, inst_idx: int) -> list[Node]:
        """Gets the goal nodes for a given instance.

        Args:
            inst_idx (int): The index of the instance.

        Returns:
            list[Node]: The goal nodes.
        """
        return self.instances[inst_idx].goal_nodes

    def get_goal_node_smallest_path_cost(self, inst_idx: int) -> Node:
        """Gets the goal node with the smallest path cost for a given instance.

        Args:
            inst_idx (int): The index of the instance.

        Returns:
            Node: The goal node with the smallest path cost.
        """
        goal_nodes: list[Node] = self.get_goal_nodes(inst_idx)
        path_costs: list[float] = [node.path_cost for node in goal_nodes]

        goal_node: Node = goal_nodes[int(np.argmin(path_costs))]
        return goal_node

    def get_num_nodes_generated(self, inst_idx: int) -> int:
        """Gets the number of nodes generated for a given instance.

        Args:
            inst_idx (int): The index of the instance.

        Returns:
            int: The number of nodes generated.
        """
        return self.instances[inst_idx].num_nodes_generated


def parse_arguments(parser: ArgumentParser, args_list: list[str] | None = None) -> dict[str, Any]:
    """Parses command-line arguments.

    Args:
        parser (ArgumentParser): The argument parser.
        args_list (list[str], optional): List of arguments. Defaults to None.

    Returns:
        dict[str, Any]: Dictionary of parsed arguments.
    """
    parser.add_argument(
        "--h_weight",
        type=float,
        default=1.0,
        help="Weight of the heuristics. Set it to 0 for performing a Uniform Cost Search",
    )
    # Parse known arguments first for the value of --h_weight
    args, _ = parser.parse_known_args(args_list)

    if args.h_weight != 0.0:
        parser.add_argument("--heur", type=str, required=True, help="Directory of heuristic function")
    else:
        parser.add_argument("--heur", type=str, default=None, help="Directory of heuristic function")

    parser.add_argument("--states", type=str, required=True, help="File containing states to solve")
    parser.add_argument("--env", type=str, required=True, help="Environment: cube3, iceslider, digitjump, sokoban")

    parser.add_argument("--env_model", type=str, required=True, help="Directory of env model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for BWQS")
    parser.add_argument("--weight", type=float, default=1.0, help="Weight of path cost")

    parser.add_argument("--results_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--start_idx", type=int, default=None, help="")
    parser.add_argument(
        "--nnet_batch_size",
        type=int,
        default=None,
        help="Set to control how many states per GPU are "
        "evaluated by the neural network at a time. "
        "Does not affect final results, "
        "but will help if nnet is running out of "
        "memory.",
    )

    parser.add_argument(
        "--per_eq_tol",
        type=float,
        required=True,
        help="Percent of latent state elements that need to be equal to declare equal",
    )

    parser.add_argument("--verbose", action="store_true", default=False, help="Set for verbose")
    parser.add_argument("--debug", action="store_true", default=False, help="Set when debugging")

    # If provided as --save_imgs 'true', then args.save_imgs will be 'true'
    # If provided as --save_imgs (without any value), then args.save_imgs will be 'true'
    # If is not provided --save_imgs at all, then args.save_imgs will be 'false'
    parser.add_argument(
        "--save_imgs",
        type=str,
        nargs="?",
        const="true",
        default="false",
        help="Save the images of the steps of solving each state to file",
    )

    # parse arguments
    args = parser.parse_args(args_list)

    if args.h_weight == 0.0:
        assert args.weight != 0, "h_weight and weight cannot be 0 at the same time"
        args.heur = None

    if args.save_imgs.lower() in {"true", "1"}:
        args.save_imgs = True
    elif args.save_imgs.lower() in {"false", "0"}:
        args.save_imgs = False
    else:
        raise ValueError("Invalid value for '--save_imgs'. Expected 'true', '1', 'false', or '0'.")

    args_dict: dict[str, Any] = vars(args)

    return args_dict


@torch.inference_mode()
def bwqs_python(
    args_dict: dict[str, Any],
    start_idx: int,
    env: Any,
    states: list[State],
    state_goals: list[State],
    results: dict[str, Any],
    results_file: str,
    save_imgs: bool,
    save_imgs_dir: str | None,
) -> None:
    """Performs the batched and weighted version of Q* search algorithm.

    Args:
        args_dict (dict[str, Any]): Dictionary of arguments.
        start_idx (int): Starting index.
        env (Environment): The environment.
        states (list[State]): List of states.
        state_goals (list[State]): List of goal states.
        results (dict[str, Any]): Dictionary to store results.
        results_file (str): Path to the results file.
        save_imgs (bool): Whether to save images. If this is True, save_imgs_dir must be provided.
        save_imgs_dir (Optional[str]): Directory to save images. This will be used only if
            save_imgs is True.
    """
    # get device
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()

    print(f"device: {device}, devices: {devices}, on_gpu: {on_gpu}\n___________________________________\n")

    # Check to see if performing UCS or Q*
    # If performing UCS, no need to load the heuristic function
    heuristic_fn = None
    if args_dict["h_weight"] != 0:
        heuristic_fn = nnet_utils.load_heuristic_fn(
            args_dict["heur"], device, on_gpu, env.get_dqn(), clip_zero=True, batch_size=args_dict["nnet_batch_size"]
        )

    env_model_file: str = f"{args_dict['env_model']}/env_state_dict.pt"
    raw_model_fn = nnet_utils.load_model_fn(
        env_model_file, device, on_gpu, env.get_env_nnet(), batch_size=args_dict["nnet_batch_size"]
    )

    # Wrapper to convert uint8 output to float32
    def model_fn(states: NDArray[float32], actions: NDArray[float32]) -> NDArray[float32]:
        out = raw_model_fn(states, actions)
        return out.astype(float32, copy=False)

    encoder_file: str = f"{args_dict['env_model']}/encoder_state_dict.pt"
    encoder = nnet_utils.load_nnet(encoder_file, env.get_encoder(), device)
    encoder.eval()
    is_solved_fn = get_is_solved_fn(args_dict["per_eq_tol"])

    if save_imgs:
        decoder_file: str = f"{args_dict['env_model']}/decoder_state_dict.pt"
        decoder = nnet_utils.load_nnet(decoder_file, env.get_decoder(), device)
        decoder.eval()

    # TODO: Actions are assumed to be of a fixed size
    num_actions_max: int = env.num_actions_max

    print(f"Total number of test states: {len(states)}")
    for state_idx in range(start_idx, len(states)):
        state: State = states[state_idx]
        state_goal: State = state_goals[state_idx]

        start_time = time.time()
        num_itrs: int = 0
        # res is used to check whether to continue search or not
        res: bool = True

        state_real = env.state_to_real([state])
        state_enc = encoder(torch.tensor(state_real, device=device).float())[1]
        state_enc_np = state_enc.detach().cpu().numpy()
        goal_node_found: Node
        if args_dict["env"] == "sokoban":
            state_goal_sokoban: SokobanState = cast(SokobanState, state_goal)

            # blank = not wall and not box
            blocked = np.logical_or(state_goal_sokoban.walls, state_goal_sokoban.boxes)
            blank_positions: NDArray[intp] = np.argwhere(~blocked).astype(intp)  # (K,2)

            state_goals_i: list[SokobanState] = [
                SokobanState(pos, state_goal_sokoban.boxes, state_goal_sokoban.walls) for pos in blank_positions
            ]
            state_goals_i_as_states: list[State] = cast(list[State], state_goals_i)
            state_goals_i_real = env.state_to_real(state_goals_i_as_states)

            state_goals_i_enc = encoder(torch.tensor(state_goals_i_real, device=device).float())[1]
            state_goals_i_enc_np = state_goals_i_enc.detach().cpu().numpy()

            states_enc_np = np.repeat(state_enc_np, state_goals_i_enc_np.shape[0], axis=0)

            # do Q* search
            qstar = QStarImag(
                states_enc_np,
                state_goals_i_enc_np,
                heuristic_fn,
                weights=[args_dict["weight"]] * states_enc_np.shape[0],
                num_actions_max=num_actions_max,
            )
            while res and not max(qstar.has_found_goal()):
                res = qstar.step(
                    heuristic_fn, model_fn, is_solved_fn, args_dict["batch_size"], verbose=args_dict["verbose"]
                )
                num_itrs += 1

            if res:
                # get goal node
                solved_idxs: NDArray[intp] = np.where(qstar.has_found_goal())[0]
                goal_node_found = qstar.get_goal_node_smallest_path_cost(solved_idxs[0])
                for solved_idx in solved_idxs[1:]:
                    goal_node_i: Node = qstar.get_goal_node_smallest_path_cost(solved_idx)
                    if goal_node_i.path_cost < goal_node_found.path_cost:
                        goal_node_found = goal_node_i

        else:
            state_goal_real = env.state_to_real([state_goal])
            state_goal_enc = encoder(torch.tensor(state_goal_real, device=device).float())[1]
            state_goal_enc_np = state_goal_enc.detach().cpu().numpy()

            qstar = QStarImag(
                state_enc_np,
                state_goal_enc_np,
                heuristic_fn,
                weights=[args_dict["weight"]],
                num_actions_max=num_actions_max,
            )
            while res and not min(qstar.has_found_goal()):
                res = qstar.step(
                    heuristic_fn, model_fn, is_solved_fn, args_dict["batch_size"], verbose=args_dict["verbose"]
                )
                num_itrs += 1

            if res:
                goal_node_found = qstar.get_goal_node_smallest_path_cost(0)

        # If the open set became empty without fidning a solution, use the last node.
        # Used only for saving the image, when save_imgs is True
        if not res:
            # Fallback to root node if no last node
            goal_node_found = qstar.last_node if qstar.last_node is not None else qstar.instances[0].root_node

        path: list[NDArray[float32]] | None
        soln: list[int] | None
        path_cost: float | None
        num_nodes_gen_idx: int

        path, soln, path_cost = get_path(goal_node_found)

        num_nodes_gen_idx = qstar.get_num_nodes_generated(0)

        solve_time = time.time() - start_time

        # check soln
        solved: bool = False
        if save_imgs:
            solved = search_utils.is_valid_soln(
                state, state_goal, soln, env, decoder, device, state_idx, path, save_imgs_dir, save_imgs
            )

        else:
            solved = search_utils.is_valid_soln(state, state_goal, soln, env)

        # assert search_utils.is_valid_soln(state, state_goal, soln, env)
        nodes_per_sec = num_nodes_gen_idx / solve_time

        # print to screen
        timing_str = ", ".join([f"{key}: {val:.2f}" for key, val in qstar.timings.items()])
        print(
            f"Times - {timing_str}, "
            f"num_itrs: {num_itrs}\n"
            f"State: {state_idx}, "
            f"Solved: {'Yes' if solved else 'No'}, "
            f"SolnCost: {path_cost:.2f}, "
            f"# Moves: {len(soln)}, "
            f"# Nodes Gen: {num_nodes_gen_idx:,}, "
            f"Time: {solve_time:.2f}, "
            f"Nodes/Sec: {nodes_per_sec:.2E}\n"
            f"___________________________________\n"
        )

        # If the State class implements the get_opt_path_len() method for getting the optimal path.
        # Note: hasattr will be True even if the abstract placeholder raises NotImplementedError.
        has_get_opt_path_len: bool = False
        if hasattr(state, "get_opt_path_len"):
            # Determine if method is actually overridden (not the abstract placeholder)
            try:
                base_impl = getattr(State, "get_opt_path_len", None)
                state_impl = getattr(state.__class__, "get_opt_path_len", None)
                if state_impl is not None and base_impl is not state_impl:
                    has_get_opt_path_len = True
            except Exception:
                # Fallback: we'll attempt call inside try/except below
                has_get_opt_path_len = True

        soln = soln if solved else None
        path_cost = path_cost if solved else None
        path = path if solved else None
        num_moves: int | None = len(soln) if solved and soln is not None else None
        results["states"].append(state)
        results["solutions"].append(soln)
        results["paths"].append(path)
        results["times"].append(solve_time)
        results["num_nodes_generated"].append(num_nodes_gen_idx)
        results["solved"].append(solved)
        results["num_itrs"].append(num_itrs)
        results["path_cost"].append(path_cost)
        results["num_moves"].append(num_moves)

        len_optimal_path: int | None = None
        is_optimal_path: int | None = None
        if has_get_opt_path_len and hasattr(state, "get_opt_path_len"):
            try:
                get_opt_path_len_method = getattr(state, "get_opt_path_len", None)
                if get_opt_path_len_method is not None and callable(get_opt_path_len_method):
                    path_len_result = get_opt_path_len_method()
                    if isinstance(path_len_result, int):
                        len_optimal_path = path_len_result
                        if soln is not None and len_optimal_path is not None:
                            is_optimal_path = bool(len(soln) <= len_optimal_path)
            except (AttributeError, TypeError, NotImplementedError):
                # Method doesn't exist, not callable, or intentionally not implemented.
                has_get_opt_path_len = False

        results["len_optimal_path"].append(len_optimal_path)
        results["is_optimal_path"].append(is_optimal_path)

        with open(results_file, "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    avg_time = np.mean(results["times"])
    avg_num_nodes_generated = np.mean(results["num_nodes_generated"])
    avg_nodes_per_sec = avg_num_nodes_generated / avg_time
    avg_moves = _get_mean(results, "num_moves")
    avg_itrs = _get_mean(results, "num_itrs")
    states_total = len(results["solved"])
    solved_total = np.sum(results["solved"])
    solved_perc = (solved_total / states_total) * 100
    optimal = "N/A"
    if has_get_opt_path_len and any(x is not None for x in results["is_optimal_path"]):
        optimal_vals = [x for x in results["is_optimal_path"] if isinstance(x, bool)]
        if optimal_vals:
            optimal_percent = np.mean(optimal_vals) * 100
            optimal = f"{optimal_percent:.2f}%"

    print(
        f"\nSummary:\n"
        f"Number of Solved States: {solved_total}, "
        f"Total Number of States: {states_total}, "
        f"Success Rate: {solved_perc:.2f}%\n"
        f"Avg # Moves: {avg_moves:.2f}, "
        f"Optimal: {optimal}, "
        f"Avg Itrs: {avg_itrs:.2f}, "
        f"Avg # Nodes Gen: {avg_num_nodes_generated:.2f}, "
        f"Avg Time: {avg_time:.2f}, Avg Nodes/Sec: {avg_nodes_per_sec:.2E}"
    )


def _get_mean(results: dict[str, Any], key: str) -> float:
    """Calculates the mean of the specified key in the results dictionary.

    Args:
        results (dict[str, Any]): Dictionary of results.
        key (str): The key to calculate the mean for.

    Returns:
        float: The mean value.
    """
    vals: list[float] = [x for x, solved in zip(results[key], results["solved"], strict=False) if solved]
    if len(vals) == 0:
        return 0

    mean_val = np.mean([x for x, solved in zip(results[key], results["solved"], strict=False) if solved])
    return float(mean_val)


@dataclass(frozen=True, slots=True)
class QStarImagConfig:
    """Configuration for running Q* or UCS search programmatically.

    If h_weight == 0.0, UCS is performed and heur can be None.
    """

    # Required
    env: str
    states: str
    env_model: str
    results_dir: str
    per_eq_tol: float

    # Optional with defaults
    weight: float = 1.0
    h_weight: float = 1.0
    heur: str | None = None
    batch_size: int = 1
    nnet_batch_size: int | None = None
    start_idx: int | None = None
    verbose: bool = False
    debug: bool = False
    save_imgs: bool = False

    @staticmethod
    def from_json(path: str | os.PathLike[str]) -> QStarImagConfig:
        """Load configuration from a JSON file, ignoring unknown keys."""
        with open(path, "rb") as f:
            data = json.loads(f.read())
        if not isinstance(data, dict):
            raise TypeError("Config JSON must be an object")
        allowed = set(QStarImagConfig.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in allowed}
        return QStarImagConfig(**filtered)


@torch.inference_mode()
def run_qstar_imag(cfg: QStarImagConfig) -> None:
    """Programmatic API: build args list from config and call main(args_list)."""
    args_list: list[str] = []

    def add_flag(name: str, value: Any) -> None:
        if isinstance(value, bool):
            if value:
                args_list.append(f"--{name}")
            return
        if value is None:
            return
        args_list.extend([f"--{name}", str(value)])

    # Required
    add_flag("env", cfg.env)
    add_flag("states", cfg.states)
    add_flag("env_model", cfg.env_model)
    add_flag("results_dir", cfg.results_dir)
    add_flag("per_eq_tol", cfg.per_eq_tol)

    # Optional
    add_flag("batch_size", cfg.batch_size)
    add_flag("weight", cfg.weight)
    add_flag("h_weight", cfg.h_weight)
    if cfg.h_weight != 0.0 and cfg.heur is not None:
        add_flag("heur", cfg.heur)
    if cfg.nnet_batch_size is not None:
        add_flag("nnet_batch_size", cfg.nnet_batch_size)
    if cfg.start_idx is not None:
        add_flag("start_idx", cfg.start_idx)
    if cfg.verbose:
        add_flag("verbose", True)
    if cfg.debug:
        add_flag("debug", True)
    if cfg.save_imgs:
        # qstar parser expects a string or bare flag; pass explicit true
        args_list.extend(["--save_imgs", "true"])

    main(args_list)


@torch.inference_mode()
def main(args_list: list[str] | None = None) -> None:
    """Main function to execute the search algorithm.

    Args:
        args_list (list[str], optional): List of arguments. Defaults to None.
    """
    # arguments
    parser: ArgumentParser = ArgumentParser()
    args_dict: dict[str, Any] = parse_arguments(parser, args_list)

    if not os.path.exists(args_dict["results_dir"]):
        os.makedirs(args_dict["results_dir"])

    results_file: str = f"{args_dict['results_dir']}/results.pkl"
    output_file: str = f"{args_dict['results_dir']}/output.txt"
    if not args_dict["debug"] and not isinstance(sys.stdout, data_utils.Logger):
        sys.stdout = data_utils.Logger(output_file, "w")

    print_args(args_dict)

    save_imgs_dir: str | None = None
    if args_dict["save_imgs"]:
        save_imgs_dir = (
            "ucs_soln_images"
            if args_dict["h_weight"] == 0
            else "qstar_soln_images"
            if args_dict["weight"] != 0
            else "bfs_soln_images"
        )
        save_imgs_dir = os.path.join(args_dict["results_dir"], save_imgs_dir)
        if not os.path.exists(save_imgs_dir):
            os.makedirs(save_imgs_dir)

    # environment
    env = env_utils.get_environment(args_dict["env"])

    # get data
    with open(args_dict["states"], "rb") as f:
        input_data = pickle.load(f)
    states: list[State] = input_data["states"]
    state_goals: list[State] = input_data["state_goals"]

    # initialize results
    if os.path.isfile(results_file):
        with open(results_file, "rb") as file:
            results = pickle.load(file)
        start_idx = len(results["solutions"])
        print("Results file exists")

    else:
        results = {}
        results["states"] = []
        results["solutions"] = []
        results["paths"] = []
        results["times"] = []
        results["num_nodes_generated"] = []
        results["solved"] = []
        results["num_itrs"] = []
        results["path_cost"] = []
        results["len_optimal_path"] = []
        results["is_optimal_path"] = []
        results["num_moves"] = []
        start_idx = 0

    if args_dict["start_idx"] is not None:
        start_idx = args_dict["start_idx"]

    print(f"Starting at idx {start_idx}")

    bwqs_python(
        args_dict, start_idx, env, states, state_goals, results, results_file, args_dict["save_imgs"], save_imgs_dir
    )


if __name__ == "__main__":
    main()
