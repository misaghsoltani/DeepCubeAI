# ========================================================================================== #
# Q* Search is a variant of A* search for DQNs. For more information, see the paper:        `#
# Agostinelli, Forest, et al. "Q* Search: Heuristic Search with Deep Q-Networks." (2024).    #
# https://prl-theworkshop.github.io/prl2024-icaps/papers/9.pdf                               #
# ========================================================================================== #

import os
import pickle
import sys
import time
import typing
from argparse import ArgumentParser
from heapq import heappop, heappush
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from deepcubeai.environments.environment_abstract import Environment, State
from deepcubeai.environments.sokoban import SokobanState
from deepcubeai.utils import data_utils, env_utils, misc_utils, nnet_utils, search_utils
from deepcubeai.utils.data_utils import print_args


class Node:

    def __init__(self, state: np.ndarray, path_cost: float, parent_move: Optional[int],
                 parent: Optional["Node"]):
        """
        Initializes a Node.

        Args:
            state (np.ndarray): The state of the node.
            path_cost (float): The cost to reach this node.
            parent_move (Optional[int]): The move made to reach this node.
            parent (Optional[Node]): The parent node.
        """
        self.state: np.ndarray = state.astype(bool)
        self.path_cost: float = path_cost
        self.parent_move: Optional[int] = parent_move
        self.parent: Optional[Node] = parent

        self.hash = None

    def __hash__(self) -> int:
        """
        Returns the hash of the node.

        Returns:
            int: The hash value.
        """
        if self.hash is None:
            self.hash = hash(self.state.tobytes())

        return self.hash

    def __eq__(self, other: "Node") -> bool:
        """
        Checks if two nodes are equal.

        Args:
            other (Node): The other node to compare.

        Returns:
            bool: True if nodes are equal, False otherwise.
        """
        return np.array_equal(self.state, other.state)


OpenSetElem = Tuple[float, int, Node, int]


class Instance:

    def __init__(self, state: np.ndarray, state_goal: np.ndarray, cost: float):
        """
        Initializes an Instance.

        Args:
            state (np.ndarray): The initial state.
            state_goal (np.ndarray): The goal state.
            cost (float): The initial cost.
        """
        self.state_goal: np.ndarray = state_goal

        self.open_set: List[OpenSetElem] = []

        self.closed_dict: Dict[Node, float] = dict()

        self.heappush_count: int = 0
        self.popped_nodes_cost: List[Tuple[Node, float]] = []
        self.goal_nodes: List[Node] = []
        self.num_nodes_generated: int = 0

        self.root_node: Node = Node(state, 0.0, None, None)

        self.push_to_open([self.root_node], [[-1]], [[cost]])

    def push_to_open(self, nodes: List[Node], moves: List[List[int]],
                     costs: List[List[float]]) -> None:
        """
        Pushes nodes to the open set.

        Args:
            nodes (List[Node]): The nodes to push.
            moves (List[List[int]]): The moves associated with the nodes.
            costs (List[List[float]]): The costs associated with the nodes.
        """
        for node, moves_node, costs_node in zip(nodes, moves, costs):
            for move, cost in zip(moves_node, costs_node):
                heappush(self.open_set, (cost, self.heappush_count, node, move))
                self.heappush_count += 1

    def pop_from_open(self, num_nodes: int) -> Tuple[List[Node], List[int]]:
        """
        Pops nodes from the open set.

        Args:
            num_nodes (int): The number of nodes to pop.

        Returns:
            Tuple[List[Node], List[int]]: The popped nodes and their associated moves.
        """
        num_to_pop: int = min(num_nodes, len(self.open_set))
        popped_elems: List[OpenSetElem] = [heappop(self.open_set) for _ in range(num_to_pop)]
        popped_nodes: List[Node] = [elem[2] for elem in popped_elems]

        popped_nodes_cost: List[Tuple[Node, float]] = [(elem[2], elem[0]) for elem in popped_elems]
        self.popped_nodes_cost.extend(popped_nodes_cost)

        # take moves
        moves: List[int] = [elem[3] for elem in popped_elems]

        return popped_nodes, moves

    def remove_in_closed(self, nodes: List[Node]) -> List[Node]:
        """
        Removes nodes that are in the closed set.

        Args:
            nodes (List[Node]): The nodes to check.

        Returns:
            List[Node]: The nodes not in the closed set.
        """
        nodes_not_in_closed: List[Node] = []

        for node in nodes:
            path_cost_prev: Optional[float] = self.closed_dict.get(node)
            if path_cost_prev is None:
                nodes_not_in_closed.append(node)
                self.closed_dict[node] = node.path_cost

            elif path_cost_prev > node.path_cost:
                nodes_not_in_closed.append(node)
                self.closed_dict[node] = node.path_cost

        return nodes_not_in_closed


def pop_from_open(instances: List[Instance], batch_size: int, model_fn: Callable,
                  is_solved_fn: Callable) -> List[List[Node]]:
    """
    Pops nodes from the open sets of instances.

    Args:
        instances (List[Instance]): The instances to pop nodes from.
        batch_size (int): The batch size.
        model_fn (Callable): The model function.
        is_solved_fn (Callable): The function to check if a state is solved.

    Returns:
        List[List[Node]]: The popped nodes for each instance.
    """
    popped_nodes_by_inst: List[List[Node]] = []
    moves: List[List[int]] = []
    for instance in instances:
        popped_nodes_inst, moves_inst = instance.pop_from_open(batch_size)

        popped_nodes_by_inst.append(popped_nodes_inst)
        moves.append(moves_inst)

    # make moves
    popped_nodes_flat, split_idxs = misc_utils.flatten(popped_nodes_by_inst)
    moves_flat, _ = misc_utils.flatten(moves)
    popped_nodes_next_flat: List[Node] = []

    if moves_flat[0] == -1:  # Initial
        for popped_nodes_inst in popped_nodes_by_inst:
            assert (len(popped_nodes_inst) == 1
                    ), "Initial condition should only happen at the first iteration"
        assert np.all(np.equal(
            moves_flat, -1)), "Initial condition should happen for all at the first iteration"

        states_next_flat: np.ndarray = np.stack(
            [popped_node.state for popped_node in popped_nodes_flat], axis=0)
        popped_nodes_next_flat = popped_nodes_flat
    else:
        # get next states
        states_flat: np.ndarray = np.stack(
            [popped_node.state for popped_node in popped_nodes_flat], axis=0)
        states_next_flat = model_fn(states_flat, moves_flat).round()

        tcs = [1.0] * states_next_flat.shape[0]  # TODO make dynamic

        # make nodes
        path_costs_parent_flat = [popped_node.path_cost for popped_node in popped_nodes_flat]

        path_costs_flat = np.array(path_costs_parent_flat) + np.array(tcs)
        for state, path_cost, move, parent in zip(states_next_flat, path_costs_flat, moves_flat,
                                                  popped_nodes_flat):
            node_move: Node = Node(state, path_cost, move, parent)
            popped_nodes_next_flat.append(node_move)

    # state goals
    state_goals_flat_l: List[np.array] = []
    for instance, popped_nodes_inst in zip(instances, popped_nodes_by_inst):
        state_goals_flat_l.extend([instance.state_goal] * len(popped_nodes_inst))
    state_goals_flat: np.ndarray = np.stack(state_goals_flat_l, axis=0)

    # check if solved
    is_solved_flat: List[bool] = list(is_solved_fn(states_next_flat, state_goals_flat))
    is_solved_by_inst: List[List[bool]] = misc_utils.unflatten(is_solved_flat, split_idxs)

    popped_nodes_next_by_inst: List[List[Node]] = misc_utils.unflatten(
        popped_nodes_next_flat, split_idxs)

    # update instances
    for instance, popped_nodes_move_inst, is_solved_inst in zip(instances,
                                                                popped_nodes_next_by_inst,
                                                                is_solved_by_inst):
        instance.goal_nodes.extend(
            [node for node, is_solved in zip(popped_nodes_move_inst, is_solved_inst) if is_solved])
        instance.num_nodes_generated += len(popped_nodes_move_inst)

    return popped_nodes_next_by_inst


def add_heuristic_and_cost(
    nodes: List[List[Node]], state_goals_flat: np.ndarray, heuristic_fn: Callable,
    weights: List[float], num_actions_max: int
) -> Tuple[List[List[List[int]]], List[List[List[float]]], np.ndarray, np.ndarray]:
    """
    Adds heuristic and cost to nodes.

    Args:
        nodes (List[List[Node]]): The nodes to add heuristic and cost to.
        state_goals_flat (np.ndarray): The flattened goal states.
        heuristic_fn (Callable): The heuristic function.
        weights (List[float]): The weights for the path costs.
        num_actions_max (int): The maximum number of actions.

    Returns:
        Tuple[List[List[List[int]]], List[List[List[float]]], np.ndarray, np.ndarray]: The moves,
            costs, parent path costs, and heuristics.
    """
    nodes_flat: List[Node]
    nodes_flat, split_idxs = misc_utils.flatten(nodes)

    if len(nodes_flat) == 0:
        return [], [], np.zeros(0), np.zeros(0)

    # get heuristic
    states_flat: np.ndarray = np.stack([node.state for node in nodes_flat], axis=0)
    path_costs_parent: np.ndarray = np.array([node.path_cost for node in nodes_flat])

    # compute node cost
    # If performing Q* search
    if heuristic_fn is not None:
        heuristics_flat: np.ndarray = heuristic_fn(states_flat, state_goals_flat)

    # If performing Uniform Cost Search
    else:
        heuristics_flat: np.ndarray = np.zeros((states_flat.shape[0], num_actions_max))

    path_cost_weighted: np.ndarray = np.expand_dims(weights * path_costs_parent, 1)
    costs_flat: List[List[float]] = (heuristics_flat + path_cost_weighted).tolist()
    moves_flat: List[List[int]] = [list(range(0, len(x))) for x in costs_flat]

    moves: List[List[List[int]]] = misc_utils.unflatten(moves_flat, split_idxs)
    costs: List[List[List[float]]] = misc_utils.unflatten(costs_flat, split_idxs)

    return moves, costs, path_costs_parent, heuristics_flat.min(axis=1)


def add_to_open(instances: List[Instance], nodes: List[List[Node]], moves: List[List[List[int]]],
                costs: List[List[List[float]]]) -> None:
    """
    Adds nodes to the open sets of instances.

    Args:
        instances (List[Instance]): The instances to add nodes to.
        nodes (List[List[Node]]): The nodes to add.
        moves (List[List[List[int]]]): The moves associated with the nodes.
        costs (List[List[List[float]]]): The costs associated with the nodes.
    """
    for instance, nodes_inst, moves_inst, costs_inst in zip(instances, nodes, moves, costs):
        instance.push_to_open(nodes_inst, moves_inst, costs_inst)


def get_path(node: Node) -> Tuple[List[np.ndarray], List[int], float]:
    """
    Gets the path from the root to the given node.

    Args:
        node (Node): The node to trace back from.

    Returns:
        Tuple[List[np.ndarray], List[int], float]: The path, moves, and path cost.
    """
    path: List[np.ndarray] = []
    moves: List[int] = []

    parent_node: Node = node
    while parent_node.parent is not None:
        path.append(parent_node.state)
        moves.append(parent_node.parent_move)
        parent_node = parent_node.parent

    path.append(parent_node.state)

    path = path[::-1]
    moves = moves[::-1]

    return path, moves, node.path_cost


def get_is_solved_fn(per_eq_tol: float) -> Callable:
    """
    Gets the function to check if states are solved.

    Args:
        per_eq_tol (float): The tolerance for equality.

    Returns:
        Callable: The function to check if states are solved.
    """

    def is_solved_fn(states: np.ndarray, states_comp: np.ndarray) -> np.ndarray:
        return (100 * np.equal(states, states_comp).mean(axis=1)) >= per_eq_tol

    return is_solved_fn


class QStarImag:

    def __init__(self, states: np.ndarray, state_goals: np.ndarray, heuristic_fn: Callable,
                 weights: List[float], num_actions_max: int):
        """
        Initializes an QStarImag instance.

        Args:
            states (np.ndarray): The initial states.
            state_goals (np.ndarray): The goal states.
            heuristic_fn (Callable): The heuristic function.
            weights (List[float]): The weights for the path costs.
            num_actions_max (int): The maximum number of actions.
        """
        self.weights: List[float] = weights
        self.step_num: int = 0
        self.num_actions_max: int = num_actions_max

        self.timings: Dict[str, float] = {
            "pop": 0.0,
            "closed": 0.0,
            "heur": 0.0,
            "add": 0.0,
            "itr": 0.0
        }

        # compute starting costs
        # If performing Q* search
        if heuristic_fn is not None:
            costs: np.ndarray = heuristic_fn(states, state_goals).min(axis=1)

        # If performing Uniform Cost Search
        else:
            costs: np.ndarray = np.zeros((1, ))

        # initialize instances
        self.instances: List[Instance] = []

        state: np.ndarray
        for state, state_goal, cost in zip(states, state_goals, costs):
            self.instances.append(Instance(state, state_goal, cost))

        self.last_node: Node = None

    # TODO make separate is_solved_fn and is_same_fn
    def step(self,
             heuristic_fn: Callable,
             model_fn: Callable,
             is_solved_fn: Callable,
             batch_size: int,
             verbose: bool = False) -> bool:
        """
        Performs a step in the Q* search.

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
            instance for instance in self.instances
            if (len(instance.goal_nodes) == 0) and len(instance.open_set) > 0
        ]
        if len(instances) == 0:
            print("Open set is empty. Returning the result ...")
            return False

        # Pop from open
        start_time = time.time()
        popped_nodes: List[List[Node]] = pop_from_open(instances, batch_size, model_fn,
                                                       is_solved_fn)
        pop_time = time.time() - start_time

        # check if popped nodes are in closed
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
        state_goals_flat_l: List[np.array] = []
        for instance, popped_nodes_inst in zip(instances, popped_nodes):
            state_goals_flat_l.extend([instance.state_goal] * len(popped_nodes_inst))

        if len(state_goals_flat_l) > 0:
            state_goals_flat: np.ndarray = np.stack(state_goals_flat_l, axis=0)
        else:
            state_goals_flat: np.ndarray = np.empty((0, 0))

        weights, _ = misc_utils.flatten(
            [[weight] * len(popped_nodes_inst)
             for weight, popped_nodes_inst in zip(self.weights, popped_nodes)])
        moves, costs, path_costs, heuristics = add_heuristic_and_cost(
            popped_nodes, state_goals_flat, heuristic_fn, weights, self.num_actions_max)
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

                print(f"Itr: {self.step_num}, Added to OPEN - Min/Max Heur(PathCost): "
                      f"{min_heur:.2f}({min_heur_pc:.2f})/{max_heur:.2f}({max_heur_pc:.2f}) ")

            print(
                f"Times - pop: {pop_time:.2f}, closed: {closed_time:.2f}, heur: {heur_time:.2f}, "
                f"add: {add_time:.2f}, itr: {itr_time:.2f}")

        # Update timings
        self.timings["pop"] += pop_time
        self.timings["closed"] += closed_time
        self.timings["heur"] += heur_time
        self.timings["add"] += add_time
        self.timings["itr"] += itr_time

        self.step_num += 1

        return True

    def has_found_goal(self) -> List[bool]:
        """
        Checks if the goal has been found for each instance.

        Returns:
            List[bool]: List indicating if the goal has been found for each instance.
        """
        goal_found: List[bool] = [
            len(self.get_goal_nodes(idx)) > 0 for idx in range(len(self.instances))
        ]
        return goal_found

    def get_goal_nodes(self, inst_idx: int) -> List[Node]:
        """
        Gets the goal nodes for a given instance.

        Args:
            inst_idx (int): The index of the instance.

        Returns:
            List[Node]: The goal nodes.
        """
        return self.instances[inst_idx].goal_nodes

    def get_goal_node_smallest_path_cost(self, inst_idx: int) -> Node:
        """
        Gets the goal node with the smallest path cost for a given instance.

        Args:
            inst_idx (int): The index of the instance.

        Returns:
            Node: The goal node with the smallest path cost.
        """
        goal_nodes: List[Node] = self.get_goal_nodes(inst_idx)
        path_costs: List[float] = [node.path_cost for node in goal_nodes]

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


def parse_arguments(parser: ArgumentParser, args_list: List[str] = None) -> Dict[str, Any]:
    """Parses command-line arguments.

    Args:
        parser (ArgumentParser): The argument parser.
        args_list (List[str], optional): List of arguments. Defaults to None.

    Returns:
        Dict[str, Any]: Dictionary of parsed arguments.
    """
    parser.add_argument(
        "--h_weight",
        type=float,
        default=1.0,
        help="Weight of the heuristics. Set it to 0 for performing a Uniform Cost Search")
    # Parse known arguments first for the value of --h_weight
    args, _ = parser.parse_known_args(args_list)
    # print(vars(args))

    if int(args.h_weight) != 0.0:
        parser.add_argument("--heur",
                            type=str,
                            required=True,
                            help="Directory of heuristic function")
    else:
        parser.add_argument("--heur",
                            type=str,
                            default=None,
                            help="Directory of heuristic function")

    parser.add_argument("--states",
                        type=str,
                        required=True,
                        help="File containing states to solve")
    parser.add_argument("--env",
                        type=str,
                        required=True,
                        help="Environment: cube3, iceslider, digitjump, sokoban")

    parser.add_argument("--env_model", type=str, required=True, help="Directory of env model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for BWQS")
    parser.add_argument("--weight", type=float, default=1.0, help="Weight of path cost")

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

    parser.add_argument(
        "--per_eq_tol",
        type=float,
        required=True,
        help="Percent of latent state elements that need to be equal to declare equal")

    parser.add_argument("--verbose", action="store_true", default=False, help="Set for verbose")
    parser.add_argument("--debug", action="store_true", default=False, help="Set when debugging")

    # If provided as --save_imgs 'true', then args.save_imgs will be 'true'
    # If provided as --save_imgs (without any value), then args.save_imgs will be 'true'
    # If is not provided --save_imgs at all, then args.save_imgs will be 'false'
    parser.add_argument("--save_imgs",
                        type=str,
                        nargs="?",
                        const="true",
                        default="false",
                        help="Save the images of the steps of solving each state to file")

    # parse arguments
    args = parser.parse_args(args_list)

    if args.h_weight == 0.0:
        assert args.weight != 0, "h_weight and weight cannot be 0 at the same time"
        args.heur = None

    if args.save_imgs.lower() in ("true", "1"):
        args.save_imgs = True
    elif args.save_imgs.lower() in ("false", "0"):
        args.save_imgs = False
    else:
        raise ValueError("Invalid value for '--save_imgs'. Expected 'true', '1', 'false', or '0'.")

    args_dict: Dict[str, Any] = vars(args)

    return args_dict


def main(args_list: List[str] = None) -> None:
    """Main function to execute the search algorithm.

    Args:
        args_list (List[str], optional): List of arguments. Defaults to None.
    """
    # arguments
    parser: ArgumentParser = ArgumentParser()
    args_dict: Dict[str, Any] = parse_arguments(parser, args_list)

    if not os.path.exists(args_dict["results_dir"]):
        os.makedirs(args_dict["results_dir"])

    results_file: str = f"{args_dict['results_dir']}/results.pkl"
    output_file: str = f"{args_dict['results_dir']}/output.txt"
    if not args_dict["debug"]:
        sys.stdout = data_utils.Logger(output_file, "w")

    print_args(args_dict)

    save_imgs_dir: str = None
    if args_dict["save_imgs"]:
        save_imgs_dir = ("ucs_soln_images" if args_dict["h_weight"] == 0 else
                         "qstar_soln_images" if args_dict["weight"] != 0 else "bfs_soln_images")
        save_imgs_dir = os.path.join(args_dict["results_dir"], save_imgs_dir)
        if not os.path.exists(save_imgs_dir):
            os.makedirs(save_imgs_dir)

    # environment
    env: Environment = env_utils.get_environment(args_dict["env"])

    # get data
    input_data = pickle.load(open(args_dict["states"], "rb"))
    states: List[State] = input_data["states"]
    state_goals: List[State] = input_data["state_goals"]

    # initialize results
    if os.path.isfile(results_file):
        with open(results_file, "rb") as file:
            results: Dict[str, Any] = pickle.load(file)
        start_idx: int = len(results["solutions"])
        print("Results file exists")

    else:
        results: Dict[str, Any] = dict()
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
        start_idx: int = 0

    if args_dict["start_idx"] is not None:
        start_idx: int = args_dict["start_idx"]

    print(f"Starting at idx {start_idx}")

    bwqs_python(args_dict, start_idx, env, states, state_goals, results, results_file,
                args_dict["save_imgs"], save_imgs_dir)


def bwqs_python(args_dict: Dict[str, Any], start_idx: int, env: Environment, states: List[State],
                state_goals: List[State], results: Dict[str, Any], results_file: str,
                save_imgs: bool, save_imgs_dir: Optional[str]) -> None:
    """Performs the batched and weighted version of Q* search algorithm.

    Args:
        args_dict (Dict[str, Any]): Dictionary of arguments.
        start_idx (int): Starting index.
        env (Environment): The environment.
        states (List[State]): List of states.
        state_goals (List[State]): List of goal states.
        results (Dict[str, Any]): Dictionary to store results.
        results_file (str): Path to the results file.
        save_imgs (bool): Whether to save images. If this is True, save_imgs_dir must be provided.
        save_imgs_dir (Optional[str]): Directory to save images. This will be used only if
            save_imgs is True.
    """
    # get device
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()

    print(f"device: {device}, devices: {devices}, on_gpu: {on_gpu}"
          f"\n___________________________________\n")

    # Check to see if performing UCS or Q*
    # If performing UCS, no need to load the heuristic function
    heuristic_fn = None
    if args_dict["h_weight"] != 0:
        heuristic_fn = nnet_utils.load_heuristic_fn(args_dict["heur"],
                                                    device,
                                                    on_gpu,
                                                    env.get_dqn(),
                                                    clip_zero=True,
                                                    batch_size=args_dict["nnet_batch_size"])

    env_model_file: str = f"{args_dict['env_model']}/env_state_dict.pt"
    model_fn = nnet_utils.load_model_fn(env_model_file,
                                        device,
                                        on_gpu,
                                        env.get_env_nnet(),
                                        batch_size=args_dict["nnet_batch_size"])

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
        state_enc_np = state_enc.cpu().data.numpy()

        if args_dict["env"] == "sokoban":
            state_goal: SokobanState = typing.cast(SokobanState, state_goal)

            # get blank indexes
            all_idxs: Set[Tuple[int, int]] = set()
            for x in range(10):
                for y in range(10):
                    all_idxs.add((x, y))

            wall_where = np.where(state_goal.walls)
            box_where = np.where(state_goal.boxes)

            wall_idxs: Set[Tuple] = set(zip(wall_where[0], wall_where[1]))
            box_idxs: Set[Tuple] = set(zip(box_where[0], box_where[1]))

            blocked_idxs: Set[Tuple[int, int]] = set()
            blocked_idxs.update(wall_idxs)
            blocked_idxs.update(box_idxs)

            blank_idxs = all_idxs - blocked_idxs

            state_goals_i: List[SokobanState] = []
            for blank_idx in blank_idxs:
                state_goal_i: SokobanState = SokobanState(blank_idx, state_goal.boxes,
                                                          state_goal.walls)
                state_goals_i.append(state_goal_i)

            state_goals_i_real = env.state_to_real(state_goals_i)

            state_goals_i_enc = encoder(torch.tensor(state_goals_i_real, device=device).float())[1]
            state_goals_i_enc_np = state_goals_i_enc.cpu().data.numpy()

            states_enc_np = np.repeat(state_enc_np, state_goals_i_enc_np.shape[0], axis=0)

            # do Q* search
            qstar = QStarImag(states_enc_np,
                              state_goals_i_enc_np,
                              heuristic_fn,
                              weights=[args_dict["weight"]] * states_enc_np.shape[0],
                              num_actions_max=num_actions_max)
            while res and not max(qstar.has_found_goal()):
                res = qstar.step(heuristic_fn,
                                 model_fn,
                                 is_solved_fn,
                                 args_dict["batch_size"],
                                 verbose=args_dict["verbose"])
                num_itrs += 1

            if res:
                # get goal node
                solved_idxs: np.ndarray = np.where(qstar.has_found_goal())[0]
                goal_node: Node = qstar.get_goal_node_smallest_path_cost(solved_idxs[0])
                for solved_idx in solved_idxs[1:]:
                    goal_node_i: Node = qstar.get_goal_node_smallest_path_cost(solved_idx)
                    if goal_node_i.path_cost < goal_node.path_cost:
                        goal_node = goal_node_i
        else:
            state_goal_real = env.state_to_real([state_goal])
            state_goal_enc = encoder(torch.tensor(state_goal_real, device=device).float())[1]
            state_goal_enc_np = state_goal_enc.cpu().data.numpy()

            qstar = QStarImag(state_enc_np,
                              state_goal_enc_np,
                              heuristic_fn,
                              weights=[args_dict["weight"]],
                              num_actions_max=num_actions_max)
            while res and not min(qstar.has_found_goal()):
                res = qstar.step(heuristic_fn,
                                 model_fn,
                                 is_solved_fn,
                                 args_dict["batch_size"],
                                 verbose=args_dict["verbose"])
                num_itrs += 1

            if res:
                goal_node: Node = qstar.get_goal_node_smallest_path_cost(0)

        # If the open set became empty without fidning a solution, use the last node.
        # Used only for saving the image, when save_imgs is True
        if not res:
            goal_node = qstar.last_node

        path: List[np.array]
        soln: List[int]
        path_cost: float
        num_nodes_gen_idx: int

        path, soln, path_cost = get_path(goal_node)

        num_nodes_gen_idx: int = qstar.get_num_nodes_generated(0)

        solve_time = time.time() - start_time

        # check soln
        solved: bool = False
        if save_imgs:
            solved = search_utils.is_valid_soln(state, state_goal, soln, env, decoder, device,
                                                state_idx, path, save_imgs_dir, save_imgs)

        else:
            solved = search_utils.is_valid_soln(state, state_goal, soln, env)

        # assert search_utils.is_valid_soln(state, state_goal, soln, env)
        nodes_per_sec = num_nodes_gen_idx / solve_time

        # print to screen
        timing_str = ", ".join([f"{key}: {val:.2f}" for key, val in qstar.timings.items()])
        print(f"Times - {timing_str}, "
              f"num_itrs: {num_itrs}\n"
              f"State: {state_idx}, "
              f"Solved: {'Yes' if solved else 'No'}, "
              f"SolnCost: {path_cost:.2f}, "
              f"# Moves: {len(soln)}, "
              f"# Nodes Gen: {num_nodes_gen_idx:,}, "
              f"Time: {solve_time:.2f}, "
              f"Nodes/Sec: {nodes_per_sec:.2E}\n"
              f"___________________________________\n")

        # If the State class implements the get_opt_path_len() mehtod for getting the optimal path
        has_get_opt_path_len: bool = hasattr(state, "get_opt_path_len")

        soln = soln if solved else None
        path_cost = path_cost if solved else None
        path = path if solved else None
        num_moves = len(soln) if solved else None
        results["states"].append(state)
        results["solutions"].append(soln)
        results["paths"].append(path)
        results["times"].append(solve_time)
        results["num_nodes_generated"].append(num_nodes_gen_idx)
        results["solved"].append(solved)
        results["num_itrs"].append(num_itrs)
        results["path_cost"].append(path_cost)
        results["num_moves"].append(num_moves)

        len_optimal_path: int = None
        is_optimal_path: int = None
        if has_get_opt_path_len:
            len_optimal_path = state.get_opt_path_len()
            if soln is not None:
                is_optimal_path = bool(len(soln) <= state.get_opt_path_len())

        results["len_optimal_path"].append(len_optimal_path)
        results["is_optimal_path"].append(is_optimal_path)

        with open(results_file, "wb") as f:
            pickle.dump(results, f, protocol=-1)

    avg_time = np.mean(results["times"])
    avg_num_nodes_generated = np.mean(results["num_nodes_generated"])
    avg_nodes_per_sec = avg_num_nodes_generated / avg_time
    avg_moves = _get_mean(results, "num_moves")
    avg_itrs = _get_mean(results, "num_itrs")
    states_total = len(results["solved"])
    solved_total = np.sum(results["solved"])
    solved_perc = (solved_total / states_total) * 100
    optimal_percent = np.mean(results["is_optimal_path"]) * 100

    print(f"\nSummary:\n"
          f"Number of Solved States: {solved_total}, Total Number of States: {states_total}, "
          f"Success Rate: {solved_perc:.2f}%\nAvg # Moves: {avg_moves:.2f}, "
          f"Optimal: {optimal_percent:.2f}%, Avg Itrs: {avg_itrs:.2f}, "
          f"Avg # Nodes Gen: {avg_num_nodes_generated:.2f}, "
          f"Avg Time: {avg_time:.2f}, Avg Nodes/Sec: {avg_nodes_per_sec:.2E}")


def _get_mean(results: Dict[str, Any], key: str) -> float:
    """Calculates the mean of the specified key in the results dictionary.

    Args:
        results (Dict[str, Any]): Dictionary of results.
        key (str): The key to calculate the mean for.

    Returns:
        float: The mean value.
    """
    vals: List = [x for x, solved in zip(results[key], results["solved"]) if solved]
    if len(vals) == 0:
        return 0

    mean_val = np.mean([x for x, solved in zip(results[key], results["solved"]) if solved])
    return float(mean_val)


if __name__ == "__main__":
    main()
