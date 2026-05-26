from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import pickle
from typing import Any

import gymnasium
import numpy as np

import ogbench.powderworld  # noqa: F401
from deepcubeai.environments.powderworld_easy import POWDERWORLD_DIFFICULTY_TO_ELEM_NAMES, PowderworldState
from deepcubeai.environments.powderworld_easy_macro import obs_to_classes_for_elems


def _elem_ids(elem_names: tuple[str, ...]) -> dict[int, int]:
    from ogbench.powderworld.sim import pw_element_names  # noqa: PLC0415

    out = {0: pw_element_names.index("empty")}
    for class_id, elem_name in enumerate(elem_names, start=1):
        out[class_id] = pw_element_names.index(elem_name)
    return out


def macro_to_semantic(
    action: int,
    elem_names: tuple[str, ...] = POWDERWORLD_DIFFICULTY_TO_ELEM_NAMES["easy"],
) -> tuple[str, int, int]:
    elem = action // 64
    rem = action % 64
    x = rem // 8
    y = rem % 8
    return elem_names[elem], x, y


def classes_to_world_ids(classes: np.ndarray, elem_names: tuple[str, ...]) -> np.ndarray:
    class_to_elem_id = _elem_ids(elem_names)
    from ogbench.powderworld.sim import pw_element_names  # noqa: PLC0415

    world_ids = np.zeros((32, 32), dtype=np.uint8)
    for cls, elem_id in class_to_elem_id.items():
        mask = classes == cls
        world_ids[mask.repeat(4, axis=0).repeat(4, axis=1)] = elem_id
    wall_id = pw_element_names.index("wall")
    world_ids[0, :] = wall_id
    world_ids[-1, :] = wall_id
    world_ids[:, 0] = wall_id
    world_ids[:, -1] = wall_id
    return world_ids


def restore_env_from_state(
    env: gymnasium.Env,
    state: PowderworldState,
    elem_names: tuple[str, ...] = POWDERWORLD_DIFFICULTY_TO_ELEM_NAMES["easy"],
    tol: int | None = None,
) -> None:
    start_ids = classes_to_world_ids(obs_to_classes_for_elems(state.observation, elem_names), elem_names)
    goal_observation = state.goal_observation
    if goal_observation is None:
        raise ValueError("PowderworldState does not contain goal_observation")
    goal_ids = classes_to_world_ids(obs_to_classes_for_elems(goal_observation, elem_names), elem_names)
    tol_value = int(tol if tol is not None else (state.task_info or {}).get("tol", 32))

    unwrapped = env.unwrapped
    unwrapped._world = unwrapped.pw.np_to_pw(start_ids[None]).copy()
    unwrapped._action_step = 0
    unwrapped._action_elem_id = None
    unwrapped._action_x = None
    unwrapped._action_y = None
    unwrapped.cur_goal_world = goal_ids.copy()
    unwrapped.cur_task_info = dict(state.task_info or {})
    unwrapped.cur_task_info["tol"] = tol_value


def compute_mismatches(
    env: gymnasium.Env,
    observation: np.ndarray,
    goal_observation: np.ndarray,
    elem_names: tuple[str, ...],
) -> dict[str, int]:
    cur_world = env.unwrapped._world[0, 0].copy()
    goal_world = env.unwrapped.cur_goal_world.copy()
    world_shifts = []
    for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
        world_shifts.append(np.roll(cur_world, (dy, dx), axis=(0, 1)))
    world_shifts = np.stack(world_shifts, axis=0)
    shift_match = (goal_world == world_shifts).any(axis=0)
    return {
        "exact_pixel_mismatch": int(np.not_equal(cur_world, goal_world).sum()),
        "shift_tolerant_mismatch": int((~shift_match).sum()),
        "class_mismatch_8x8": int(
            np.not_equal(
                obs_to_classes_for_elems(observation, elem_names),
                obs_to_classes_for_elems(goal_observation, elem_names),
            ).sum()
        ),
    }


def eval_results(
    results_path: str | Path,
    output_path: str | Path | None = None,
    tol: int | None = None,
    difficulty: str = "easy",
) -> dict[str, Any]:
    results_path = Path(results_path)
    with results_path.open("rb") as f:
        results = pickle.load(f)

    elem_names = POWDERWORLD_DIFFICULTY_TO_ELEM_NAMES[difficulty]
    states = results["states"]
    solutions = results["solutions"]
    env = gymnasium.make(f"powderworld-{difficulty}-v0", mode="task", max_episode_steps=1001)

    episode_results: list[dict[str, Any]] = []
    for idx, (state, solution) in enumerate(zip(states, solutions, strict=False)):
        if not isinstance(state, PowderworldState):
            raise TypeError(f"Expected PowderworldState at index {idx}, got {type(state)!r}")

        env.reset()
        restore_env_from_state(env, state, elem_names, tol=tol)
        tol_value = int(tol if tol is not None else (state.task_info or {}).get("tol", 32))
        reset_matches = True
        success = 0.0
        reward = 0.0
        terminated = False
        truncated = False
        primitive_steps = 0

        if solution is not None:
            for action in solution:
                semantic = macro_to_semantic(int(action), elem_names)
                for _ in range(3):
                    primitive_action = env.unwrapped.semantic_action_to_action(*semantic)
                    ob, reward, terminated, truncated, info = env.step(primitive_action)
                    primitive_steps += 1
                    success = float(info.get("success", 0.0))
                    if terminated or truncated:
                        break
                if terminated or truncated:
                    break

        if state.goal_observation is None:
            raise ValueError("PowderworldState does not contain goal_observation")
        final_ob = env.unwrapped._get_ob()
        mismatches = compute_mismatches(env, final_ob, state.goal_observation, elem_names)

        episode_results.append(
            {
                "idx": idx,
                "seed": state.seed,
                "task_id": state.task_id,
                "task_name": None if state.task_info is None else state.task_info.get("task_name"),
                "tol": tol_value,
                "reset_matches": reset_matches,
                "success": success,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "macro_steps": 0 if solution is None else len(solution),
                "primitive_steps": primitive_steps,
                **mismatches,
            }
        )

    env.close()

    success_rate = float(np.mean([ep["success"] for ep in episode_results])) if episode_results else 0.0
    reset_match_rate = float(np.mean([ep["reset_matches"] for ep in episode_results])) if episode_results else 0.0
    summary = {
        "results_path": str(results_path),
        "difficulty": difficulty,
        "tol": tol,
        "num_episodes": len(episode_results),
        "success_rate": success_rate,
        "reset_match_rate": reset_match_rate,
        "avg_exact_pixel_mismatch": float(np.mean([ep["exact_pixel_mismatch"] for ep in episode_results]))
        if episode_results
        else 0.0,
        "avg_shift_tolerant_mismatch": float(np.mean([ep["shift_tolerant_mismatch"] for ep in episode_results]))
        if episode_results
        else 0.0,
        "avg_class_mismatch_8x8": float(np.mean([ep["class_mismatch_8x8"] for ep in episode_results]))
        if episode_results
        else 0.0,
        "episodes": episode_results,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = ArgumentParser(description="Execute Powderworld macro Q* solutions in the real OGBench env.")
    parser.add_argument("--results", required=True, help="Q* results.pkl path.")
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--tol", type=int, default=None, help="Override Powderworld success tolerance.")
    args = parser.parse_args()
    eval_results(args.results, args.out, tol=args.tol, difficulty=args.difficulty)


if __name__ == "__main__":
    main()
