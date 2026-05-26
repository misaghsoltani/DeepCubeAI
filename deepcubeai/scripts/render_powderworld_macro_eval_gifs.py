from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import pickle
from typing import Any

import gymnasium
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import ogbench.powderworld  # noqa: F401
from deepcubeai.environments.powderworld_easy import POWDERWORLD_DIFFICULTY_TO_ELEM_NAMES, PowderworldState
from deepcubeai.scripts.eval_powderworld_macro_qstar_real import macro_to_semantic, restore_env_from_state


def _rgb_from_observation(observation: np.ndarray) -> np.ndarray:
    return np.asarray(observation[..., :3], dtype=np.uint8)


def _scale_rgb(rgb: np.ndarray, scale: int) -> Image.Image:
    img = Image.fromarray(rgb, mode="RGB")
    return img.resize((rgb.shape[1] * scale, rgb.shape[0] * scale), Image.Resampling.NEAREST)


def _label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    x, y = xy
    draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=(245, 245, 245), font=font)


def _make_frame(
    start_rgb: np.ndarray,
    current_rgb: np.ndarray,
    goal_rgb: np.ndarray,
    *,
    scale: int,
    step_text: str,
) -> np.ndarray:
    font = ImageFont.load_default()
    panel_w = start_rgb.shape[1] * scale
    panel_h = start_rgb.shape[0] * scale
    pad = 10
    title_h = 34
    footer_h = 22
    width = panel_w * 3 + pad * 4
    height = title_h + panel_h + footer_h
    canvas = Image.new("RGB", (width, height), color=(24, 28, 31))
    draw = ImageDraw.Draw(canvas)

    panels = [("start", start_rgb), ("current", current_rgb), ("goal", goal_rgb)]
    for idx, (title, rgb) in enumerate(panels):
        x = pad + idx * (panel_w + pad)
        canvas.paste(_scale_rgb(rgb, scale), (x, title_h))
        _label(draw, (x, 10), title, font)

    _label(draw, (pad, title_h + panel_h + 5), step_text, font)
    return np.asarray(canvas)


def _render_episode(
    env: gymnasium.Env,
    state: PowderworldState,
    solution: list[int],
    out_path: Path,
    *,
    elem_names: tuple[str, ...],
    scale: int,
    duration_ms: int,
    final_pause_ms: int,
) -> dict[str, Any]:
    env.reset()
    restore_env_from_state(env, state, elem_names)

    start_rgb = _rgb_from_observation(state.observation)
    if state.goal_observation is None:
        raise ValueError("PowderworldState does not contain goal_observation")
    goal_rgb = _rgb_from_observation(state.goal_observation)
    current_rgb = _rgb_from_observation(env.unwrapped._get_ob())

    frames: list[np.ndarray] = [
        _make_frame(start_rgb, current_rgb, goal_rgb, scale=scale, step_text="step 000 | start")
    ]
    durations: list[int] = [final_pause_ms]

    success = 0.0
    terminated = False
    truncated = False
    primitive_steps = 0
    for macro_idx, action in enumerate(solution, start=1):
        semantic = macro_to_semantic(int(action), elem_names)
        for _ in range(3):
            primitive_action = env.unwrapped.semantic_action_to_action(*semantic)
            ob, reward, terminated, truncated, info = env.step(primitive_action)
            primitive_steps += 1
            success = float(info.get("success", 0.0))
            if terminated or truncated:
                break

        elem_name, x, y = semantic
        current_rgb = _rgb_from_observation(ob)
        frames.append(
            _make_frame(
                start_rgb,
                current_rgb,
                goal_rgb,
                scale=scale,
                step_text=f"step {macro_idx:03d} | {elem_name} x={x} y={y} | success={success:.0f}",
            )
        )
        durations.append(final_pause_ms if terminated or truncated or macro_idx == len(solution) else duration_ms)
        if terminated or truncated:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, duration=[d / 1000.0 for d in durations], loop=0)
    return {
        "gif": str(out_path),
        "macro_steps_rendered": len(frames) - 1,
        "primitive_steps": primitive_steps,
        "success": success,
        "terminated": terminated,
        "truncated": truncated,
    }


def render_gifs(
    results_path: str | Path,
    out_dir: str | Path,
    indices: list[int],
    *,
    difficulty: str,
    scale: int,
    duration_ms: int,
    final_pause_ms: int,
) -> list[dict[str, Any]]:
    results_path = Path(results_path)
    out_dir = Path(out_dir)
    with results_path.open("rb") as f:
        results = pickle.load(f)

    states = results["states"]
    solutions = results["solutions"]
    elem_names = POWDERWORLD_DIFFICULTY_TO_ELEM_NAMES[difficulty]
    env = gymnasium.make(f"powderworld-{difficulty}-v0", mode="task", max_episode_steps=1001)

    summaries: list[dict[str, Any]] = []
    try:
        for idx in indices:
            state = states[idx]
            solution = solutions[idx]
            if not isinstance(state, PowderworldState):
                raise TypeError(f"Expected PowderworldState at index {idx}, got {type(state)!r}")
            if solution is None:
                raise ValueError(f"No solution for index {idx}")

            out_path = out_dir / f"episode_{idx:03d}.gif"
            summary = _render_episode(
                env,
                state,
                solution,
                out_path,
                elem_names=elem_names,
                scale=scale,
                duration_ms=duration_ms,
                final_pause_ms=final_pause_ms,
            )
            summary["idx"] = idx
            summary["seed"] = state.seed
            summaries.append(summary)
            print(summary)
    finally:
        env.close()

    return summaries


def main() -> None:
    parser = ArgumentParser(description="Render Powderworld macro Q* solutions as GIFs.")
    parser.add_argument("--results", required=True, help="Q* results.pkl path.")
    parser.add_argument("--out_dir", required=True, help="Output directory for GIFs.")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 1], help="Episode indices to render.")
    parser.add_argument("--scale", type=int, default=6, help="Nearest-neighbor image scale.")
    parser.add_argument("--duration_ms", type=int, default=140, help="Duration of intermediate GIF frames.")
    parser.add_argument("--final_pause_ms", type=int, default=900, help="Duration for first and final GIF frames.")
    args = parser.parse_args()
    render_gifs(
        args.results,
        args.out_dir,
        args.indices,
        difficulty=args.difficulty,
        scale=args.scale,
        duration_ms=args.duration_ms,
        final_pause_ms=args.final_pause_ms,
    )


if __name__ == "__main__":
    main()
