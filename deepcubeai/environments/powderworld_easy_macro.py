from __future__ import annotations

import numpy as np
from numpy import float32
from numpy.typing import NDArray
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from deepcubeai.environments.environment_abstract import State
from deepcubeai.environments.powderworld_easy import (
    POWDERWORLD_DIFFICULTY_TO_ELEM_NAMES,
    PowderworldEasyEnvironment,
    PowderworldHardEnvironment,
    PowderworldMediumEnvironment,
    PowderworldState,
)


GRID_SIZE = 8
BRUSH_SIZE = 4
EASY_ELEM_NAMES = POWDERWORLD_DIFFICULTY_TO_ELEM_NAMES["easy"]
PALETTE_UINT8 = np.asarray(
    [
        [236, 240, 241],  # empty
        [38, 194, 129],  # plant
        [38, 67, 72],  # stone
    ],
    dtype=np.uint8,
)
WALL_COLOR_UINT8 = np.asarray([127, 127, 127], dtype=np.uint8)


def _palette_for_elem_names(elem_names: tuple[str, ...]) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Return RGB palette for empty + drawable Powderworld elements."""
    if elem_names == EASY_ELEM_NAMES:
        return PALETTE_UINT8, WALL_COLOR_UINT8

    try:
        from ogbench.powderworld.sim import PWRenderer, pw_element_names  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - depends on local OGBench install.
        raise RuntimeError(
            "Generic Powderworld palettes require OGBench. Install mqe-release or add it to PYTHONPATH."
        ) from exc

    renderer = PWRenderer()
    names = ("empty", *elem_names)
    elem_idxs = [pw_element_names.index(name) for name in names]
    colors = np.rint(renderer.elem_vecs_array[elem_idxs] * 255.0).clip(0, 255).astype(np.uint8)
    wall_color = np.rint(renderer.elem_vecs_array[pw_element_names.index("wall")] * 255.0).clip(0, 255).astype(np.uint8)
    return colors, wall_color


def obs_to_classes_for_elems(
    observation: NDArray[np.uint8],
    elem_names: tuple[str, ...] = EASY_ELEM_NAMES,
) -> NDArray[np.int64]:
    """Convert a Powderworld RGB observation to an 8x8 class grid."""
    palette, _ = _palette_for_elem_names(tuple(elem_names))
    world = observation[..., :3].astype(np.float32)
    cell_rgb = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.float32)
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            patch = world[y * BRUSH_SIZE + 1 : y * BRUSH_SIZE + 3, x * BRUSH_SIZE + 1 : x * BRUSH_SIZE + 3]
            cell_rgb[y, x] = patch.mean(axis=(0, 1))
    diffs = cell_rgb[:, :, None, :] - palette.astype(np.float32)[None, None, :, :]
    return np.argmin(np.sum(diffs * diffs, axis=-1), axis=-1).astype(np.int64)


def obs_to_classes(observation: NDArray[np.uint8]) -> NDArray[np.int64]:
    """Convert an easy Powderworld RGB observation to an 8x8 class grid."""
    return obs_to_classes_for_elems(observation, EASY_ELEM_NAMES)


def classes_to_observation_for_elems(
    classes: NDArray[np.int64],
    elem_names: tuple[str, ...] = EASY_ELEM_NAMES,
) -> NDArray[np.uint8]:
    """Render an 8x8 class grid as a phase-0 32x32x6 observation."""
    palette, wall_color = _palette_for_elem_names(tuple(elem_names))
    world = palette[classes].repeat(BRUSH_SIZE, axis=0).repeat(BRUSH_SIZE, axis=1)
    world[0, :, :] = wall_color
    world[-1, :, :] = wall_color
    world[:, 0, :] = wall_color
    world[:, -1, :] = wall_color
    action_frame = np.zeros_like(world)
    return np.concatenate((world, action_frame), axis=-1)


def classes_to_observation(classes: NDArray[np.int64]) -> NDArray[np.uint8]:
    """Render an easy 8x8 class grid as a phase-0 32x32x6 observation."""
    return classes_to_observation_for_elems(classes, EASY_ELEM_NAMES)


class SymbolicMacroEncoder(nn.Module):
    """Deterministic image-to-grid encoder for Powderworld easy phase-0 states.

    The latent is a 3-way one-hot class grid over 8x8 brush cells:
    empty, plant, stone. This is an oracle baseline used to validate Q* on the
    task before replacing it with a learned latent.
    """

    grid_size = GRID_SIZE
    brush_size = BRUSH_SIZE

    def __init__(self, elem_names: tuple[str, ...] = EASY_ELEM_NAMES) -> None:
        super().__init__()
        self.elem_names = tuple(elem_names)
        self.num_classes = len(self.elem_names) + 1

    def _palette(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        # RGB colors observed in Powderworld easy rendered frames.
        palette, _ = _palette_for_elem_names(self.elem_names)
        return torch.tensor(palette.astype(np.float32), device=device, dtype=dtype) / 255.0

    def forward(self, states: Tensor) -> tuple[Tensor, Tensor]:
        world = states[:, :3]
        # Use the central 2x2 pixels of every 4x4 brush cell. This avoids the
        # one-pixel wall border and gives stable classes at map edges.
        rows = torch.arange(1, 32, self.brush_size, device=states.device)
        cols = torch.arange(1, 32, self.brush_size, device=states.device)
        patches = []
        for row in rows:
            row_patches = []
            for col in cols:
                patch = world[:, :, row : row + 2, col : col + 2].mean(dim=(2, 3))
                row_patches.append(patch)
            patches.append(torch.stack(row_patches, dim=2))
        cell_rgb = torch.stack(patches, dim=2)  # (B, 3, 8, 8)

        palette = self._palette(states.device, states.dtype)
        diffs = cell_rgb.permute(0, 2, 3, 1).unsqueeze(-2) - palette.view(1, 1, 1, self.num_classes, 3)
        class_ids = torch.argmin(torch.sum(diffs * diffs, dim=-1), dim=-1)
        one_hot = F.one_hot(class_ids, self.num_classes).permute(0, 3, 1, 2).float()
        enc = one_hot.flatten(start_dim=1)
        return enc, enc


class SymbolicMacroDecoder(nn.Module):
    """Decode 8x8 one-hot class grid back to a 32x32x6 phase-0 observation."""

    grid_size = GRID_SIZE
    brush_size = BRUSH_SIZE

    def __init__(self, elem_names: tuple[str, ...] = EASY_ELEM_NAMES) -> None:
        super().__init__()
        self.elem_names = tuple(elem_names)
        self.num_classes = len(self.elem_names) + 1

    def _palette(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        palette, _ = _palette_for_elem_names(self.elem_names)
        return torch.tensor(palette.astype(np.float32), device=device, dtype=dtype) / 255.0

    def _wall_color(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        _, wall_color = _palette_for_elem_names(self.elem_names)
        return torch.tensor(wall_color.astype(np.float32), device=device, dtype=dtype) / 255.0

    def forward(self, encs: Tensor) -> Tensor:
        batch_size = encs.shape[0]
        classes = encs.view(batch_size, self.num_classes, self.grid_size, self.grid_size).argmax(dim=1)
        palette = self._palette(encs.device, encs.dtype)
        world = palette[classes].permute(0, 3, 1, 2)
        world = world.repeat_interleave(self.brush_size, dim=2).repeat_interleave(self.brush_size, dim=3)
        wall_color = self._wall_color(encs.device, encs.dtype).view(1, 3, 1)
        world[:, :, 0, :] = wall_color
        world[:, :, -1, :] = wall_color
        world[:, :, :, 0] = wall_color
        world[:, :, :, -1] = wall_color
        action_frame = torch.zeros_like(world)
        return torch.cat((world, action_frame), dim=1)


class LearnedMacroEncoder(nn.Module):
    """Small CNN encoder trained to predict the 8x8 symbolic Powderworld grid."""

    grid_size = GRID_SIZE

    def __init__(self, chan_in: int = 6, hidden_dim: int = 64, num_classes: int = 3) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.net = nn.Sequential(
            nn.Conv2d(chan_in, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, self.num_classes, kernel_size=4, stride=4),
        )

    def logits(self, states: Tensor) -> Tensor:
        return self.net(states.float())

    def forward(self, states: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.logits(states)
        probs = F.softmax(logits, dim=1).flatten(start_dim=1)
        class_ids = torch.argmax(logits, dim=1)
        hard = F.one_hot(class_ids, self.num_classes).permute(0, 3, 1, 2).float()
        return probs, hard.flatten(start_dim=1)


class SymbolicMacroEnvModel(nn.Module):
    """Exact macro transition: set one 8x8 brush cell to plant or stone."""

    grid_size = GRID_SIZE

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.num_classes = int(num_classes)

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        batch_size = states.shape[0]
        classes = states.view(batch_size, self.num_classes, self.grid_size, self.grid_size).argmax(dim=1)
        actions_l = actions.long()
        elem = actions_l // 64
        rem = actions_l % 64
        x = rem // 8
        y = rem % 8
        next_classes = classes.clone()
        next_classes[torch.arange(batch_size, device=states.device), y, x] = elem + 1
        return F.one_hot(next_classes, self.num_classes).permute(0, 3, 1, 2).float().flatten(start_dim=1)


class LearnedMacroEnvModel(nn.Module):
    """Neural macro transition model over 8x8 one-hot Powderworld latents."""

    grid_size = GRID_SIZE

    def __init__(self, hidden_dim: int = 64, num_classes: int = 3, num_actions: int = 128) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_actions = int(num_actions)
        self.num_draw_elems = self.num_actions // (self.grid_size * self.grid_size)
        self.net = nn.Sequential(
            nn.Conv2d(self.num_classes + self.num_draw_elems, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, self.num_classes, kernel_size=1),
        )

    def _action_planes(self, actions: Tensor) -> Tensor:
        batch_size = actions.shape[0]
        actions_l = actions.long()
        elem = actions_l // 64
        rem = actions_l % 64
        x = rem // 8
        y = rem % 8
        planes = torch.zeros(batch_size, self.num_draw_elems, self.grid_size, self.grid_size, device=actions.device)
        planes[torch.arange(batch_size, device=actions.device), elem, y, x] = 1.0
        return planes

    def logits(self, states: Tensor, actions: Tensor) -> Tensor:
        states_grid = states.view(-1, self.num_classes, self.grid_size, self.grid_size).float()
        return self.net(torch.cat((states_grid, self._action_planes(actions)), dim=1))

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        logits = self.logits(states, actions)
        return F.softmax(logits, dim=1).flatten(start_dim=1)


class SymbolicMacroHeuristic(nn.Module):
    """Per-action mismatch-count heuristic for Q* over macro actions."""

    grid_size = GRID_SIZE

    def __init__(self, num_classes: int = 3, num_actions: int = 128) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_actions = int(num_actions)

    def forward(self, states: Tensor, states_goal: Tensor) -> Tensor:
        batch_size = states.shape[0]
        classes = states.view(batch_size, self.num_classes, self.grid_size, self.grid_size).argmax(dim=1)
        goals = states_goal.view(batch_size, self.num_classes, self.grid_size, self.grid_size).argmax(dim=1)
        costs = []
        batch_idxs = torch.arange(batch_size, device=states.device)
        for action in range(self.num_actions):
            elem = action // 64
            rem = action % 64
            x = rem // 8
            y = rem % 8
            next_classes = classes.clone()
            next_classes[batch_idxs, y, x] = elem + 1
            # Slightly overweight h to break the enormous permutation plateau:
            # along a good path g + h stays constant, while 1.01*h prefers
            # states with fewer remaining mismatched cells.
            costs.append((next_classes != goals).sum(dim=(1, 2)).float() * 1.01)
        return torch.stack(costs, dim=1)


class LearnedMacroHeuristic(nn.Module):
    """Neural action-value heuristic trained from symbolic mismatch targets."""

    grid_size = GRID_SIZE

    def __init__(self, hidden_dim: int = 64, num_classes: int = 3, num_actions: int = 128) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_actions = int(num_actions)
        self.num_draw_elems = self.num_actions // (self.grid_size * self.grid_size)
        self.before_net = nn.Sequential(
            nn.Conv2d(self.num_classes * 2, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        self.after_net = nn.Sequential(
            nn.Conv2d(self.num_classes * 2, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, self.num_draw_elems, kernel_size=1),
        )

    def mismatch_logits(self, states: Tensor, states_goal: Tensor) -> tuple[Tensor, Tensor]:
        states_grid = states.view(-1, self.num_classes, self.grid_size, self.grid_size).float()
        goals_grid = states_goal.view(-1, self.num_classes, self.grid_size, self.grid_size).float()
        features = torch.cat((states_grid, goals_grid), dim=1)
        return self.before_net(features).squeeze(1), self.after_net(features)

    def forward(self, states: Tensor, states_goal: Tensor) -> Tensor:
        before_logits, after_logits = self.mismatch_logits(states, states_goal)
        before = torch.sigmoid(before_logits)
        after = torch.sigmoid(after_logits)
        total_before = before.sum(dim=(1, 2), keepdim=True)

        action_costs = []
        for elem_idx in range(self.num_draw_elems):
            elem_costs = total_before - before + after[:, elem_idx]
            action_costs.append(elem_costs.permute(0, 2, 1).flatten(start_dim=1))
        costs = torch.cat(action_costs, dim=1) * 1.01
        # Learned values are accurate but tiny numerical differences can choose
        # unstable macro-action orders in the real Powderworld simulator. The
        # bias only breaks ties between equal-cost actions.
        action_bias = torch.arange(self.num_actions, device=states.device, dtype=costs.dtype) * 1e-4
        return costs + action_bias.view(1, -1)


class PowderworldEasyMacroEnvironment(PowderworldEasyEnvironment):
    """Powderworld easy phase-0 macro-action environment for Q* baselines."""

    enc_dim = 3 * 8 * 8
    _num_actions_max = 128
    num_classes = 3
    use_task_tol_for_macro_solved = False

    @property
    def env_name(self) -> str:
        return "powderworld_easy_macro"

    @property
    def num_actions_max(self) -> int:
        return len(self.elem_names) * GRID_SIZE * GRID_SIZE

    def rand_action(self, states: list[State]) -> list[int]:
        return list(np.random.randint(0, self.num_actions_max, size=len(states)))

    def next_state(self, states: list[State], actions: list[int]) -> tuple[list[State], list[float]]:
        next_states: list[State] = []
        costs: list[float] = []
        for state, action in zip(states, actions, strict=False):
            if not isinstance(state, PowderworldState):
                raise TypeError(f"Expected PowderworldState, got {type(state)!r}")
            action_i = int(action)
            elem = action_i // 64
            rem = action_i % 64
            x = rem // 8
            y = rem % 8
            classes = obs_to_classes_for_elems(state.observation, self.elem_names)
            classes[y, x] = elem + 1
            next_states.append(
                PowderworldState(
                    classes_to_observation_for_elems(classes, self.elem_names),
                    seed=state.seed,
                    task_info=state.task_info,
                    task_id=state.task_id,
                )
            )
            costs.append(1.0)
        return next_states, costs

    def is_solved(self, states: list[State], states_goal: list[State]) -> NDArray[np.bool_]:
        solved: list[bool] = []
        for state, goal in zip(states, states_goal, strict=False):
            if isinstance(state, PowderworldState) and isinstance(goal, PowderworldState):
                state_classes = obs_to_classes_for_elems(state.observation, self.elem_names)
                goal_classes = obs_to_classes_for_elems(goal.observation, self.elem_names)
                mismatch = int(np.not_equal(state_classes, goal_classes).sum())
                tol_cells = 0
                if self.use_task_tol_for_macro_solved:
                    task_info = goal.task_info or state.task_info or {}
                    tol_pixels = int(task_info.get("tol", 0))
                    tol_cells = int(np.ceil(tol_pixels / float(BRUSH_SIZE * BRUSH_SIZE)))
                solved.append(mismatch <= tol_cells)
            else:
                solved.append(False)
        return np.asarray(solved, dtype=np.bool_)

    def get_dqn(self) -> nn.Module:
        return SymbolicMacroHeuristic(self.num_classes, self.num_actions_max)

    def get_mqe_heur(self) -> nn.Module:
        return SymbolicMacroHeuristic(self.num_classes, self.num_actions_max)

    def get_env_nnet(self) -> nn.Module:
        return SymbolicMacroEnvModel(self.num_classes)

    def get_env_nnet_cont(self) -> nn.Module:
        return SymbolicMacroEnvModel(self.num_classes)

    def get_encoder(self) -> nn.Module:
        return SymbolicMacroEncoder(self.elem_names)

    def get_decoder(self) -> nn.Module:
        return SymbolicMacroDecoder(self.elem_names)

    def get_goals(self, states: list[State], num_steps: int | None) -> list[State]:
        del num_steps
        goals: list[State] = []
        for state in states:
            if isinstance(state, PowderworldState) and state.goal_observation is not None:
                goal_ob = state.goal_observation.copy()
                goal_ob[..., 3:] = 0
                goals.append(
                    PowderworldState(goal_ob, seed=state.seed, task_info=state.task_info, task_id=state.task_id)
                )
            else:
                raise ValueError("PowderworldState does not contain a goal observation.")
        return goals


class PowderworldEasyMacroLearnedEnvironment(PowderworldEasyMacroEnvironment):
    """Macro Powderworld Q* env with a trainable image-to-symbolic-grid encoder."""

    @property
    def env_name(self) -> str:
        return "powderworld_easy_macro_learned"

    def get_encoder(self) -> nn.Module:
        return LearnedMacroEncoder(self.chan_in, num_classes=self.num_classes)


class PowderworldEasyMacroNeuralEnvironment(PowderworldEasyMacroLearnedEnvironment):
    """Macro Powderworld env with learned encoder, transition model, and heuristic."""

    @property
    def env_name(self) -> str:
        return "powderworld_easy_macro_neural"

    def get_dqn(self) -> nn.Module:
        return LearnedMacroHeuristic(num_classes=self.num_classes, num_actions=self.num_actions_max)

    def get_mqe_heur(self) -> nn.Module:
        from deepcubeai.utils.mqe_models import MQEHeurNet  # noqa: PLC0415

        return MQEHeurNet(
            input_dim=self.enc_dim,
            num_actions=self.num_actions_max,
            latent_dim=256,
            hidden_dims=[256, 256, 256],
            mrn_components=8,
        )

    def get_env_nnet(self) -> nn.Module:
        return LearnedMacroEnvModel(num_classes=self.num_classes, num_actions=self.num_actions_max)

    def get_env_nnet_cont(self) -> nn.Module:
        return LearnedMacroEnvModel(num_classes=self.num_classes, num_actions=self.num_actions_max)


class PowderworldMediumMacroEnvironment(PowderworldEasyMacroEnvironment):
    """Powderworld medium phase-0 macro-action environment for Q* baselines."""

    difficulty = PowderworldMediumEnvironment.difficulty
    elem_names = PowderworldMediumEnvironment.elem_names
    enc_dim = 6 * GRID_SIZE * GRID_SIZE
    num_classes = 6
    use_task_tol_for_macro_solved = True

    @property
    def env_name(self) -> str:
        return "powderworld_medium_macro"


class PowderworldMediumMacroLearnedEnvironment(PowderworldMediumMacroEnvironment):
    """Macro Powderworld medium Q* env with a trainable image-to-grid encoder."""

    @property
    def env_name(self) -> str:
        return "powderworld_medium_macro_learned"

    def get_encoder(self) -> nn.Module:
        return LearnedMacroEncoder(self.chan_in, num_classes=self.num_classes)


class PowderworldMediumMacroNeuralEnvironment(PowderworldMediumMacroLearnedEnvironment):
    """Macro Powderworld medium env with learned encoder, transition model, and heuristic."""

    @property
    def env_name(self) -> str:
        return "powderworld_medium_macro_neural"

    def get_dqn(self) -> nn.Module:
        return LearnedMacroHeuristic(num_classes=self.num_classes, num_actions=self.num_actions_max)

    def get_env_nnet(self) -> nn.Module:
        return LearnedMacroEnvModel(num_classes=self.num_classes, num_actions=self.num_actions_max)

    def get_env_nnet_cont(self) -> nn.Module:
        return LearnedMacroEnvModel(num_classes=self.num_classes, num_actions=self.num_actions_max)


class PowderworldHardMacroEnvironment(PowderworldEasyMacroEnvironment):
    """Powderworld hard phase-0 macro-action environment for Q* baselines."""

    difficulty = PowderworldHardEnvironment.difficulty
    elem_names = PowderworldHardEnvironment.elem_names
    enc_dim = 9 * GRID_SIZE * GRID_SIZE
    num_classes = 9
    use_task_tol_for_macro_solved = True

    @property
    def env_name(self) -> str:
        return "powderworld_hard_macro"


class PowderworldHardMacroLearnedEnvironment(PowderworldHardMacroEnvironment):
    """Macro Powderworld hard Q* env with a trainable image-to-grid encoder."""

    @property
    def env_name(self) -> str:
        return "powderworld_hard_macro_learned"

    def get_encoder(self) -> nn.Module:
        return LearnedMacroEncoder(self.chan_in, num_classes=self.num_classes)


class PowderworldHardMacroNeuralEnvironment(PowderworldHardMacroLearnedEnvironment):
    """Macro Powderworld hard env with learned encoder, transition model, and heuristic."""

    @property
    def env_name(self) -> str:
        return "powderworld_hard_macro_neural"

    def get_dqn(self) -> nn.Module:
        return LearnedMacroHeuristic(num_classes=self.num_classes, num_actions=self.num_actions_max)

    def get_env_nnet(self) -> nn.Module:
        return LearnedMacroEnvModel(num_classes=self.num_classes, num_actions=self.num_actions_max)

    def get_env_nnet_cont(self) -> nn.Module:
        return LearnedMacroEnvModel(num_classes=self.num_classes, num_actions=self.num_actions_max)
