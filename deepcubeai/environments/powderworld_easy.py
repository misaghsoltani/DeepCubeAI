from __future__ import annotations

from typing import Any

import numpy as np
from numpy import float32, uint8
from numpy.typing import NDArray
import torch
from torch import Tensor, nn
from torch.autograd import Function as AutogradFunc
import torch.nn.functional as F

from deepcubeai.environments.environment_abstract import Environment, State
from deepcubeai.utils.pytorch_models import Conv2dModel, FullyConnectedModel, ResnetConv2dModel, STEThresh


class PowderworldState(State):
    """Image observation state for OGBench Powderworld.

    The observation is stored as HWC uint8 with 6 channels:
    RGB world frame plus RGB action-stage frame.
    """

    def __init__(
        self,
        observation: NDArray[uint8],
        goal_observation: NDArray[uint8] | None = None,
        seed: int | None = None,
        task_info: dict[str, Any] | None = None,
        task_id: int | None = None,
    ) -> None:
        super().__init__()
        self.observation: NDArray[uint8] = np.asarray(observation, dtype=uint8)
        self.goal_observation: NDArray[uint8] | None = (
            None if goal_observation is None else np.asarray(goal_observation, dtype=uint8)
        )
        self.seed = seed
        self.task_info = None if task_info is None else dict(task_info)
        self.task_id = task_id

    def __hash__(self) -> int:
        return hash(self.observation.tobytes())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PowderworldState) and np.array_equal(self.observation, other.observation)


class PowderworldDQN(nn.Module):
    """Goal-conditioned heuristic network over binary Powderworld latents."""

    def __init__(
        self,
        chan_enc: int,
        enc_hw: tuple[int, int],
        resnet_chan: int,
        num_resnet_blocks: int,
        num_actions: int,
        batch_norm: bool,
    ) -> None:
        super().__init__()

        fc_in = resnet_chan * enc_hw[0] * enc_hw[1]
        use_bias_with_norm = False
        self.chan_enc = chan_enc
        self.enc_hw = enc_hw

        self.dqn = nn.Sequential(
            Conv2dModel(
                chan_enc * 2,
                [resnet_chan],
                [3],
                [1],
                [False],
                ["RELU"],
                group_norms=[1],
                use_bias_with_norm=use_bias_with_norm,
            ),
            ResnetConv2dModel(
                resnet_chan,
                resnet_chan,
                resnet_chan,
                3,
                1,
                num_resnet_blocks,
                batch_norm,
                "RELU",
                group_norm=0,
                use_bias_with_norm=use_bias_with_norm,
            ),
            nn.Flatten(),
            FullyConnectedModel(
                fc_in,
                [3 * fc_in, num_actions],
                [batch_norm, batch_norm],
                ["RELU", "LINEAR"],
                use_bias_with_norm=use_bias_with_norm,
            ),
        )

    def forward(self, states: Tensor, states_goal: Tensor) -> Tensor:
        states_conv = states.view(-1, self.chan_enc, self.enc_hw[0], self.enc_hw[1])
        goals_conv = states_goal.view(-1, self.chan_enc, self.enc_hw[0], self.enc_hw[1])
        return self.dqn(torch.cat((states_conv.float(), goals_conv.float()), dim=1))


class Encoder(nn.Module):
    """Binary 256-bit encoder for Powderworld observations."""

    def __init__(self, chan_in: int, chan_enc: int) -> None:
        super().__init__()
        use_bias_with_norm = False
        self.encoder = nn.Sequential(
            Conv2dModel(
                chan_in,
                [32, chan_enc],
                [4, 2],
                [0, 0],
                [True, False],
                ["RELU", "SIGMOID"],
                strides=[4, 2],
                group_norms=[0, 0],
                use_bias_with_norm=use_bias_with_norm,
            ),
            nn.Flatten(),
        )
        self.ste_thresh: AutogradFunc = STEThresh()

    def forward(self, states: Tensor) -> tuple[Tensor, Tensor]:
        encs = self.encoder(states)
        encs_d = self.ste_thresh.apply(encs, 0.5)
        return encs, encs_d  # pyright: ignore[reportReturnType]


class Decoder(nn.Module):
    """Decoder from the 256-bit binary latent back to 32x32x6 observation."""

    def __init__(self, chan_in: int, chan_enc: int, enc_hw: tuple[int, int]) -> None:
        super().__init__()
        self.chan_enc = chan_enc
        self.enc_hw = enc_hw
        use_bias_with_norm = False
        self.decoder_conv = nn.Sequential(
            Conv2dModel(
                chan_enc,
                [32, 32],
                [2, 4],
                [0, 0],
                [True, False],
                ["RELU", "SIGMOID"],
                strides=[2, 4],
                group_norms=[0, 0],
                transpose=True,
                use_bias_with_norm=use_bias_with_norm,
            ),
            Conv2dModel(32, [chan_in], [1], [0], [False], ["LINEAR"], use_bias_with_norm=use_bias_with_norm),
        )

    def forward(self, encs: Tensor) -> Tensor:
        decs = encs.view(-1, self.chan_enc, self.enc_hw[0], self.enc_hw[1])
        return self.decoder_conv(decs)


class EnvModel(nn.Module):
    """Latent transition model for one primitive Powderworld action."""

    def __init__(
        self,
        chan_enc: int,
        enc_hw: tuple[int, int],
        resnet_chan: int,
        num_resnet_blocks: int,
        num_actions: int,
    ) -> None:
        super().__init__()
        self.chan_enc = chan_enc
        self.enc_hw = enc_hw
        self.num_actions = num_actions
        use_bias_with_norm = False
        self.mask_net = nn.Sequential(
            Conv2dModel(
                chan_enc + num_actions,
                [resnet_chan],
                [1],
                [0],
                [False],
                ["RELU"],
                group_norms=[1],
                use_bias_with_norm=use_bias_with_norm,
            ),
            ResnetConv2dModel(
                resnet_chan,
                resnet_chan,
                resnet_chan,
                3,
                1,
                num_resnet_blocks,
                True,
                "RELU",
                group_norm=0,
                use_bias_with_norm=use_bias_with_norm,
            ),
            Conv2dModel(
                resnet_chan,
                [chan_enc, chan_enc],
                [1, 1],
                [0, 0],
                [True, False],
                ["RELU", "SIGMOID"],
                group_norms=[0, 0],
                use_bias_with_norm=use_bias_with_norm,
            ),
            nn.Flatten(),
        )

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        states_conv = states.view(-1, self.chan_enc, self.enc_hw[0], self.enc_hw[1])
        actions_oh = F.one_hot(actions.long(), self.num_actions).float()
        actions_oh = actions_oh.view(-1, self.num_actions, 1, 1)
        actions_oh = actions_oh.repeat(1, 1, states_conv.shape[2], states_conv.shape[3])
        return self.mask_net(torch.cat((states_conv.float(), actions_oh), dim=1))


class EnvModelContinuous(nn.Module):
    """Observation-space transition model kept for API compatibility."""

    def __init__(
        self,
        chan_in: int,
        chan_enc: int,
        resnet_chan: int,
        num_resnet_blocks: int,
        num_actions: int,
    ) -> None:
        super().__init__()
        self.chan_enc = chan_enc
        self.num_actions = num_actions
        use_bias_with_norm = False
        self.encoder = nn.Sequential(
            Conv2dModel(
                chan_in,
                [32, chan_enc],
                [4, 2],
                [0, 0],
                [True, False],
                ["RELU", "SIGMOID"],
                strides=[4, 2],
                group_norms=[0, 0],
                use_bias_with_norm=use_bias_with_norm,
            )
        )
        self.env_model = nn.Sequential(
            Conv2dModel(
                chan_enc + num_actions,
                [resnet_chan],
                [1],
                [0],
                [False],
                ["RELU"],
                group_norms=[1],
                use_bias_with_norm=use_bias_with_norm,
            ),
            ResnetConv2dModel(
                resnet_chan,
                resnet_chan,
                resnet_chan,
                3,
                1,
                num_resnet_blocks,
                True,
                "RELU",
                group_norm=0,
                use_bias_with_norm=use_bias_with_norm,
            ),
            Conv2dModel(
                resnet_chan,
                [chan_enc, chan_enc],
                [1, 1],
                [0, 0],
                [True, False],
                ["RELU", "SIGMOID"],
                group_norms=[0, 0],
                use_bias_with_norm=use_bias_with_norm,
            ),
            Conv2dModel(
                chan_enc,
                [32, 32],
                [2, 4],
                [0, 0],
                [True, False],
                ["RELU", "SIGMOID"],
                strides=[2, 4],
                group_norms=[0, 0],
                transpose=True,
                use_bias_with_norm=use_bias_with_norm,
            ),
            Conv2dModel(32, [chan_in], [1], [0], [False], ["LINEAR"], use_bias_with_norm=use_bias_with_norm),
        )

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        states_conv = self.encoder(states)
        actions_oh = F.one_hot(actions.long(), self.num_actions).float()
        actions_oh = actions_oh.view(-1, self.num_actions, 1, 1)
        actions_oh = actions_oh.repeat(1, 1, states_conv.shape[2], states_conv.shape[3])
        return self.env_model(torch.cat((states_conv.float(), actions_oh), dim=1))


POWDERWORLD_DIFFICULTY_TO_ELEM_NAMES: dict[str, tuple[str, ...]] = {
    "easy": ("plant", "stone"),
    "medium": ("sand", "water", "fire", "plant", "stone"),
    "hard": ("sand", "water", "fire", "plant", "stone", "gas", "wood", "ice"),
}


class PowderworldEasyEnvironment(Environment):
    """DeepCubeAI wrapper for OGBench powderworld-easy with primitive actions."""

    world_size = 32
    chan_in = 6
    chan_enc = 16
    enc_hw = (4, 4)
    enc_dim = chan_enc * enc_hw[0] * enc_hw[1]
    _num_actions_max = 8
    difficulty = "easy"
    elem_names = POWDERWORLD_DIFFICULTY_TO_ELEM_NAMES[difficulty]

    def __init__(self) -> None:
        super().__init__()
        self.dtype = float32

    @property
    def env_name(self) -> str:
        return "powderworld_easy"

    @property
    def num_actions_max(self) -> int:
        return self._num_actions_max

    @classmethod
    def _gym_env_id(cls) -> str:
        return f"powderworld-{cls.difficulty}-v0"

    @classmethod
    def _make_gym_env(cls, seed: int | None = None) -> Any:
        try:
            import gymnasium
            import ogbench.powderworld  # noqa: F401, PLC0415
        except Exception as exc:  # pragma: no cover - only needed for rollout data generation.
            raise RuntimeError(
                "Powderworld gym env is unavailable. Add mqe-release to PYTHONPATH before "
                "calling generate_start_states/get_goals."
            ) from exc

        env = gymnasium.make(cls._gym_env_id(), mode="task", max_episode_steps=1001)
        if seed is not None:
            env.reset(seed=seed)
        return env

    def next_state(self, states: list[State], actions: list[int]) -> tuple[list[State], list[float]]:
        raise NotImplementedError("Use the learned EnvModel for Powderworld latent planning.")

    def rand_action(self, states: list[State]) -> list[int]:
        return list(np.random.randint(0, self.num_actions_max, size=len(states)))

    @staticmethod
    def is_solved(states: list[State], states_goal: list[State]) -> NDArray[np.bool_]:
        solved = []
        for state, goal in zip(states, states_goal, strict=False):
            if isinstance(state, PowderworldState) and isinstance(goal, PowderworldState):
                solved.append(np.array_equal(state.observation[..., :3], goal.observation[..., :3]))
            else:
                solved.append(False)
        return np.asarray(solved, dtype=np.bool_)

    def state_to_real(self, states: list[State]) -> NDArray[float32]:
        states_real = np.zeros((len(states), self.world_size, self.world_size, self.chan_in), dtype=float32)
        for idx, state in enumerate(states):
            if isinstance(state, PowderworldState):
                states_real[idx] = state.observation.astype(float32) / 255.0
        return states_real.transpose([0, 3, 1, 2])

    def get_dqn(self) -> nn.Module:
        resnet_chan = 7 * self.chan_enc * 2
        return PowderworldDQN(self.chan_enc, self.enc_hw, resnet_chan, 4, self.num_actions_max, True)

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
        return EnvModel(self.chan_enc, self.enc_hw, 64, 4, self.num_actions_max)

    def get_env_nnet_cont(self) -> nn.Module:
        return EnvModelContinuous(self.chan_in, self.chan_enc, 64, 4, self.num_actions_max)

    def get_encoder(self) -> nn.Module:
        return Encoder(self.chan_in, self.chan_enc)

    def get_decoder(self) -> nn.Module:
        return Decoder(self.chan_in, self.chan_enc, self.enc_hw)

    def generate_start_states(self, num_states: int, level_seeds: list[int] | None = None) -> list[State]:
        start_states: list[State] = []
        env = self._make_gym_env()
        for idx in range(num_states):
            seed = None if level_seeds is None else level_seeds[idx]
            ob, info = env.reset(seed=seed)
            unwrapped = env.unwrapped
            task_info = getattr(unwrapped, "cur_task_info", None)
            task_id = getattr(unwrapped, "cur_task_id", None)
            start_states.append(PowderworldState(ob, info.get("goal"), seed=seed, task_info=task_info, task_id=task_id))
        env.close()
        return start_states

    @staticmethod
    def get_goals(states: list[State], num_steps: int | None) -> list[State]:
        del num_steps
        goals: list[State] = []
        for state in states:
            if isinstance(state, PowderworldState) and state.goal_observation is not None:
                goals.append(
                    PowderworldState(
                        state.goal_observation,
                        seed=state.seed,
                        task_info=state.task_info,
                        task_id=state.task_id,
                    )
                )
            else:
                raise ValueError("PowderworldState does not contain a goal observation.")
        return goals


class PowderworldMediumEnvironment(PowderworldEasyEnvironment):
    """DeepCubeAI wrapper for OGBench powderworld-medium with primitive actions."""

    difficulty = "medium"
    elem_names = POWDERWORLD_DIFFICULTY_TO_ELEM_NAMES[difficulty]

    @property
    def env_name(self) -> str:
        return "powderworld_medium"


class PowderworldHardEnvironment(PowderworldEasyEnvironment):
    """DeepCubeAI wrapper for OGBench powderworld-hard with primitive actions."""

    difficulty = "hard"
    elem_names = POWDERWORLD_DIFFICULTY_TO_ELEM_NAMES[difficulty]

    @property
    def env_name(self) -> str:
        return "powderworld_hard"
