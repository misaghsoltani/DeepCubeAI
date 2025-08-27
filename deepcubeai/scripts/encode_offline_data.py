from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import time
from typing import Any

import numpy as np
from numpy import float32, uint8
from numpy.typing import NDArray
import torch
from torch import nn

from deepcubeai.environments.environment_abstract import Environment
from deepcubeai.utils import env_utils, nnet_utils
from deepcubeai.utils.data_utils import print_args


def parse_arguments() -> ArgumentParser:
    """Parses command-line arguments.

    Returns:
        ArgumentParser: The argument parser with the defined arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Environment")
    parser.add_argument("--env_dir", type=str, required=True, help="Directory of environment")
    parser.add_argument("--data", type=str, required=True, help="File from which to load data")
    parser.add_argument("--data_enc", type=str, required=True, help="File to save data")
    return parser


@dataclass(frozen=True, slots=True)
class EncodeOfflineConfig:
    """Config for encode_offline_data entrypoint."""

    env: str
    env_dir: str
    data: str
    data_enc: str

    @staticmethod
    def from_json(path: str | Path) -> EncodeOfflineConfig:
        """Load config from JSON path."""
        raw: dict[str, str] = json.loads(Path(path).read_bytes())
        return EncodeOfflineConfig(**raw)


def load_data(data_path: str) -> tuple[list[NDArray[float32 | uint8]], list[list[int]]]:
    """Loads data from a specified file.

    Args:
        data_path (str): Path to the data file.

    Returns:
        tuple[list[NDArray], list[list[int]]]: Loaded state and action episodes.
    """
    with open(data_path, "rb") as data_file:
        state_episodes, action_episodes = pickle.load(data_file)
    return state_episodes, action_episodes


def load_models(env_dir: str, env: Environment, device: torch.device) -> tuple[nn.Module, nn.Module]:
    """Loads the encoder and decoder models.

    Args:
        env_dir (str): Directory of the environment.
        env (Environment): The environment object.
        device (torch.device): The device to load the models onto.

    Returns:
        tuple[nn.Module, nn.Module]: Loaded encoder and decoder models.
    """
    encoder: nn.Module = nnet_utils.load_nnet(f"{env_dir}/encoder_state_dict.pt", env.get_encoder(), device=device)
    encoder.to(device)
    encoder.eval()

    decoder: nn.Module = nnet_utils.load_nnet(f"{env_dir}/decoder_state_dict.pt", env.get_decoder(), device=device)
    decoder.to(device)
    decoder.eval()

    return encoder, decoder


@torch.inference_mode()
def encode_episodes(
    state_episodes: list[NDArray[float32 | uint8]], encoder: nn.Module, decoder: nn.Module, device: torch.device
) -> tuple[list[NDArray[Any]], list[float]]:
    """Encodes state episodes and calculates reconstruction errors.

    Args:
        state_episodes (list[NDArray]): List of state episodes.
        encoder (nn.Module): Encoder model.
        decoder (nn.Module): Decoder model.
        device (torch.device): Device to perform computations on.

    Returns:
        tuple[list[NDArray], list[float]]: Encoded state episodes and reconstruction errors.
    """
    state_enc_episodes: list[NDArray[Any]] = []
    recon_errs: list[float] = []
    display_steps: list[int] = list(np.linspace(1, len(state_episodes), 10, dtype=int))
    start_time: float = time.time()

    for episode_num, state_episode in enumerate(state_episodes):
        # encode
        state_episode_tens: torch.Tensor = torch.tensor(state_episode, device=device).float().contiguous()
        _, state_episode_enc_tens = encoder(state_episode_tens)
        state_episode_enc = state_episode_enc_tens.cpu().data.numpy()

        for num in np.unique(state_episode_enc):
            assert num in {0, 1}, "Encoding must be binary"

        state_episode_enc = state_episode_enc.reshape((state_episode_enc.shape[0], -1)).astype(uint8)
        state_enc_episodes.append(state_episode_enc)

        # decode to check error
        state_episode_dec_tens = decoder(state_episode_enc_tens)
        errs = torch.flatten(torch.pow(state_episode_tens - state_episode_dec_tens, 2), start_dim=1).mean(dim=1)
        recon_errs.extend(list(errs.cpu().data.numpy()))

        if episode_num in display_steps:
            print(f"{100 * episode_num / len(state_episodes):.2f}% (Total time: {time.time() - start_time:.2f})")

    return state_enc_episodes, recon_errs


def save_encoded_data(
    data_enc_path: str, state_enc_episodes: Sequence[NDArray[Any]], action_episodes: list[list[int]]
) -> None:
    """Saves encoded data to a specified file.

    Args:
        data_enc_path (str): Path to save the encoded data.
        state_enc_episodes (list[NDArray]): Encoded state episodes.
        action_episodes (list[list[int]]): Action episodes.
    """
    p = Path(data_enc_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "wb") as data_enc_file:
        pickle.dump((state_enc_episodes, action_episodes), data_enc_file, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(p)


@torch.inference_mode()
def run_encode_offline(cfg: EncodeOfflineConfig) -> None:
    """Entry point for programmatic use (no CLI stdout redirection)."""
    env: Environment = env_utils.get_environment(cfg.env)
    state_episodes, action_episodes = load_data(cfg.data)
    device, _, _ = nnet_utils.get_device()
    encoder, decoder = load_models(cfg.env_dir, env, device)
    state_enc_episodes, _ = encode_episodes(state_episodes, encoder, decoder, device)
    save_encoded_data(cfg.data_enc, state_enc_episodes, action_episodes)


@torch.inference_mode()
def main() -> None:
    """Main function to execute the encoding process."""
    parser = parse_arguments()
    args = parser.parse_args()
    print_args(args)
    cfg = EncodeOfflineConfig(env=args.env, env_dir=args.env_dir, data=args.data, data_enc=args.data_enc)
    env: Environment = env_utils.get_environment(cfg.env)
    state_episodes, action_episodes = load_data(cfg.data)
    print(f"Episodes: {len(state_episodes)}")
    device, _, _ = nnet_utils.get_device()
    encoder, decoder = load_models(cfg.env_dir, env, device)
    state_enc_episodes, recon_errs = encode_episodes(state_episodes, encoder, decoder, device)
    print(
        f"Recon Errs Mean(Min/Max/Std): "
        f"{float(np.mean(recon_errs)):.2E}"
        f"({float(np.min(recon_errs)):.2E}/"
        f"{float(np.max(recon_errs)):.2E}/"
        f"{float(np.std(recon_errs)):.2E})"
    )
    save_encoded_data(cfg.data_enc, state_enc_episodes, action_episodes)


if __name__ == "__main__":
    main()
