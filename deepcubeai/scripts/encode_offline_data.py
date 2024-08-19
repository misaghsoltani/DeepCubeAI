import os
import pickle
import time
from argparse import ArgumentParser
from typing import List, Tuple

import numpy as np
import torch

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
    parser.add_argument("--data_enc", type=str, required=True, help="File from which to save data")
    return parser


def load_data(data_path: str) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Loads data from a specified file.

    Args:
        data_path (str): Path to the data file.

    Returns:
        Tuple[List[np.ndarray], List[List[int]]]: Loaded state and action episodes.
    """
    with open(data_path, "rb") as data_file:
        state_episodes, action_episodes = pickle.load(data_file)
    return state_episodes, action_episodes


def load_models(env_dir: str, env: Environment,
                device: torch.device) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Loads the encoder and decoder models.

    Args:
        env_dir (str): Directory of the environment.
        env (Environment): The environment object.
        device (torch.device): The device to load the models onto.

    Returns:
        Tuple[torch.nn.Module, torch.nn.Module]: Loaded encoder and decoder models.
    """
    encoder = nnet_utils.load_nnet(f"{env_dir}/encoder_state_dict.pt",
                                   env.get_encoder(),
                                   device=device)
    encoder.to(device)
    encoder.eval()

    decoder = nnet_utils.load_nnet(f"{env_dir}/decoder_state_dict.pt",
                                   env.get_decoder(),
                                   device=device)
    decoder.to(device)
    decoder.eval()

    return encoder, decoder


def encode_episodes(state_episodes: List[np.ndarray], encoder: torch.nn.Module,
                    decoder: torch.nn.Module,
                    device: torch.device) -> Tuple[List[np.ndarray], List[float]]:
    """Encodes state episodes and calculates reconstruction errors.

    Args:
        state_episodes (List[np.ndarray]): List of state episodes.
        encoder (torch.nn.Module): Encoder model.
        decoder (torch.nn.Module): Decoder model.
        device (torch.device): Device to perform computations on.

    Returns:
        Tuple[List[np.ndarray], List[float]]: Encoded state episodes and reconstruction errors.
    """
    state_enc_episodes = []
    recon_errs = []
    display_steps = list(np.linspace(1, len(state_episodes), 10, dtype=int))
    start_time = time.time()

    for episode_num, state_episode in enumerate(state_episodes):
        # encode
        state_episode_tens = torch.tensor(state_episode, device=device).float().contiguous()
        _, state_episode_enc_tens = encoder(state_episode_tens)
        state_episode_enc = state_episode_enc_tens.cpu().data.numpy()

        for num in np.unique(state_episode_enc):
            assert num in [0, 1], "Encoding must be binary"

        state_episode_enc = state_episode_enc.reshape(
            (state_episode_enc.shape[0], -1)).astype(np.uint8)
        state_enc_episodes.append(state_episode_enc)

        # decode to check error
        state_episode_dec_tens = decoder(state_episode_enc_tens)
        errs = torch.flatten(torch.pow(state_episode_tens - state_episode_dec_tens, 2),
                             start_dim=1).mean(dim=1)
        recon_errs.extend(list(errs.cpu().data.numpy()))

        if episode_num in display_steps:
            print(f"{100 * episode_num / len(state_episodes):.2f}% "
                  f"(Total time: {time.time() - start_time:.2f})")

    return state_enc_episodes, recon_errs


def save_encoded_data(data_enc_path: str, state_enc_episodes: List[np.ndarray],
                      action_episodes: List[List[int]]) -> None:
    """Saves encoded data to a specified file.

    Args:
        data_enc_path (str): Path to save the encoded data.
        state_enc_episodes (List[np.ndarray]): Encoded state episodes.
        action_episodes (List[List[int]]): Action episodes.
    """
    data_enc_dir = os.path.dirname(data_enc_path)
    if not os.path.exists(data_enc_dir):
        os.makedirs(data_enc_dir)

    with open(data_enc_path, "wb") as data_enc_file:
        pickle.dump((state_enc_episodes, action_episodes), data_enc_file, protocol=-1)


def main():
    """Main function to execute the encoding process."""
    parser = parse_arguments()
    args = parser.parse_args()
    print_args(args)

    env = env_utils.get_environment(args.env)
    state_episodes, action_episodes = load_data(args.data)
    print(f"Episodes: {len(state_episodes)}")

    device, _, _ = nnet_utils.get_device()
    encoder, decoder = load_models(args.env_dir, env, device)

    state_enc_episodes, recon_errs = encode_episodes(state_episodes, encoder, decoder, device)

    print(f"Recon Errs Mean(Min/Max/Std): "
          f"{float(np.mean(recon_errs)):.2E}"
          f"({float(np.min(recon_errs)):.2E}/"
          f"{float(np.max(recon_errs)):.2E}/"
          f"{float(np.std(recon_errs)):.2E})")

    save_encoded_data(args.data_enc, state_enc_episodes, action_episodes)


if __name__ == "__main__":
    main()
