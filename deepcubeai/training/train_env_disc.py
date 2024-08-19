import os
import pickle
import sys
import time
from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Dict, List, OrderedDict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.optim.optimizer import Optimizer

from deepcubeai.environments.environment_abstract import Environment
from deepcubeai.utils import data_utils, env_utils, nnet_utils
from deepcubeai.utils.data_utils import print_args


def parse_arguments(parser: ArgumentParser) -> Dict[str, Any]:
    """Parses command-line arguments.

    Args:
        parser (ArgumentParser): The argument parser instance.

    Returns:
        Dict[str, Any]: A dictionary of parsed arguments.
    """
    # Environment
    parser.add_argument('--env', type=str, required=True, help="Environment")

    # Data
    parser.add_argument('--train_data', type=str, required=True, help="Location of training data")
    parser.add_argument('--val_data', type=str, required=True, help="Location of validation data")

    # Debug
    parser.add_argument('--debug', action='store_true', default=False, help="")

    # Gradient Descent
    parser.add_argument('--env_coeff', type=float, default=0.5, help="")

    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate")
    parser.add_argument('--lr_d',
                        type=float,
                        default=0.9999993,
                        help="Learning rate decay for every iteration. "
                        "Learning rate is decayed according to: "
                        "lr * (lr_d ^ itr)")

    # Training
    parser.add_argument('--max_itrs',
                        type=int,
                        default=1000000,
                        help="Maximum number of iterations")
    parser.add_argument('--batch_size', type=int, default=1000, help="Batch size")

    parser.add_argument('--path_len_incr_itr',
                        type=int,
                        default=3000,
                        help="Increment path length every x itrs")
    parser.add_argument('--num_steps',
                        type=int,
                        default=100,
                        help="Maximum number of steps to predict")

    parser.add_argument('--only_env', action='store_true', default=False, help="")

    # model
    parser.add_argument('--nnet_name', type=str, required=True, help="Name of neural network")
    parser.add_argument('--save_dir',
                        type=str,
                        default="saved_env_models",
                        help="Directory to which to save model")

    # parse arguments
    args = parser.parse_args()
    args_dict: Dict[str, Any] = vars(args)

    assert (0.0 <= args_dict['env_coeff'] <= 1.0)

    # make save directory
    train_dir: str = f"{args_dict['save_dir']}/{args_dict['nnet_name']}/"
    args_dict["train_dir"] = train_dir
    args_dict["nnet_model_dir"] = f"{args_dict['train_dir']}/"
    best_model_dir = f"{args_dict['nnet_model_dir']}/best_model"
    if not os.path.exists(args_dict["nnet_model_dir"]):
        os.makedirs(args_dict["nnet_model_dir"])

    if not os.path.exists(best_model_dir) and False:
        os.makedirs(best_model_dir)

    if not os.path.exists(f"{args_dict['nnet_model_dir']}/pics"):
        os.makedirs(f"{args_dict['nnet_model_dir']}/pics")

    args_dict["output_save_loc"] = f"{train_dir}/output.txt"

    # save args
    args_save_loc = f"{train_dir}/args.pkl"
    print(f"Saving arguments to {args_save_loc}")
    with open(args_save_loc, "wb") as f:
        pickle.dump(args, f, protocol=-1)

    print(f"Batch size: {args_dict['batch_size']}")

    return args_dict


def load_nnet(nnet_dir: str, env: Environment) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """Loads the neural network models.

    Args:
        nnet_dir (str): Directory of the neural network.
        env (Environment): The environment instance.

    Returns:
        Tuple[nn.Module, nn.Module, nn.Module]: Encoder, decoder, and environment models.
    """
    env_file: str = f"{nnet_dir}/env_state_dict.pt"
    encoder_file: str = f"{nnet_dir}/encoder_state_dict.pt"
    decoder_file: str = f"{nnet_dir}/decoder_state_dict.pt"

    if os.path.isfile(env_file):
        env_model = nnet_utils.load_nnet(env_file, env.get_env_nnet())
        encoder = nnet_utils.load_nnet(encoder_file, env.get_encoder())
        decoder = nnet_utils.load_nnet(decoder_file, env.get_decoder())

    else:
        env_model: nn.Module = env.get_env_nnet()
        encoder = env.get_encoder()
        decoder = env.get_decoder()

    return encoder, decoder, env_model


def load_best_model_info(nnet_model_dir: str) -> Dict[str, Any]:
    """Loads the best model information.

    Args:
        nnet_model_dir (str): Directory of the neural network model.

    Returns:
        Dict[str, Any]: Best model information.
    """
    best_model_info_file: str = f"{nnet_model_dir}/best_model/model_info.pkl"
    best_model_info = None
    if os.path.isfile(best_model_info_file):
        with open(best_model_info_file, "rb") as f:
            best_model_info: Dict[str, Any] = pickle.load(f)

    return best_model_info


def load_train_state(
        nnet_model_dir: str,
        env: Environment) -> Tuple[nn.Module, nn.Module, nn.Module, int, Dict[str, Any]]:
    """Loads the training state.

    Args:
        nnet_model_dir (str): Directory of the neural network model.
        env (Environment): The environment instance.

    Returns:
        Tuple[nn.Module, nn.Module, nn.Module, int, Dict[str, Any]]: Encoder, decoder, environment
            model, iteration, and best model information.
    """
    best_model_itr_file: str = f"{nnet_model_dir}/best_model/train_itr.pkl"
    best_model_info_file: str = f"{nnet_model_dir}/best_model/model_info.pkl"
    itr_file: str = f"{nnet_model_dir}/train_itr.pkl"
    best_model_info = None
    if os.path.isfile(best_model_itr_file) and False:
        with open(best_model_itr_file, "rb") as f:
            itr: int = pickle.load(f)
        nnet_model_dir: str = f"{nnet_model_dir}/best_model"
        with open(best_model_info_file, "rb") as f:
            best_model_info: Dict[str, Any] = pickle.load(f)

    elif os.path.isfile(itr_file):
        with open(itr_file, "rb") as f:
            itr: int = pickle.load(f) + 1

    else:
        itr: int = 0

    encoder, decoder, env_model = load_nnet(nnet_model_dir, env)

    return encoder, decoder, env_model, itr, best_model_info


class Autoencoder(nn.Module):
    """Autoencoder class for encoding and decoding states."""

    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        """Initializes the Autoencoder.

        Args:
            encoder (nn.Module): Encoder model.
            decoder (nn.Module): Decoder model.
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass of the Autoencoder.

        Args:
            states (Tensor): Input states.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Encoded states, encoded states with dropout, and
                reconstructed states.
        """
        enc, enc_d = self.encoder(states)
        recon = self.decoder(enc_d)
        return enc, enc_d, recon


def test_model(encoder: nn.Module, env_model: nn.Module, state_episodes_all: List[np.ndarray],
               action_episodes_all: List[List[int]], device: torch.device,
               batch_size: int) -> None:
    """Tests the model.

    Args:
        encoder (nn.Module): Encoder model.
        env_model (nn.Module): Environment model.
        state_episodes_all (List[np.ndarray]): List of state episodes.
        action_episodes_all (List[List[int]]): List of action episodes.
        device (torch.device): Device to run the model on.
        batch_size (int): Batch size.
    """
    encoder.eval()
    env_model.eval()

    num_steps: int = max([len(x) for x in action_episodes_all])

    episode_lens: np.array = np.array(
        [state_episode.shape[0] for state_episode in state_episodes_all])
    state_episodes, action_episodes, start_idxs = get_batch(state_episodes_all,
                                                            action_episodes_all, episode_lens,
                                                            batch_size, num_steps)

    # get initial current state
    states_np: np.ndarray = np.stack(
        [state_episode[idx] for state_episode, idx in zip(state_episodes, start_idxs)], axis=0)
    states = torch.tensor(states_np, device=device).float().contiguous()
    _, encs = encoder(states)

    all_match_count: int = 0
    eq_bit_min_all: float = np.inf
    for step in range(num_steps):
        # get action
        actions_np: np.array = np.array([
            action_episode[idx + step] for action_episode, idx in zip(action_episodes, start_idxs)
        ])
        actions = torch.tensor(actions_np, device=device).float()

        # predict next state
        encs_next_pred = env_model(encs, actions)
        encs_next_pred = torch.round(encs_next_pred)

        # get ground truth next state
        states_next_np: np.ndarray = np.stack([
            state_episode[idx + step + 1]
            for state_episode, idx in zip(state_episodes, start_idxs)
        ],
                                              axis=0)
        states_next = torch.tensor(states_next_np, device=device).float().contiguous()
        _, encs_next = encoder(states_next)

        # get percent equal
        eq = 100 * torch.all(torch.eq(encs_next_pred, encs_next),
                             dim=1).float().mean().cpu().data.numpy()
        eq_bit = 100 * torch.eq(encs_next_pred, encs_next).float().mean().cpu().data.numpy()
        eq_bit_min = 100 * torch.eq(encs_next_pred,
                                    encs_next).float().mean(dim=1).min().cpu().data.numpy()

        # print results
        all_match: bool = eq == 100.0
        print(f"step: {step}, eq_bit: {eq_bit:.2f}%, eq_bit_min: {eq_bit_min:.2f}%, "
              f"eq: {eq:.2f}%, all: {int(all_match)}")

        encs = torch.round(encs_next_pred.detach())
        all_match_count += int(all_match)
        eq_bit_min_all = min(eq_bit_min_all, eq_bit_min)

    print(f"eq_bit_min (all): {eq_bit_min_all:.2f}%")
    print(f"{all_match_count} out of {num_steps} have all match")


def step_model(
        autoencoder: Autoencoder, env_nnet: nn.Module, state_episodes: List[np.ndarray],
        action_episodes: List[List[int]], start_idxs: np.array, device: torch.device,
        env_coeff: float
) -> Tuple[Tensor, Tensor, Tensor, float, float, float, float, Tensor, Tensor]:
    """Performs a single step of the model.

    Args:
        autoencoder (Autoencoder): Autoencoder model.
        env_nnet (nn.Module): Environment neural network.
        state_episodes (List[np.ndarray]): List of state episodes.
        action_episodes (List[List[int]]): List of action episodes.
        start_idxs (np.array): Start indices.
        device (torch.device): Device to run the model on.
        env_coeff (float): Environment coefficient.

    Returns:
        Tuple[Tensor, Tensor, Tensor, float, float, float, float, Tensor, Tensor]: Losses and
            other metrics.
    """
    # get initial states
    states_np: np.ndarray = np.stack(
        [state_episode[idx] for state_episode, idx in zip(state_episodes, start_idxs)], axis=0)
    states = torch.tensor(states_np, device=device).float().contiguous()

    # get actions
    actions_np: np.array = np.array(
        [action_episode[idx] for action_episode, idx in zip(action_episodes, start_idxs)])
    actions = torch.tensor(actions_np, device=device).float()

    # get ground truth next states
    states_next_gt_np: np.ndarray = np.stack(
        [state_episode[idx + 1] for state_episode, idx in zip(state_episodes, start_idxs)], axis=0)
    states_next = torch.tensor(states_next_gt_np, device=device).float().contiguous()

    # encode and decode states
    _, states_enc_d, states_dec = autoencoder(states)
    _, states_next_enc_d, states_next_dec = autoencoder(states_next)

    decoder = None
    if not autoencoder.training:
        decoder = autoencoder.decoder

    # loss reconstruction
    loss_recon = torch.mean(torch.pow(states - states_dec, 2)) / 2.0
    loss_recon = loss_recon + torch.mean(torch.pow(states_next - states_next_dec, 2)) / 2.0

    percent_on = 100 * (torch.mean(states_enc_d) + torch.mean(states_next_enc_d)) / 2.0

    # predict next state
    states_enc_d = states_enc_d.reshape((states_enc_d.shape[0], -1))

    states_next_enc_pred = env_nnet(states_enc_d, actions)

    states_next_dec_pred = None
    if decoder is not None:
        states_next_dec_pred = decoder(torch.round(states_next_enc_pred))

    states_next_enc_d = states_next_enc_d.reshape((states_next_enc_d.shape[0], -1))

    loss_env = torch.mean(
        torch.pow(states_next_enc_d - torch.round(states_next_enc_pred.detach()), 2)) * 0.5
    loss_env = loss_env + torch.mean(
        torch.pow(states_next_enc_d.detach() - states_next_enc_pred, 2)) * 0.5

    eq = 100 * torch.all(torch.eq(torch.round(states_next_enc_pred),
                                  torch.round(states_next_enc_d)),
                         dim=1).float().mean()
    eq_bit = 100 * torch.eq(torch.round(states_next_enc_pred),
                            torch.round(states_next_enc_d)).float().mean()
    eq_bit_min = 100 * torch.eq(torch.round(states_next_enc_pred),
                                torch.round(states_next_enc_d)).float().mean(dim=1).min()

    loss = (1 - env_coeff) * loss_recon + env_coeff * loss_env

    return (loss, loss_recon, loss_env, percent_on, eq, eq_bit, eq_bit_min, states_dec,
            states_next_dec_pred)


def get_batch(state_episodes: List[np.ndarray], action_episodes: List[List[int]],
              episode_lens: np.array, batch_size: int,
              num_steps: int) -> Tuple[List[np.ndarray], List[List[int]], np.array]:
    """Gets a batch of state and action episodes.

    Args:
        state_episodes (List[np.ndarray]): List of state episodes.
        action_episodes (List[List[int]]): List of action episodes.
        episode_lens (np.array): Array of episode lengths.
        batch_size (int): Batch size.
        num_steps (int): Number of steps.

    Returns:
        Tuple[List[np.ndarray], List[List[int]], np.array]: Batch of state episodes, action
            episodes, and start indices.
    """
    episode_idxs: np.array = np.random.randint(len(state_episodes), size=batch_size)
    start_idxs: np.array = np.random.uniform(
        0, 1, size=batch_size) * (episode_lens[episode_idxs] - num_steps - 1)
    start_idxs = start_idxs.round().astype(int)

    state_episodes_batch: List[np.ndarray] = [state_episodes[idx] for idx in episode_idxs]
    action_episodes_batch: List[List[int]] = [action_episodes[idx] for idx in episode_idxs]

    return state_episodes_batch, action_episodes_batch, start_idxs


def train_nnet(autoencoder: Autoencoder, env_nnet: nn.Module, nnet_dir: str,
               state_episodes_train: List[np.ndarray], action_episodes_train: List[List[int]],
               state_episodes_val: List[np.ndarray], action_episodes_val: List[List[int]],
               device: torch.device, batch_size: int, num_itrs: int, start_itr: int, lr: float,
               lr_d: float, env_coeff: float, only_env: bool, best_model_info: Dict[str,
                                                                                    Any]) -> None:
    """Trains the neural network.

    Args:
        autoencoder (Autoencoder): Autoencoder model.
        env_nnet (nn.Module): Environment neural network.
        nnet_dir (str): Directory to save the neural network.
        state_episodes_train (List[np.ndarray]): List of training state episodes.
        action_episodes_train (List[List[int]]): List of training action episodes.
        state_episodes_val (List[np.ndarray]): List of validation state episodes.
        action_episodes_val (List[List[int]]): List of validation action episodes.
        device (torch.device): Device to run the model on.
        batch_size (int): Batch size.
        num_itrs (int): Number of iterations.
        start_itr (int): Starting iteration.
        lr (float): Learning rate.
        lr_d (float): Learning rate decay.
        env_coeff (float): Environment coefficient.
        only_env (bool): Whether to train only the environment model.
        best_model_info (Dict[str, Any]): Information about the best model.
    """
    # initialize
    if best_model_info is None:
        best_model_info = {
            'l_env': float('inf'),
            'l_env_val': float('inf'),
            'l_recon': float('inf'),
            'l_recon_val': float('inf'),
            'loss': float('inf'),
            'loss_val': float('inf'),
            'itr': -1,
            'max_itr': -1,
            'lr': -1,
            'env_coeff': -1
        }
    best_model_updated: bool = False
    best_env_net_state_dict: OrderedDict[str, Tensor]
    best_encoder_state_dict: OrderedDict[str, Tensor]
    best_decoder_state_dict: OrderedDict[str, Tensor]
    env_nnet.train()
    autoencoder.train()
    episode_lens_train: np.array = np.array(
        [state_episode.shape[0] for state_episode in state_episodes_train])
    episode_lens_val: np.array = np.array(
        [state_episode.shape[0] for state_episode in state_episodes_val])

    # optimization
    display_itrs: int = 100
    if only_env:
        optimizer: Optimizer = optim.Adam(list(env_nnet.parameters()), lr=lr)
    else:
        optimizer: Optimizer = optim.Adam(list(autoencoder.parameters()) +
                                          list(env_nnet.parameters()),
                                          lr=lr)

    # initialize status tracking
    start_time_all = time.time()

    num_steps: int = 1
    for train_itr in range(start_itr, num_itrs):
        batch_size_eff: int = int(np.ceil(batch_size / num_steps))

        # zero the parameter gradients
        optimizer.zero_grad()
        # if lr <= 0.0000001:
        #     lr_d = 0.99999993
        lr_itr: float = lr * (lr_d**train_itr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_itr

        # do steps
        state_episodes_batch, action_episodes_batch, start_idxs = get_batch(
            state_episodes_train, action_episodes_train, episode_lens_train, batch_size_eff,
            num_steps)

        loss, loss_recon, loss_env, per_on, eq, eq_bit, eq_bit_min, _, _ = step_model(
            autoencoder, env_nnet, state_episodes_batch, action_episodes_batch, start_idxs, device,
            env_coeff)

        # backwards
        loss.backward()

        # step
        optimizer.step()

        # display progress
        if train_itr % display_itrs == 0 or train_itr == num_itrs - 1:
            print("-------")
            env_nnet.eval()
            autoencoder.eval()

            print(
                f"Train\nloss: {loss.item():.2E}, l_recon: {loss_recon.item():.2E}, "
                f"l_env: {loss_env.item():.2E}, %%on: {per_on.item():.2f}, eq_bit: {eq_bit:.2f}, "
                f"eq_bit_min: {eq_bit_min:.2f}, eq: {eq:.2f}")

            # validation

            state_episodes_batch, action_episodes_batch, start_idxs = get_batch(
                state_episodes_val, action_episodes_val, episode_lens_val, batch_size_eff,
                num_steps)

            (loss_val, loss_recon_val, loss_env_val, per_on_val, eq_val, eq_bit_val,
             eq_bit_min_val, states_dec,
             states_next_dec_pred) = step_model(autoencoder, env_nnet, state_episodes_batch,
                                                action_episodes_batch, start_idxs, device,
                                                env_coeff)
            print(f"Validation\nloss: {loss_val.item():.2E}, "
                  f"l_recon: {loss_recon_val.item():.2E}, "
                  f"l_env: {loss_env_val.item():.2E}, "
                  f"%%on: {per_on_val.item():.2f}, "
                  f"eq_bit: {eq_bit_val:.2f}, "
                  f"eq_bit_min: {eq_bit_min_val:.2f}, "
                  f"eq: {eq_val:.2f}")

            print(f"Itr: {train_itr}, lr: {lr_itr:.2E}, env_coeff: {env_coeff}, "
                  f"times - all: {time.time() - start_time_all:.2f}")

            if train_itr % 10000 == 0 or train_itr == num_itrs - 1 or train_itr in [
                    100, 300, 600, 1000, 5000
            ]:
                state_gt = np.transpose(state_episodes_batch[0][start_idxs[0]], [1, 2, 0])
                next_state_gt = np.transpose(state_episodes_batch[0][start_idxs[0] + 1], [1, 2, 0])

                state_nnet = states_dec[0].cpu().data.numpy()
                state_nnet = np.transpose(state_nnet, [1, 2, 0])

                next_state_nnet = states_next_dec_pred[0].cpu().data.numpy()
                next_state_nnet = np.transpose(next_state_nnet, [1, 2, 0])

                state_gt = np.clip(state_gt, 0, 1)
                state_nnet = np.clip(state_nnet, 0, 1)

                next_state_gt = np.clip(next_state_gt, 0, 1)
                next_state_nnet = np.clip(next_state_nnet, 0, 1)

                fig = plt.figure(dpi=200)

                if state_nnet.shape[2] == 6:
                    ax1 = fig.add_subplot(241)
                    ax2 = fig.add_subplot(242)
                    ax3 = fig.add_subplot(243)
                    ax4 = fig.add_subplot(244)

                    ax5 = fig.add_subplot(245)
                    ax6 = fig.add_subplot(246)
                    ax7 = fig.add_subplot(247)
                    ax8 = fig.add_subplot(248)

                    ax1.imshow(state_gt[:, :, :3])
                    ax1.set_title("gt")
                    ax2.imshow(state_gt[:, :, 3:])
                    ax2.set_title("gt")

                    ax3.imshow(state_nnet[:, :, :3])
                    ax3.set_title("nnet")
                    ax4.imshow(state_nnet[:, :, 3:])
                    ax4.set_title("nnet")

                    ax5.imshow(next_state_gt[:, :, :3])
                    ax5.set_title("gt next")
                    ax6.imshow(next_state_gt[:, :, 3:])
                    ax6.set_title("gt next")

                    ax7.imshow(next_state_nnet[:, :, :3])
                    ax7.set_title("nnet next")
                    ax8.imshow(next_state_nnet[:, :, 3:])
                    ax8.set_title("nnet next")

                else:
                    ax1 = fig.add_subplot(221)
                    ax2 = fig.add_subplot(222)
                    ax3 = fig.add_subplot(223)
                    ax4 = fig.add_subplot(224)

                    ax1.imshow(state_gt)
                    ax1.set_title("gt")
                    ax2.imshow(state_nnet)
                    ax2.set_title("nnet")
                    ax3.imshow(next_state_gt)
                    ax3.set_title("gt next")
                    ax4.imshow(next_state_nnet)
                    ax4.set_title("nnet next")

                fig.tight_layout()

                plt.savefig(f"{nnet_dir}/pics/recon_itr{train_itr}.jpg")
                plt.title("Reconstruction")
                plt.close()

            if True:
                # Access the original modules if wrapped with DataParallel
                env_nnet_module = env_nnet.module if isinstance(env_nnet,
                                                                nn.DataParallel) else env_nnet
                encoder_module = autoencoder.module.encoder if isinstance(
                    autoencoder, nn.DataParallel) else autoencoder.encoder
                decoder_module = autoencoder.module.decoder if isinstance(
                    autoencoder, nn.DataParallel) else autoencoder.decoder

                torch.save(env_nnet_module.state_dict(), f"{nnet_dir}/env_state_dict.pt")
                torch.save(encoder_module.state_dict(), f"{nnet_dir}/encoder_state_dict.pt")
                torch.save(decoder_module.state_dict(), f"{nnet_dir}/decoder_state_dict.pt")
                with open(f"{nnet_dir}/train_itr.pkl", "wb") as f:
                    pickle.dump(train_itr, f, protocol=-1)

            if env_coeff == 0.5 and ((best_model_info['loss_val'] + best_model_info['loss']) / 2
                                     > (loss_val.item() + loss.item()) / 2) and False:
                #  and ((best_model_info['l_env_val'] >= loss_env_val.item()) or (best_model_info['l_recon_val'] >= loss_recon_val.item()))
                #  and ((best_model_info['l_env'] >= loss_env.item()) or (best_model_info['l_recon'] >= loss_recon.item()))):
                # if env_coeff == 0.5 and (best_model_info['l_env_val'] >= loss_env_val.item()
                #                          and best_model_info['l_recon_val'] >= loss_recon_val.item()):
                best_model_info['l_env'] = loss_env.item()
                best_model_info['l_recon'] = loss_recon.item()
                best_model_info['loss'] = loss.item()
                best_model_info['l_env_val'] = loss_env_val.item()
                best_model_info['l_recon_val'] = loss_recon_val.item()
                best_model_info['loss_val'] = loss_val.item()
                best_model_info['itr'] = train_itr
                best_model_info['max_itr'] = num_itrs
                best_model_info['lr'] = lr
                best_model_info['env_coeff'] = env_coeff
                best_env_net_state_dict = deepcopy(env_nnet_module.state_dict())
                best_encoder_state_dict = deepcopy(encoder_module.state_dict())
                best_decoder_state_dict = deepcopy(decoder_module.state_dict())
                best_model_updated = True

            print("")

            env_nnet.train()
            autoencoder.train()
            start_time_all = time.time()

    if best_model_updated:
        torch.save(best_env_net_state_dict, f"{nnet_dir}/best_model/env_state_dict.pt")
        torch.save(best_encoder_state_dict, f"{nnet_dir}/best_model/encoder_state_dict.pt")
        torch.save(best_decoder_state_dict, f"{nnet_dir}/best_model/decoder_state_dict.pt")
        with open(f"{nnet_dir}/best_model/train_itr.pkl", "wb") as f:
            pickle.dump(best_model_info['itr'], f, protocol=-1)
        with open(f"{nnet_dir}/best_model/model_info.pkl", "wb") as f:
            pickle.dump(best_model_info, f, protocol=-1)
        with open(f"{nnet_dir}/best_model/model_info.text", 'a') as file:
            file.write(f"{best_model_info}\n")


def main():
    """Main function to run the training process."""
    # arguments
    parser: ArgumentParser = ArgumentParser()
    args_dict: Dict[str, Any] = parse_arguments(parser)

    if not args_dict["debug"]:
        sys.stdout = data_utils.Logger(args_dict["output_save_loc"], "a")

    print_args(args_dict)

    # environment
    env: Environment = env_utils.get_environment(args_dict['env'])

    # get device
    device: torch.device
    devices: List[torch.device]
    on_gpu: bool
    device, devices, on_gpu = nnet_utils.get_device()
    print(f"device: {device}, devices: {devices}, on_gpu: {on_gpu}")

    # load nnet
    env_nnet: nn.Module
    start_itr: int
    encoder: nn.Module
    decoder: nn.Module
    best_model_info: Dict[str, Any]
    encoder, decoder, env_nnet, start_itr, best_model_info = load_train_state(
        args_dict['nnet_model_dir'], env)
    env_nnet.to(device)
    autoencoder: Autoencoder = Autoencoder(encoder, decoder)
    autoencoder.to(device)

    if on_gpu and len(devices) > 1:
        env_nnet = nn.DataParallel(env_nnet)
        autoencoder = nn.DataParallel(autoencoder)

    print(f"Using {len(devices)} GPU(s): {devices}")

    if best_model_info is not None:
        itrs_num = 20000
        args_dict['max_itrs'] = best_model_info['itr'] + itrs_num

    best_model_info = load_best_model_info(args_dict['nnet_model_dir'])

    print(f"Starting iteration: {start_itr}, Max iteration: {args_dict['max_itrs']}")

    if args_dict['max_itrs'] <= start_itr:
        print("Starting iteration >= Max iteration. Skipping training for these iterations.")
        return

    # load data
    start_time = time.time()
    print("Loading data ...")
    state_episodes_train: List[np.ndarray]
    action_episodes_train: List[List[int]]
    state_episodes_val: List[np.ndarray]
    action_episodes_val: List[List[int]]

    with open(args_dict['train_data'], "rb") as f:
        state_episodes_train, action_episodes_train = pickle.load(f)

    with open(args_dict['val_data'], "rb") as f:
        state_episodes_val, action_episodes_val = pickle.load(f)

    print(f"{len(state_episodes_train)} train episodes, {len(state_episodes_val)} val episodes")
    print(f"Data load time: {time.time() - start_time}")

    # test
    print("Testing before training")
    test_model(encoder, env_nnet, state_episodes_val, action_episodes_val, device,
               args_dict['batch_size'])

    # train nnet
    train_nnet(autoencoder, env_nnet, args_dict['nnet_model_dir'], state_episodes_train,
               action_episodes_train, state_episodes_val, action_episodes_val, device,
               args_dict['batch_size'], args_dict['max_itrs'], start_itr, args_dict['lr'],
               args_dict['lr_d'], args_dict['env_coeff'], args_dict['only_env'], best_model_info)

    print("Testing after training")
    test_model(encoder, env_nnet, state_episodes_val, action_episodes_val, device,
               args_dict['batch_size'])

    print("Done\n")


if __name__ == "__main__":
    main()
