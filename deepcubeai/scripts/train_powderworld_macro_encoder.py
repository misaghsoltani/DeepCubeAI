from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from deepcubeai.environments.powderworld_easy import PowderworldState
from deepcubeai.environments.powderworld_easy_macro import LearnedMacroEncoder, obs_to_classes_for_elems
from deepcubeai.utils import env_utils


def _difficulty_env_key(difficulty: str, suffix: str) -> str:
    return f"powderworld_{difficulty}_{suffix}"


def _resolve_args(args: Any) -> Any:
    if args.env is None:
        args.env = _difficulty_env_key(args.difficulty, "macro")
    if args.env_model_dir is None:
        args.env_model_dir = f"deepcubeai/saved_env_models/powderworld_{args.difficulty}_macro_learned"
    if args.heur_dir is None:
        args.heur_dir = f"deepcubeai/saved_heur_models/powderworld_{args.difficulty}_macro_learned/current"
    return args


def _collect_obs_and_labels(num_pairs: int, seed: int, env_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Collect reset and goal frames, labeled by the oracle 8x8 class parser."""
    np.random.seed(seed)
    env = env_utils.get_environment(env_key)
    states = env.generate_start_states(num_pairs, level_seeds=list(range(seed, seed + num_pairs)))
    elem_names = tuple(getattr(env, "elem_names"))

    obs_l: list[np.ndarray] = []
    labels_l: list[np.ndarray] = []
    for state in states:
        if not isinstance(state, PowderworldState):
            raise TypeError(f"Expected PowderworldState, got {type(state)!r}")

        observations = [state.observation]
        if state.goal_observation is not None:
            observations.append(state.goal_observation)

        for observation in observations:
            obs_l.append(observation.astype(np.float32) / 255.0)
            labels_l.append(obs_to_classes_for_elems(observation, elem_names))

    obs = np.stack(obs_l, axis=0).transpose(0, 3, 1, 2).astype(np.float32)
    labels = np.stack(labels_l, axis=0).astype(np.int64)
    return obs, labels


@torch.inference_mode()
def _eval_encoder(
    encoder: LearnedMacroEncoder,
    obs: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    encoder.eval()
    total_cells = 0
    correct_cells = 0
    total_frames = 0
    correct_frames = 0
    loss_sum = 0.0

    for start in range(0, obs.shape[0], batch_size):
        end = min(start + batch_size, obs.shape[0])
        logits = encoder.logits(obs[start:end])
        batch_labels = labels[start:end]
        loss = F.cross_entropy(logits, batch_labels, reduction="sum")
        pred = torch.argmax(logits, dim=1)
        correct = pred.eq(batch_labels)

        loss_sum += float(loss.item())
        correct_cells += int(correct.sum().item())
        total_cells += int(correct.numel())
        correct_frames += int(correct.flatten(start_dim=1).all(dim=1).sum().item())
        total_frames += int(correct.shape[0])

    return {
        "loss": loss_sum / max(total_frames, 1),
        "cell_acc": correct_cells / max(total_cells, 1),
        "frame_acc": correct_frames / max(total_frames, 1),
    }


def _save_qstar_checkpoints(encoder: nn.Module, env_model_dir: Path, heur_dir: Path, env_key: str) -> None:
    oracle_env = env_utils.get_environment(env_key)
    env_model_dir.mkdir(parents=True, exist_ok=True)
    heur_dir.mkdir(parents=True, exist_ok=True)

    torch.save(encoder.state_dict(), env_model_dir / "encoder_state_dict.pt")
    torch.save(oracle_env.get_decoder().state_dict(), env_model_dir / "decoder_state_dict.pt")
    torch.save(oracle_env.get_env_nnet().state_dict(), env_model_dir / "env_state_dict.pt")
    torch.save(oracle_env.get_dqn().state_dict(), heur_dir / "model_state_dict.pt")


def train_encoder(args: Any) -> dict[str, Any]:
    args = _resolve_args(args)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    env = env_utils.get_environment(args.env)
    num_classes = int(getattr(env, "num_classes"))

    train_obs_np, train_labels_np = _collect_obs_and_labels(args.num_train_pairs, args.seed, args.env)
    val_obs_np, val_labels_np = _collect_obs_and_labels(args.num_val_pairs, args.val_seed, args.env)

    train_obs = torch.tensor(train_obs_np, device=device)
    train_labels = torch.tensor(train_labels_np, device=device)
    val_obs = torch.tensor(val_obs_np, device=device)
    val_labels = torch.tensor(val_labels_np, device=device)

    encoder = LearnedMacroEncoder(
        chan_in=train_obs.shape[1],
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
    ).to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    last_metrics: dict[str, float] = {}
    for itr in range(1, args.iters + 1):
        encoder.train()
        idxs = torch.randint(0, train_obs.shape[0], (args.batch_size,), device=device)
        logits = encoder.logits(train_obs[idxs])
        loss = F.cross_entropy(logits, train_labels[idxs])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if itr == 1 or itr % args.report_every == 0 or itr == args.iters:
            train_metrics = _eval_encoder(encoder, train_obs, train_labels, args.eval_batch_size)
            val_metrics = _eval_encoder(encoder, val_obs, val_labels, args.eval_batch_size)
            last_metrics = {
                "itr": float(itr),
                "train_loss": train_metrics["loss"],
                "train_cell_acc": train_metrics["cell_acc"],
                "train_frame_acc": train_metrics["frame_acc"],
                "val_loss": val_metrics["loss"],
                "val_cell_acc": val_metrics["cell_acc"],
                "val_frame_acc": val_metrics["frame_acc"],
            }
            print(
                f"itr={itr:05d} "
                f"loss={float(loss.item()):.6f} "
                f"train_cell={train_metrics['cell_acc'] * 100:.2f}% "
                f"train_frame={train_metrics['frame_acc'] * 100:.2f}% "
                f"val_cell={val_metrics['cell_acc'] * 100:.2f}% "
                f"val_frame={val_metrics['frame_acc'] * 100:.2f}%"
            )

    env_model_dir = Path(args.env_model_dir)
    heur_dir = Path(args.heur_dir)
    _save_qstar_checkpoints(encoder.cpu(), env_model_dir, heur_dir, args.env)

    summary: dict[str, Any] = {
        "num_train_images": int(train_obs_np.shape[0]),
        "num_val_images": int(val_obs_np.shape[0]),
        "num_train_pairs": int(args.num_train_pairs),
        "num_val_pairs": int(args.num_val_pairs),
        "env": args.env,
        "difficulty": args.difficulty,
        "num_classes": num_classes,
        "env_model_dir": str(env_model_dir),
        "heur_dir": str(heur_dir),
        "metrics": last_metrics,
        "args": vars(args),
    }
    metrics_path = env_model_dir / "encoder_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = ArgumentParser(description="Train a learned Powderworld macro symbolic-grid encoder.")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument(
        "--env",
        default=None,
        help="Macro environment key. Defaults to powderworld_<difficulty>_macro.",
    )
    parser.add_argument(
        "--env_model_dir",
        default=None,
        help="Output directory for Q* env checkpoints.",
    )
    parser.add_argument(
        "--heur_dir",
        default=None,
        help="Output directory containing model_state_dict.pt for Q* heuristic.",
    )
    parser.add_argument("--num_train_pairs", type=int, default=2000)
    parser.add_argument("--num_val_pairs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_seed", type=int, default=100000)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--report_every", type=int, default=100)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    args = parser.parse_args()
    train_encoder(args)


if __name__ == "__main__":
    main()
