from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from deepcubeai.environments.powderworld_easy_macro import (
    LearnedMacroEnvModel,
    LearnedMacroHeuristic,
    LearnedMacroEncoder,
)
from deepcubeai.utils import env_utils


GRID_SIZE = 8


def _difficulty_env_key(difficulty: str, suffix: str) -> str:
    return f"powderworld_{difficulty}_{suffix}"


def _resolve_args(args: Any) -> Any:
    if args.env is None:
        args.env = _difficulty_env_key(args.difficulty, "macro_neural")
    if args.encoder_checkpoint is None:
        args.encoder_checkpoint = (
            f"deepcubeai/saved_env_models/powderworld_{args.difficulty}_macro_learned/encoder_state_dict.pt"
        )
    if args.env_model_dir is None:
        args.env_model_dir = f"deepcubeai/saved_env_models/powderworld_{args.difficulty}_macro_neural"
    if args.heur_dir is None:
        args.heur_dir = f"deepcubeai/saved_heur_models/powderworld_{args.difficulty}_macro_neural/current"
    return args


def _sample_classes(batch_size: int, num_classes: int, empty_prob: float, device: torch.device) -> Tensor:
    probs = torch.full((num_classes,), (1.0 - empty_prob) / max(num_classes - 1, 1), device=device)
    probs[0] = empty_prob
    flat = torch.multinomial(probs, batch_size * GRID_SIZE * GRID_SIZE, replacement=True)
    return flat.view(batch_size, GRID_SIZE, GRID_SIZE)


def _one_hot(classes: Tensor, num_classes: int) -> Tensor:
    return F.one_hot(classes.long(), num_classes).permute(0, 3, 1, 2).float()


def _apply_actions(classes: Tensor, actions: Tensor) -> Tensor:
    next_classes = classes.clone()
    elem = actions.long() // 64
    rem = actions.long() % 64
    x = rem // 8
    y = rem % 8
    next_classes[torch.arange(classes.shape[0], device=classes.device), y, x] = elem + 1
    return next_classes


def _heuristic_targets(classes: Tensor, goals: Tensor) -> Tensor:
    num_draw_elems = int(goals.max().item()) if goals.numel() > 0 else 0
    num_draw_elems = max(num_draw_elems, 1)
    before = classes.ne(goals).float()
    total_before = before.sum(dim=(1, 2), keepdim=True)

    targets = []
    for class_id in range(1, num_draw_elems + 1):
        after = goals.ne(class_id).float()
        costs = total_before - before + after
        targets.append(costs.permute(0, 2, 1).flatten(start_dim=1))
    return torch.cat(targets, dim=1) * 1.01


def _heuristic_targets_with_num_elems(classes: Tensor, goals: Tensor, num_draw_elems: int) -> Tensor:
    before = classes.ne(goals).float()
    total_before = before.sum(dim=(1, 2), keepdim=True)
    targets = []
    for class_id in range(1, num_draw_elems + 1):
        after = goals.ne(class_id).float()
        costs = total_before - before + after
        targets.append(costs.permute(0, 2, 1).flatten(start_dim=1))
    return torch.cat(targets, dim=1) * 1.01


def _heuristic_classifier_targets(classes: Tensor, goals: Tensor, num_draw_elems: int) -> tuple[Tensor, Tensor]:
    before = classes.ne(goals).float()
    after = torch.stack([goals.ne(class_id).float() for class_id in range(1, num_draw_elems + 1)], dim=1)
    return before, after


@torch.inference_mode()
def _eval_transition(
    model: LearnedMacroEnvModel,
    batch_size: int,
    batches: int,
    num_classes: int,
    num_actions: int,
    empty_prob: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_cells = 0
    correct_cells = 0
    total_frames = 0
    correct_frames = 0
    round_correct_frames = 0
    valid_round_cells = 0

    for _ in range(batches):
        classes = _sample_classes(batch_size, num_classes, empty_prob, device)
        actions = torch.randint(0, num_actions, (batch_size,), device=device)
        targets = _apply_actions(classes, actions)
        target_one_hot = _one_hot(targets, num_classes)
        out = model(_one_hot(classes, num_classes).flatten(start_dim=1), actions).view(
            batch_size, num_classes, GRID_SIZE, GRID_SIZE
        )
        pred = torch.argmax(out, dim=1)
        correct = pred.eq(targets)
        rounded = out.round()

        total_cells += int(correct.numel())
        correct_cells += int(correct.sum().item())
        total_frames += batch_size
        correct_frames += int(correct.flatten(start_dim=1).all(dim=1).sum().item())
        round_correct_frames += int(rounded.eq(target_one_hot).flatten(start_dim=1).all(dim=1).sum().item())
        valid_round_cells += int(rounded.sum(dim=1).eq(1).sum().item())

    return {
        "cell_acc": correct_cells / max(total_cells, 1),
        "frame_acc": correct_frames / max(total_frames, 1),
        "round_frame_acc": round_correct_frames / max(total_frames, 1),
        "round_valid_cell_rate": valid_round_cells / max(total_cells, 1),
    }


@torch.inference_mode()
def _eval_heuristic(
    model: LearnedMacroHeuristic,
    batch_size: int,
    batches: int,
    num_classes: int,
    num_draw_elems: int,
    empty_prob: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total = 0
    mae_sum = 0.0
    argmin_hits = 0
    before_correct = 0
    after_correct = 0
    before_total = 0
    after_total = 0

    for _ in range(batches):
        classes = _sample_classes(batch_size, num_classes, empty_prob, device)
        goals = _sample_classes(batch_size, num_classes, empty_prob, device)
        states = _one_hot(classes, num_classes).flatten(start_dim=1)
        goal_states = _one_hot(goals, num_classes).flatten(start_dim=1)
        targets = _heuristic_targets_with_num_elems(classes, goals, num_draw_elems)
        pred = model(states, goal_states)
        before_targets, after_targets = _heuristic_classifier_targets(classes, goals, num_draw_elems)
        before_logits, after_logits = model.mismatch_logits(states, goal_states)

        mae_sum += float(torch.abs(pred - targets).sum().item())
        total += int(targets.numel())
        optimal = targets.eq(targets.min(dim=1, keepdim=True).values)
        pred_argmin = torch.argmin(pred, dim=1, keepdim=True)
        argmin_hits += int(torch.gather(optimal, 1, pred_argmin).sum().item())

        before_pred = torch.sigmoid(before_logits).ge(0.5)
        after_pred = torch.sigmoid(after_logits).ge(0.5)
        before_correct += int(before_pred.eq(before_targets.bool()).sum().item())
        after_correct += int(after_pred.eq(after_targets.bool()).sum().item())
        before_total += int(before_targets.numel())
        after_total += int(after_targets.numel())

    return {
        "mae": mae_sum / max(total, 1),
        "argmin_hit": argmin_hits / max(batch_size * batches, 1),
        "before_acc": before_correct / max(before_total, 1),
        "after_acc": after_correct / max(after_total, 1),
    }


def _save_qstar_checkpoints(
    encoder_checkpoint: Path,
    env_model: nn.Module,
    heuristic: nn.Module,
    env_model_dir: Path,
    heur_dir: Path,
    env_key: str,
) -> None:
    env = env_utils.get_environment(env_key)
    env_model_dir.mkdir(parents=True, exist_ok=True)
    heur_dir.mkdir(parents=True, exist_ok=True)

    encoder_state = torch.load(encoder_checkpoint, map_location="cpu")
    encoder = LearnedMacroEncoder(num_classes=int(getattr(env, "num_classes")))
    encoder.load_state_dict(encoder_state)

    torch.save(encoder.state_dict(), env_model_dir / "encoder_state_dict.pt")
    torch.save(env.get_decoder().state_dict(), env_model_dir / "decoder_state_dict.pt")
    torch.save(env_model.cpu().state_dict(), env_model_dir / "env_state_dict.pt")
    torch.save(heuristic.cpu().state_dict(), heur_dir / "model_state_dict.pt")


def train_models(args: Any) -> dict[str, Any]:
    args = _resolve_args(args)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    env = env_utils.get_environment(args.env)
    num_classes = int(getattr(env, "num_classes"))
    num_actions = int(env.num_actions_max)
    num_draw_elems = num_actions // (GRID_SIZE * GRID_SIZE)

    env_model = LearnedMacroEnvModel(
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_actions=num_actions,
    ).to(device)
    heuristic = LearnedMacroHeuristic(
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_actions=num_actions,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(env_model.parameters()) + list(heuristic.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    last_metrics: dict[str, float] = {}
    for itr in range(1, args.iters + 1):
        env_model.train()
        heuristic.train()

        classes = _sample_classes(args.batch_size, num_classes, args.empty_prob, device)
        actions = torch.randint(0, num_actions, (args.batch_size,), device=device)
        next_classes = _apply_actions(classes, actions)
        trans_logits = env_model.logits(_one_hot(classes, num_classes).flatten(start_dim=1), actions)
        trans_loss = F.cross_entropy(trans_logits, next_classes)

        heur_classes = _sample_classes(args.batch_size, num_classes, args.empty_prob, device)
        goals = _sample_classes(args.batch_size, num_classes, args.empty_prob, device)
        states = _one_hot(heur_classes, num_classes).flatten(start_dim=1)
        goal_states = _one_hot(goals, num_classes).flatten(start_dim=1)
        before_targets, after_targets = _heuristic_classifier_targets(heur_classes, goals, num_draw_elems)
        before_logits, after_logits = heuristic.mismatch_logits(states, goal_states)
        heur_loss = F.binary_cross_entropy_with_logits(before_logits, before_targets)
        heur_loss = heur_loss + F.binary_cross_entropy_with_logits(after_logits, after_targets)

        loss = trans_loss + heur_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if itr == 1 or itr % args.report_every == 0 or itr == args.iters:
            trans_metrics = _eval_transition(
                env_model,
                args.eval_batch_size,
                args.eval_batches,
                num_classes,
                num_actions,
                args.empty_prob,
                device,
            )
            heur_metrics = _eval_heuristic(
                heuristic,
                args.eval_batch_size,
                args.eval_batches,
                num_classes,
                num_draw_elems,
                args.empty_prob,
                device,
            )
            last_metrics = {
                "itr": float(itr),
                "loss": float(loss.item()),
                "transition_loss": float(trans_loss.item()),
                "heuristic_loss": float(heur_loss.item()),
                **{f"transition_{key}": val for key, val in trans_metrics.items()},
                **{f"heuristic_{key}": val for key, val in heur_metrics.items()},
            }
            print(
                f"itr={itr:05d} "
                f"loss={float(loss.item()):.6f} "
                f"trans_frame={trans_metrics['frame_acc'] * 100:.2f}% "
                f"trans_round={trans_metrics['round_frame_acc'] * 100:.2f}% "
                f"heur_mae={heur_metrics['mae']:.4f} "
                f"heur_argmin={heur_metrics['argmin_hit'] * 100:.2f}%"
            )

    env_model_dir = Path(args.env_model_dir)
    heur_dir = Path(args.heur_dir)
    _save_qstar_checkpoints(Path(args.encoder_checkpoint), env_model, heuristic, env_model_dir, heur_dir, args.env)

    summary: dict[str, Any] = {
        "env_model_dir": str(env_model_dir),
        "heur_dir": str(heur_dir),
        "encoder_checkpoint": str(args.encoder_checkpoint),
        "env": args.env,
        "difficulty": args.difficulty,
        "num_classes": num_classes,
        "num_actions": num_actions,
        "metrics": last_metrics,
        "args": vars(args),
    }
    metrics_path = env_model_dir / "macro_model_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = ArgumentParser(description="Train neural Powderworld macro transition and heuristic models.")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument(
        "--env",
        default=None,
        help="Neural macro environment key. Defaults to powderworld_<difficulty>_macro_neural.",
    )
    parser.add_argument(
        "--encoder_checkpoint",
        default=None,
        help="Learned macro encoder checkpoint to copy into the Q* env model directory.",
    )
    parser.add_argument(
        "--env_model_dir",
        default=None,
        help="Output directory for encoder/decoder/env checkpoints.",
    )
    parser.add_argument(
        "--heur_dir",
        default=None,
        help="Output directory containing model_state_dict.pt for Q* heuristic.",
    )
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--eval_batch_size", type=int, default=4096)
    parser.add_argument("--eval_batches", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--empty_prob", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report_every", type=int, default=100)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    args = parser.parse_args()
    train_models(args)


if __name__ == "__main__":
    main()
