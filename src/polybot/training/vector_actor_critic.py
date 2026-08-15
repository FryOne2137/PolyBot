"""High-throughput, on-policy actor-critic curriculum without MCTS."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm

from ..config import ModelConfig, TrainingConfig
from ..infrastructure.device import select_device
from ..infrastructure.model_io import save_model
from ..models.policy_value_belief import PolicyValueBeliefNet
from .vector_teacher import _teacher_targets, _vector_tensors


def _normalise_advantage(advantage: torch.Tensor) -> torch.Tensor:
    """Standardise a rollout batch without producing NaN for one env."""
    if advantage.numel() < 2:
        return advantage
    return (advantage - advantage.mean()) / advantage.std(
        unbiased=False
    ).clamp_min(1e-4)


def _visible_action_reward(features: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Dense, visible-only curriculum reward for early flat self-play.

    This is not an engine reward and never reads the full map. It merely makes
    sparse terminal reward useful before the value/policy have learned basic
    expansion, combat and city-capture behaviour.
    """
    chosen = features[np.arange(features.shape[0]), rows]
    action_type, tile_action = chosen[:, 0], chosen[:, 8]
    damage, affordable = chosen[:, 15], chosen[:, 14]
    reward = np.zeros(features.shape[0], dtype=np.float32)
    reward += np.where((action_type == 8) & (tile_action == 14), 0.30, 0.0)  # capture city
    reward += np.where((action_type == 8) & (tile_action == 11), 0.16, 0.0)  # found city
    reward += np.where(action_type == 1, 0.035 + np.clip(damage, 0, 10) * 0.006, 0.0)
    reward += np.where((action_type == 8) & np.isin(tile_action, (10, 12)), 0.04, 0.0)
    reward += np.where(np.isin(action_type, (5, 6)), 0.02, 0.0)
    reward += np.where(np.isin(action_type, (4, 7)), 0.01, 0.0)
    reward -= np.where(action_type == 3, 0.025, 0.0)  # premature end-turn
    reward -= np.where(affordable == 0, 0.03, 0.0)
    return reward


def train_vector_actor_critic(
    training: TrainingConfig,
    model_config: ModelConfig,
    num_envs: int,
    steps: int,
    num_threads: Optional[int] = None,
    max_actions: int = 1024,
    entropy_weight: float = 0.008,
    value_weight: float = 0.5,
    teacher_weight: float = 0.25,
    gamma: float = 0.995,
    model_out: Optional[Path] = None,
    model: Optional[PolicyValueBeliefNet] = None,
) -> Dict[str, float]:
    """Train policy/value directly from native vector rollouts in RAM only."""
    if num_envs < 1 or steps < 1 or max_actions < 1:
        raise ValueError("num_envs, steps and max_actions must be positive")
    if training.map_type != "lakes":
        raise ValueError("the documented VectorGameEnv constructor currently exposes the default lakes ruleset only")
    try:
        from PolyEnv import Bardur, Imperius, VectorGameEnv
    except ImportError as error:
        raise RuntimeError("Update PolyEnv to a version exposing VectorGameEnv.") from error

    device = select_device(training.device)
    model = model.to(device) if model is not None else PolicyValueBeliefNet(model_config).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    cuda_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=cuda_amp)
    env = VectorGameEnv(
        num_envs=num_envs, seed=training.seed, map_size=training.map_size,
        players=(Bardur, Imperius), num_threads=num_threads, max_actions=max_actions,
        auto_reset=True, visible_event_history=0,
    )
    batch = env.reset(seed=training.seed)
    totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "teacher_loss": 0.0, "entropy": 0.0, "reward": 0.0}
    terminals = 0

    with tqdm(total=steps, desc=f"flat actor-critic ({device})", unit="step", dynamic_ncols=True) as progress:
        for step in range(steps):
            tensors = _vector_tensors(batch, device, training)
            action_features = np.asarray(batch["action_features"], dtype=np.int64)
            teacher_target = torch.as_tensor(
                _teacher_targets(action_features, np.asarray(batch["action_mask"], dtype=bool)),
                dtype=torch.float32, device=device,
            )
            player_before = np.asarray(batch["state"], dtype=np.int32)[:, 1]
            model.train()
            autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if cuda_amp else nullcontext()
            with autocast:
                output = model(
                    tensors["map_tokens"], tensors["global_features"],
                    tensors["action_categorical"], tensors["action_continuous"],
                    tensors["action_mask"], tensors["event_features"], include_belief=False,
                )
                distribution = Categorical(logits=output.policy_logits)
                actions = distribution.sample()
                log_probability = distribution.log_prob(actions)
                entropy = distribution.entropy().mean()
            action_rows = actions.detach().cpu().numpy().astype(np.int64)
            shaped = _visible_action_reward(action_features, action_rows)
            action_ids = np.asarray(batch["action_id"], dtype=np.int32)[np.arange(num_envs), action_rows]
            try:
                next_batch = env.step(action_ids)
            except RuntimeError:
                # Keep the user's "models only" storage rule while avoiding
                # losing a long RAM-only run to an underestimated action
                # capacity. The partial checkpoint is still a normal model.
                if model_out is not None:
                    save_model(model_out, model, model_config)
                raise
            terminal = np.asarray(next_batch["terminated"], dtype=bool)
            terminal_reward = np.asarray(next_batch["reward"], dtype=np.float32)
            reward = shaped + terminal_reward
            terminals += int(terminal.sum())

            # Bootstrap from the next player-visible state. A turn switch
            # negates the zero-sum value; auto-reset terminal rows never boot.
            with torch.no_grad():
                next_tensors = _vector_tensors(next_batch, device, training)
                next_output = model(
                    next_tensors["map_tokens"], next_tensors["global_features"],
                    next_tensors["action_categorical"], next_tensors["action_continuous"],
                    next_tensors["action_mask"], next_tensors["event_features"], include_belief=False,
                )
                player_after = np.asarray(next_batch["state"], dtype=np.int32)[:, 1]
                sign = torch.as_tensor(np.where(player_after == player_before, 1.0, -1.0), dtype=torch.float32, device=device)
                done = torch.as_tensor(terminal, dtype=torch.float32, device=device)
                reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=device)
                td_target = reward_tensor + gamma * (1.0 - done) * sign * next_output.value

            advantage = td_target - output.value
            normalised_advantage = _normalise_advantage(advantage)
            policy_loss = -(log_probability * normalised_advantage.detach()).mean()
            critic_loss = F.smooth_l1_loss(output.value, td_target)
            log_probabilities = F.log_softmax(output.policy_logits, dim=-1)
            teacher_loss = -(teacher_target * torch.where(tensors["action_mask"], log_probabilities, 0.0)).sum(dim=1).mean()
            # The supervised anchor is deliberately small: it keeps early
            # exploration from destroying a useful teacher policy, while TD
            # gradients are still free to improve it.
            loss = policy_loss + value_weight * critic_loss + teacher_weight * teacher_loss - entropy_weight * entropy
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()
            batch = next_batch

            totals["loss"] += float(loss.detach().cpu())
            totals["policy_loss"] += float(policy_loss.detach().cpu())
            totals["value_loss"] += float(critic_loss.detach().cpu())
            totals["teacher_loss"] += float(teacher_loss.detach().cpu())
            totals["entropy"] += float(entropy.detach().cpu())
            totals["reward"] += float(reward.mean())
            progress.update(1)
            if (step + 1) % 20 == 0 or step == 0:
                progress.set_postfix(
                    env_steps=(step + 1) * num_envs,
                    reward=f"{reward.mean():.3f}", terminals=terminals,
                )

    if model_out is not None:
        save_model(model_out, model, model_config)
    return {
        "flat_envs": float(num_envs), "flat_steps": float(steps),
        "environment_steps": float(num_envs * steps), "terminal_episodes": float(terminals),
        **{"flat_" + key: value / steps for key, value in totals.items()},
    }
