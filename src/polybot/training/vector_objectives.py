"""Short-horizon objective curriculum for the raw-policy league.

The regular flat actor-critic intentionally performs one inexpensive update per
native environment step.  It is excellent for throughput, but a one-step TD
target gives a weak signal for the common Polytopia sequence ``explore ->
found/capture -> convert the new city into a win``.  This module collects a
small, on-policy vector rollout before updating.  Generalised advantage
estimation then carries the visible objective reward across the whole short
sequence without inventing hidden state or custom map positions.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F
from tqdm.auto import tqdm

from ..config import ModelConfig, TrainingConfig
from ..infrastructure.device import select_device
from ..infrastructure.model_io import save_model
from ..infrastructure.randomness import seed_everything
from ..models.policy_value_belief import PolicyValueBeliefNet
from .run_artifacts import TrainingRun
from .vector_actor_critic import _normalise_advantage, _visible_action_reward
from .vector_teacher import _teacher_targets, _vector_tensors, _visible_goal_progress


def objective_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminals: torch.Tensor,
    player_signs: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """Return player-relative GAE for a ``[time, environment]`` rollout.

    PolyEnv reports a value in the perspective of the currently active
    player.  A turn switch therefore flips both the bootstrap value and the
    continuation advantage.  Terminal/auto-reset rows explicitly stop that
    continuation, so one finished game can never leak into a new slot.
    """
    expected = rewards.shape
    if (
        rewards.ndim != 2
        or values.shape != expected
        or next_values.shape != expected
        or terminals.shape != expected
        or player_signs.shape != expected
    ):
        raise ValueError("objective GAE inputs must share [time, environment] shape")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be between zero and one")

    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for index in range(rewards.shape[0] - 1, -1, -1):
        not_done = 1.0 - terminals[index]
        sign = player_signs[index]
        delta = (
            rewards[index]
            + float(gamma) * not_done * sign * next_values[index]
            - values[index]
        )
        carry = delta + float(gamma) * float(gae_lambda) * not_done * sign * carry
        advantages[index] = carry
    return advantages


def _forward(
    model: PolicyValueBeliefNet,
    tensors: Dict[str, torch.Tensor],
) -> Any:
    return model(
        tensors["map_tokens"],
        tensors["global_features"],
        tensors["action_categorical"],
        tensors["action_continuous"],
        tensors["action_mask"],
        tensors["event_features"],
        include_belief=False,
    )


def train_vector_objective_curriculum(
    training: TrainingConfig,
    model_config: ModelConfig,
    *,
    num_envs: int,
    steps: int,
    num_threads: int | None,
    max_actions: int,
    rollout_horizon: int = 8,
    updates_per_rollout: int = 2,
    objective_teacher_probability: float = 0.35,
    teacher_weight: float = 0.25,
    shaping_weight: float = 0.20,
    entropy_weight: float = 0.003,
    value_weight: float = 0.50,
    policy_anchor_weight: float = 0.75,
    value_anchor_weight: float = 0.50,
    gamma: float = 0.995,
    gae_lambda: float = 0.90,
    model_out: Optional[Path] = None,
    model: Optional[PolicyValueBeliefNet] = None,
    anchor_model: Optional[PolicyValueBeliefNet] = None,
    run: TrainingRun | None = None,
) -> Dict[str, float | str]:
    """Train on short actual-game rollouts before unrestricted self-play.

    The environment is never seeded with an invented position.  Every target
    comes from a legal action and the player-visible state returned by
    ``VectorGameEnv``.  A small teacher fraction makes the early rollouts
    reach villages/cities; policy-gradient loss is applied only to actions
    sampled from the candidate itself.
    """
    if num_envs < 1 or steps < 1 or max_actions < 1:
        raise ValueError("num_envs, steps and max_actions must be positive")
    if rollout_horizon < 2 or updates_per_rollout < 1:
        raise ValueError("rollout_horizon must be at least two and updates positive")
    if training.map_type != "lakes":
        raise ValueError("VectorGameEnv objective curriculum currently supports lakes only")
    for name, value in {
        "objective_teacher_probability": objective_teacher_probability,
        "teacher_weight": teacher_weight,
        "shaping_weight": shaping_weight,
        "entropy_weight": entropy_weight,
        "value_weight": value_weight,
        "policy_anchor_weight": policy_anchor_weight,
        "value_anchor_weight": value_anchor_weight,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")
    try:
        from PolyEnv import Bardur, Imperius, VectorGameEnv
    except ImportError as error:
        raise RuntimeError("PolyEnv VectorGameEnv is required") from error

    seed_everything(training.seed)
    device = select_device(training.device)
    run = run or TrainingRun.create("runs", "vector_objectives")
    run.write_config(
        training=training,
        model=model_config,
        stage={
            "num_envs": num_envs,
            "steps": steps,
            "num_threads": num_threads,
            "max_actions": max_actions,
            "rollout_horizon": rollout_horizon,
            "updates_per_rollout": updates_per_rollout,
            "objective_teacher_probability": objective_teacher_probability,
            "teacher_weight": teacher_weight,
            "shaping_weight": shaping_weight,
            "entropy_weight": entropy_weight,
            "value_weight": value_weight,
            "policy_anchor_weight": policy_anchor_weight,
            "value_anchor_weight": value_anchor_weight,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
        },
    )
    model = model.to(device) if model is not None else PolicyValueBeliefNet(model_config).to(device)
    if anchor_model is model:
        raise ValueError("anchor_model must be a frozen copy")
    if anchor_model is not None:
        anchor_model = anchor_model.to(device).eval()
        for parameter in anchor_model.parameters():
            parameter.requires_grad_(False)

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    cuda_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=cuda_amp)
    env = VectorGameEnv(
        num_envs=num_envs,
        seed=training.seed,
        map_size=training.map_size,
        players=(Bardur, Imperius),
        num_threads=num_threads,
        max_actions=max_actions,
        auto_reset=True,
        visible_event_history=0,
    )
    batch = env.reset(seed=training.seed)
    rng = np.random.default_rng(int(training.seed) ^ 0x6F424A)
    totals = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "teacher_loss": 0.0,
        "policy_anchor_loss": 0.0,
        "value_anchor_loss": 0.0,
        "entropy": 0.0,
        "reward": 0.0,
        "advantage": 0.0,
    }
    updates = 0
    terminals = 0
    learner_actions = 0
    teacher_actions = 0
    village_entries = 0
    city_captures = 0
    city_founds = 0
    completed_steps = 0

    with tqdm(
        total=steps,
        desc=f"objective rollouts ({device})",
        unit="step",
        dynamic_ncols=True,
    ) as progress:
        while completed_steps < steps:
            local_steps = min(rollout_horizon, steps - completed_steps)
            transitions: list[Dict[str, Any]] = []
            rewards: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            next_values: list[torch.Tensor] = []
            terminal_rows: list[torch.Tensor] = []
            player_signs: list[torch.Tensor] = []

            for _ in range(local_steps):
                tensors = _vector_tensors(batch, device, training)
                features = np.asarray(batch["action_features"], dtype=np.int64)
                action_mask = np.asarray(batch["action_mask"], dtype=bool)
                tokens = np.asarray(batch["map_tokens"], dtype=np.int64)
                state = np.asarray(batch["state"], dtype=np.int64)
                target = torch.as_tensor(
                    _teacher_targets(
                        features, action_mask, tokens, state, training.map_size
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                players_before = state[:, 1].copy()
                model.eval()
                with torch.no_grad():
                    output = _forward(model, tensors)
                    sampled_rows = Categorical(logits=output.policy_logits).sample()
                teacher_rows = target.argmax(dim=1)
                use_teacher = rng.random(num_envs) < objective_teacher_probability
                use_teacher_tensor = torch.as_tensor(
                    use_teacher, dtype=torch.bool, device=device
                )
                action_rows_tensor = torch.where(
                    use_teacher_tensor, teacher_rows, sampled_rows
                )
                action_rows = action_rows_tensor.cpu().numpy().astype(np.int64)
                shaped = _visible_action_reward(
                    features, action_rows, tokens, state, training.map_size
                )
                _, enters_village = _visible_goal_progress(
                    features, tokens, state, training.map_size
                )
                chosen = features[np.arange(num_envs), action_rows]
                action_ids = np.asarray(batch["action_id"], dtype=np.int32)[
                    np.arange(num_envs), action_rows
                ]
                try:
                    next_batch = env.step(action_ids)
                except RuntimeError:
                    if model_out is not None:
                        save_model(model_out, model, model_config)
                    raise
                terminal = np.asarray(next_batch["terminated"], dtype=bool)
                terminal_reward = np.asarray(next_batch["reward"], dtype=np.float32)
                reward = shaping_weight * shaped + terminal_reward
                next_state = np.asarray(next_batch["state"], dtype=np.int64)
                sign = np.where(next_state[:, 1] == players_before, 1.0, -1.0)
                model.eval()
                with torch.no_grad():
                    next_value = _forward(
                        model, _vector_tensors(next_batch, device, training)
                    ).value

                transitions.append(
                    {
                        "tensors": tensors,
                        "target": target,
                        "actions": action_rows_tensor,
                        "on_policy": ~use_teacher_tensor,
                    }
                )
                rewards.append(torch.as_tensor(reward, dtype=torch.float32, device=device))
                values.append(output.value.detach())
                next_values.append(next_value.detach())
                terminal_rows.append(torch.as_tensor(terminal, dtype=torch.float32, device=device))
                player_signs.append(torch.as_tensor(sign, dtype=torch.float32, device=device))
                terminals += int(terminal.sum())
                learner_actions += int((~use_teacher).sum())
                teacher_actions += int(use_teacher.sum())
                village_entries += int(enters_village[np.arange(num_envs), action_rows].sum())
                city_captures += int(
                    ((chosen[:, 0] == 8) & (chosen[:, 8] == 14)).sum()
                )
                city_founds += int(
                    ((chosen[:, 0] == 8) & (chosen[:, 8] == 11)).sum()
                )
                batch = next_batch

            reward_tensor = torch.stack(rewards)
            value_tensor = torch.stack(values)
            next_value_tensor = torch.stack(next_values)
            terminal_tensor = torch.stack(terminal_rows)
            sign_tensor = torch.stack(player_signs)
            advantages = objective_gae(
                reward_tensor,
                value_tensor,
                next_value_tensor,
                terminal_tensor,
                sign_tensor,
                gamma=gamma,
                gae_lambda=gae_lambda,
            ).detach()
            targets = (value_tensor + advantages).detach()
            policy_advantages = advantages[
                torch.stack([item["on_policy"] for item in transitions])
            ]
            normalised_policy_advantages = _normalise_advantage(policy_advantages)
            advantage_offset = 0

            for _ in range(updates_per_rollout):
                model.train()
                optimiser.zero_grad(set_to_none=True)
                autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if cuda_amp
                    else nullcontext()
                )
                policy_losses: list[torch.Tensor] = []
                value_losses: list[torch.Tensor] = []
                teacher_losses: list[torch.Tensor] = []
                policy_anchor_losses: list[torch.Tensor] = []
                value_anchor_losses: list[torch.Tensor] = []
                entropies: list[torch.Tensor] = []
                with autocast:
                    for index, transition in enumerate(transitions):
                        output = _forward(model, transition["tensors"])
                        distribution = Categorical(logits=output.policy_logits)
                        on_policy = transition["on_policy"]
                        on_policy_count = int(on_policy.sum().item())
                        if on_policy_count:
                            step_advantage = normalised_policy_advantages[
                                advantage_offset:advantage_offset + on_policy_count
                            ]
                            actions = transition["actions"][on_policy]
                            policy_losses.append(
                                -(
                                    distribution.log_prob(actions)[on_policy]
                                    * step_advantage.detach()
                                ).mean()
                            )
                            entropies.append(distribution.entropy()[on_policy].mean())
                        else:
                            policy_losses.append(output.value.sum() * 0.0)
                            entropies.append(output.value.sum() * 0.0)
                        advantage_offset += on_policy_count
                        value_losses.append(
                            F.smooth_l1_loss(output.value, targets[index])
                        )
                        log_probabilities = F.log_softmax(output.policy_logits, dim=-1)
                        teacher_losses.append(
                            -(
                                transition["target"]
                                * torch.where(
                                    transition["tensors"]["action_mask"],
                                    log_probabilities,
                                    torch.zeros_like(log_probabilities),
                                )
                            ).sum(dim=1).mean()
                        )
                        if anchor_model is not None:
                            with torch.no_grad():
                                anchor = _forward(anchor_model, transition["tensors"])
                                anchor_policy = torch.softmax(anchor.policy_logits, dim=-1)
                            policy_anchor_losses.append(
                                -(
                                    anchor_policy
                                    * torch.where(
                                        transition["tensors"]["action_mask"],
                                        log_probabilities,
                                        torch.zeros_like(log_probabilities),
                                    )
                                ).sum(dim=1).mean()
                            )
                            value_anchor_losses.append(
                                F.smooth_l1_loss(output.value, anchor.value)
                            )
                        else:
                            policy_anchor_losses.append(output.value.sum() * 0.0)
                            value_anchor_losses.append(output.value.sum() * 0.0)

                    policy_loss = torch.stack(policy_losses).mean()
                    value_loss = torch.stack(value_losses).mean()
                    teacher_loss = torch.stack(teacher_losses).mean()
                    policy_anchor_loss = torch.stack(policy_anchor_losses).mean()
                    value_anchor_loss = torch.stack(value_anchor_losses).mean()
                    entropy = torch.stack(entropies).mean()
                    loss = (
                        policy_loss
                        + value_weight * value_loss
                        + teacher_weight * teacher_loss
                        + policy_anchor_weight * policy_anchor_loss
                        + value_anchor_weight * value_anchor_loss
                        - entropy_weight * entropy
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimiser)
                scaler.update()
                updates += 1
                totals["loss"] += float(loss.detach().float().cpu())
                totals["policy_loss"] += float(policy_loss.detach().float().cpu())
                totals["value_loss"] += float(value_loss.detach().float().cpu())
                totals["teacher_loss"] += float(teacher_loss.detach().float().cpu())
                totals["policy_anchor_loss"] += float(policy_anchor_loss.detach().float().cpu())
                totals["value_anchor_loss"] += float(value_anchor_loss.detach().float().cpu())
                totals["entropy"] += float(entropy.detach().float().cpu())
                totals["reward"] += float(reward_tensor.mean().detach().cpu())
                totals["advantage"] += float(advantages.abs().mean().detach().cpu())
                advantage_offset = 0

            completed_steps += local_steps
            progress.update(local_steps)
            run.log(
                step=completed_steps,
                update=updates,
                environment_step=completed_steps * num_envs,
                objective_terminals=terminals,
                objective_learner_actions=learner_actions,
                objective_teacher_actions=teacher_actions,
                objective_village_entries=village_entries,
                objective_city_captures=city_captures,
                objective_city_founds=city_founds,
                objective_mean_reward=totals["reward"] / max(1, updates),
                objective_mean_abs_advantage=totals["advantage"] / max(1, updates),
            )
            if completed_steps % max(rollout_horizon * 4, 1) == 0 or completed_steps == steps:
                progress.set_postfix(
                    env_steps=completed_steps * num_envs,
                    reward=f"{totals['reward'] / max(1, updates):.3f}",
                    terminals=terminals,
                    villages=village_entries,
                    captures=city_captures,
                )

    if model_out is not None:
        save_model(model_out, model, model_config)
    summary: Dict[str, float | str] = {
        "objective_envs": float(num_envs),
        "objective_steps": float(completed_steps),
        "objective_environment_steps": float(completed_steps * num_envs),
        "objective_rollout_horizon": float(rollout_horizon),
        "objective_updates_per_rollout": float(updates_per_rollout),
        "objective_optimisation_updates": float(updates),
        "objective_loss": totals["loss"] / max(1, updates),
        "objective_policy_loss": totals["policy_loss"] / max(1, updates),
        "objective_value_loss": totals["value_loss"] / max(1, updates),
        "objective_teacher_loss": totals["teacher_loss"] / max(1, updates),
        "objective_policy_anchor_loss": totals["policy_anchor_loss"] / max(1, updates),
        "objective_value_anchor_loss": totals["value_anchor_loss"] / max(1, updates),
        "objective_entropy": totals["entropy"] / max(1, updates),
        "objective_mean_reward": totals["reward"] / max(1, updates),
        "objective_mean_abs_advantage": totals["advantage"] / max(1, updates),
        "objective_terminal_episodes": float(terminals),
        "objective_learner_actions": float(learner_actions),
        "objective_teacher_actions": float(teacher_actions),
        "objective_village_entries": float(village_entries),
        "objective_city_captures": float(city_captures),
        "objective_city_founds": float(city_founds),
        "objective_teacher_probability": float(objective_teacher_probability),
        "objective_teacher_weight": float(teacher_weight),
        "objective_shaping_weight": float(shaping_weight),
        "objective_gae_lambda": float(gae_lambda),
    }
    summary["run_dir"] = str(run.finish(summary))
    return summary
