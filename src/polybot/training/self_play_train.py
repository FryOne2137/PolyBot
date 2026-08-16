"""Concurrent, RAM-only AlphaZero-style self-play after belief warm-up."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from ..config import MCTSConfig, ModelConfig, TrainingConfig
from ..data.polyenv_features import stack_examples
from ..infrastructure.device import select_device
from ..infrastructure.model_io import save_model
from ..infrastructure.progress import ConsoleProgress
from ..models.policy_value_belief import PolicyValueBeliefNet
from ..planning.batched_inference import BatchedInference, NativeLeafInference
from ..planning.belief_mcts import BeliefMCTS
from .optimisation import train_batch
from .replay_buffer import ReplayBuffer
from .run_artifacts import TrainingRun
from .self_play import SelfPlayRunner
from .teacher_bootstrap import _teacher_game


def train_self_play(
    training: TrainingConfig,
    model_config: ModelConfig,
    mcts_config: MCTSConfig,
    games: int,
    updates_per_game: int,
    opponent: str = "self_play",
    random_opponent_rate: float = 0.5,
    mixed_opponent: str = "heuristic",
    teacher_replay_fraction: float = 0.25,
    workers: int = 1,
    global_inference_batch_size: int = 256,
    inference_wait_ms: float = 1.0,
    model_out: Optional[Path] = None,
    model: Optional[PolicyValueBeliefNet] = None,
    opponent_models: Sequence[PolicyValueBeliefNet] = (),
    opponent_model_weights: Sequence[float] = (),
    game_callback: Callable[[Mapping[str, Any]], None] | None = None,
    run: TrainingRun | None = None,
) -> Dict[str, float | str]:
    """Run independent games concurrently and retain examples only in RAM.

    Every worker owns a GameEnv and a MCTS tree. The single inference service
    groups requests from every tree into one CUDA/MPS forward pass. A worker
    receives the next game immediately after it returns its current result.
    Optimisation runs after actors have stopped, so it cannot race inference.
    """
    if opponent not in {"self_play", "random", "heuristic", "model", "mixed"}:
        raise ValueError("opponent must be self_play, random, heuristic, model or mixed")
    if not 0.0 <= random_opponent_rate <= 1.0:
        raise ValueError("random_opponent_rate must be between 0 and 1")
    if mixed_opponent not in {"random", "heuristic", "model"}:
        raise ValueError("mixed_opponent must be random, heuristic or model")
    if (opponent == "model" or mixed_opponent == "model") and not opponent_models:
        raise ValueError("model opponent mode requires at least one frozen opponent model")
    if not 0.0 <= teacher_replay_fraction <= 1.0:
        raise ValueError("teacher_replay_fraction must be between 0 and 1")
    if workers < 1:
        raise ValueError("workers must be at least one")
    if global_inference_batch_size < 1:
        raise ValueError("global_inference_batch_size must be at least one")
    run = run or TrainingRun.create("runs", "self_play")
    run.write_config(
        training=training,
        model=model_config,
        mcts=mcts_config,
        stage={
            "games": games,
            "updates_per_game": updates_per_game,
            "opponent": opponent,
            "random_opponent_rate": random_opponent_rate,
            "mixed_opponent": mixed_opponent,
            "teacher_replay_fraction": teacher_replay_fraction,
            "workers": workers,
            "global_inference_batch_size": global_inference_batch_size,
            "inference_wait_ms": inference_wait_ms,
            "opponent_model_weights": list(opponent_model_weights),
            "model_out": model_out,
        },
    )

    # Modern PolyEnv exposes a single native MctsPool over all active games.
    # Use it in preference to the legacy Python-worker scheduler below.
    try:
        from PolyEnv import MctsPool  # noqa: F401
    except ImportError:
        pass
    else:
        from .global_mcts_train import train_global_mcts_self_play
        return train_global_mcts_self_play(
            training, model_config, mcts_config, games, updates_per_game,
            opponent, random_opponent_rate, mixed_opponent, teacher_replay_fraction,
            active_envs=workers, model_out=model_out, model=model,
            opponent_models=opponent_models,
            opponent_model_weights=opponent_model_weights,
            game_callback=game_callback,
            run=run,
        )

    if opponent == "model" or mixed_opponent == "model":
        raise RuntimeError("frozen model opponents require PolyEnv's native MctsPool")

    device = select_device(training.device)
    model = model.to(device) if model is not None else PolicyValueBeliefNet(model_config).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    replay = ReplayBuffer(training.replay_capacity)
    teacher_replay = ReplayBuffer(training.replay_capacity)
    rng = np.random.default_rng(training.seed)
    schedule_rng = np.random.default_rng(training.seed ^ 0xA17C0)
    progress = ConsoleProgress()
    progress.stage(
        f"device: {device} | games: {games} | workers: {workers} | MCTS simulations: "
        f"{mcts_config.simulations} | particles: {mcts_config.particles} | "
        f"global inference batch: {global_inference_batch_size}"
    )

    # The schedule is decided before threading, making it independent of which
    # worker happens to finish first.
    schedules = []
    for game_index in range(games):
        use_anchor = opponent in {"random", "heuristic"} or (
            opponent == "mixed" and schedule_rng.random() < random_opponent_rate
        )
        opponent_agent = (mixed_opponent if opponent == "mixed" else opponent) if use_anchor else None
        schedules.append((use_anchor, opponent_agent))

    model.eval()
    template = BeliefMCTS(model, device, mcts_config, map_size=training.map_size, map_type=training.map_type)
    inference = BatchedInference(
        template._evaluate_packets_direct,
        max_batch_size=global_inference_batch_size,
        wait_ms=inference_wait_ms,
    )
    native_leaf_inference = NativeLeafInference(
        lambda leaves: inference.exclusive(lambda: template._evaluate_native_leaves_direct(leaves)),
        max_batch_size=global_inference_batch_size,
        wait_ms=inference_wait_ms,
    )

    def play_one(game_index: int):
        use_anchor, opponent_agent = schedules[game_index]
        planner = BeliefMCTS(
            model, device, mcts_config,
            inference=inference,
            native_leaf_inference=native_leaf_inference,
            map_size=training.map_size,
            map_type=training.map_type,
        )
        runner = SelfPlayRunner(planner, training, logger=None)
        examples, metrics = runner.play_game(
            training.seed + game_index,
            iteration=0,
            game_index=game_index,
            train_player=game_index % 2 if use_anchor else None,
            opponent_agent=opponent_agent,
        )
        teacher_examples = []
        if teacher_replay_fraction > 0.0:
            teacher_examples, _ = _teacher_game(training.seed + 1_000_000 + game_index, training)
        return use_anchor, examples, teacher_examples, metrics

    losses = []
    rewards = []
    exploration_actions = 0.0
    opponent_games = 0.0
    reward_totals: Dict[str, float] = {}
    action_totals: Dict[str, float] = {}
    completed_games = 0
    try:
        # Submitting all seeds gives ThreadPoolExecutor a work queue: an idle
        # worker picks the next game at once; no fixed batch barrier exists.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="polybot-game") as executor:
            futures = [executor.submit(play_one, game_index) for game_index in range(games)]
            with tqdm(total=games, desc=f"parallel MCTS self-play ({device})", unit="env", dynamic_ncols=True) as bar:
                for future in as_completed(futures):
                    use_anchor, examples, teacher_examples, metrics = future.result()
                    completed_games += 1
                    replay.extend(examples)
                    teacher_replay.extend(teacher_examples)
                    opponent_games += float(use_anchor)
                    exploration_actions += metrics.get("exploration_actions", 0.0)
                    rewards.extend(value for name, value in metrics.items() if name.startswith("shaped_reward_player_"))
                    for name, value in metrics.items():
                        if name.startswith("reward_"):
                            reward_totals[name] = reward_totals.get(name, 0.0) + value
                        if name == "model_actions" or name.startswith("model_action_"):
                            action_totals[name] = action_totals.get(name, 0.0) + value
                    run.log(
                        game=completed_games,
                        replay_examples=len(replay),
                        teacher_replay_examples=len(teacher_replay),
                        game_examples=len(examples),
                        opponent_game=float(use_anchor),
                        **metrics,
                    )
                    bar.update(1)
                    bar.set_postfix(examples=len(replay), moves=int(metrics["moves"]), workers=workers)
    finally:
        inference.close()
        native_leaf_inference.close()

    # Preserve the requested update budget while avoiding concurrent AdamW and
    # inference access to the same model parameters.
    if len(replay) >= training.batch_size:
        update_count = games * updates_per_game
        for update in tqdm(range(update_count), desc=f"optimising ({device})", unit="update", dynamic_ncols=True):
            teacher_count = min(int(round(training.batch_size * teacher_replay_fraction)), len(teacher_replay))
            self_play_count = training.batch_size - teacher_count
            if len(replay) < self_play_count:
                teacher_count = training.batch_size
                self_play_count = 0
            examples_batch = []
            if self_play_count:
                examples_batch.extend(replay.sample(self_play_count, rng, training.priority_replay_fraction))
            if teacher_count:
                examples_batch.extend(teacher_replay.sample(teacher_count, rng, training.priority_replay_fraction))
            update_loss = train_batch(model, optimiser, stack_examples(examples_batch, device), training)
            losses.append(update_loss)
            run.log(
                game=games,
                update=update + 1,
                replay_examples=len(replay),
                teacher_replay_examples=len(teacher_replay),
                **update_loss,
            )

    if model_out is not None:
        save_model(model_out, model, model_config)
    summary: Dict[str, float | str] = {
        "games": float(games),
        "workers": float(workers),
        "replay_examples": float(len(replay)),
        "teacher_replay_examples": float(len(teacher_replay)),
        "teacher_replay_fraction": teacher_replay_fraction,
        "mean_shaped_reward": float(np.mean(rewards)) if rewards else 0.0,
        "actual_exploration_rate": exploration_actions / max(1, len(replay)),
        "opponent_games": opponent_games,
    }
    total_examples = max(1, len(replay))
    for name, value in reward_totals.items():
        summary["mean_" + name] = value / total_examples
    model_actions = max(1.0, action_totals.get("model_actions", 0.0))
    for name, value in action_totals.items():
        if name != "model_actions":
            summary[name + "_rate"] = value / model_actions
    if losses:
        for name in losses[0]:
            summary[name] = float(np.mean([loss[name] for loss in losses]))
    summary.update(inference.stats())
    summary.update(native_leaf_inference.stats())
    summary["run_dir"] = str(run.finish(summary))
    return summary
