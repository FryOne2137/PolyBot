"""Promotion-gated, model-only training league for the native SelfPlayPool.

The fast pool currently performs same-network self-play.  This runner keeps
that high-throughput collection path, then prevents regressions by promoting a
candidate only after a deterministic MCTS match against the frozen champion.
Only promoted PyTorch checkpoints are written; games and replay data remain in
memory for each round.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict

from ..config import MCTSConfig, ModelConfig, TrainingConfig
from ..evaluation.matches import evaluate_matches
from ..infrastructure.device import select_device
from ..infrastructure.model_io import load_model, save_model
from .run_artifacts import TrainingRun
from .self_play_pool_train import train_self_play_pool


def _match_score(result: Dict[str, float]) -> float:
    games = max(1.0, float(result["games"]))
    return (float(result["model_a_wins"]) + 0.5 * float(result["draws"])) / games


def _next_checkpoint_path(directory: Path, prefix: str) -> Path:
    """Return an unused, monotonically numbered champion checkpoint path."""
    index = 1
    while True:
        path = directory / f"{prefix}-{index:03d}.pt"
        if not path.exists():
            return path
        index += 1


def train_promotion_league(
    training: TrainingConfig,
    *,
    champion_in: Path,
    checkpoint_dir: Path,
    rounds: int,
    promotion_games: int,
    promotion_score: float,
    evaluation_seed: int,
    checkpoint_prefix: str,
    mcts: MCTSConfig,
    num_envs: int,
    steps: int,
    updates: int,
    num_threads: int | None,
    max_pending_leaves: int,
    visible_action_history: int,
    belief_retries: int,
    run: TrainingRun | None = None,
) -> Dict[str, float | str]:
    """Train sequential candidates and retain only promotion-approved models.

    Each round starts from the current champion.  The collection seed changes
    per round, whereas the promotion match uses one fixed seed schedule and
    alternates player slots, making every gate comparable.  A rejected
    candidate never reaches disk.
    """
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if promotion_games < 1:
        raise ValueError("promotion_games must be positive")
    if not 0.0 <= promotion_score <= 1.0:
        raise ValueError("promotion_score must be between 0 and 1")
    if not checkpoint_prefix:
        raise ValueError("checkpoint_prefix must not be empty")
    if not champion_in.is_file():
        raise FileNotFoundError(f"champion checkpoint does not exist: {champion_in}")

    device = select_device(training.device)
    run = run or TrainingRun.create("runs", "league")
    run.write_config(
        training=training,
        mcts=mcts,
        stage={
            "champion_in": champion_in,
            "checkpoint_dir": checkpoint_dir,
            "rounds": rounds,
            "promotion_games": promotion_games,
            "promotion_score": promotion_score,
            "evaluation_seed": evaluation_seed,
            "checkpoint_prefix": checkpoint_prefix,
            "num_envs": num_envs,
            "steps": steps,
            "updates": updates,
            "num_threads": num_threads,
            "max_pending_leaves": max_pending_leaves,
            "visible_action_history": visible_action_history,
            "belief_retries": belief_retries,
        },
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    champion_path = champion_in
    next_checkpoint = _next_checkpoint_path(checkpoint_dir, checkpoint_prefix)
    promotions = 0
    last_candidate_score = 0.0
    last_training: Dict[str, float | str] = {}
    last_match: Dict[str, float] = {}

    # Greedy evaluation and an unanchored prior ensure that the gate measures
    # the checkpoint rather than stochastic exploration or teacher bias.
    evaluation_mcts = replace(
        mcts, temperature=0.0, strategic_prior_weight=0.0, teacher_prior_weight=0.0
    )

    for round_index in range(1, rounds + 1):
        round_training = replace(training, seed=training.seed + round_index - 1)
        print(
            f"league round {round_index}/{rounds} | champion: {champion_path} | "
            f"collection seed: {round_training.seed}",
            flush=True,
        )
        candidate, model_config = load_model(champion_path, device)
        last_training = train_self_play_pool(
            round_training,
            model_config,
            mcts,
            num_envs=num_envs,
            steps=steps,
            updates=updates,
            num_threads=num_threads,
            max_pending_leaves=max_pending_leaves,
            visible_action_history=visible_action_history,
            belief_retries=belief_retries,
            # Candidates remain only in RAM until the promotion gate passes.
            model_out=None,
            model=candidate,
            run=run.child(f"round_{round_index:03d}"),
        )

        champion, champion_config = load_model(champion_path, device)
        if champion_config != model_config:
            raise RuntimeError("candidate and champion ModelConfig differ")
        evaluation_training = replace(training, seed=evaluation_seed)
        last_match = evaluate_matches(
            candidate,
            device,
            evaluation_training,
            evaluation_mcts,
            promotion_games,
            champion,
        )
        last_candidate_score = _match_score(last_match)
        promoted = last_candidate_score >= promotion_score
        print(
            f"league gate {round_index}/{rounds} | score={last_candidate_score:.4f} | "
            f"threshold={promotion_score:.4f} | promoted={int(promoted)}",
            flush=True,
        )
        if promoted:
            promotions += 1
            champion_path = next_checkpoint
            save_model(champion_path, candidate, model_config)
            print(f"league promotion {promotions}: saved {champion_path}", flush=True)
            next_checkpoint = _next_checkpoint_path(checkpoint_dir, checkpoint_prefix)

        run.log(
            round=round_index,
            candidate_score=last_candidate_score,
            promoted=float(promoted),
            promotions=promotions,
            **{f"train_{key}": value for key, value in last_training.items()},
            **{f"match_{key}": value for key, value in last_match.items()},
        )

        # Drop GPU references before the next candidate is loaded.  This does
        # not persist any rejected model or training state.
        del candidate, champion

    summary: Dict[str, float | str] = {
        "league_rounds": float(rounds),
        "league_promotions": float(promotions),
        "league_last_candidate_score": last_candidate_score,
        "league_promotion_threshold": promotion_score,
        "league_champion_path": str(champion_path),
        "league_evaluation_seed": float(evaluation_seed),
    }
    summary.update({f"last_train_{key}": value for key, value in last_training.items()})
    summary.update({f"last_match_{key}": value for key, value in last_match.items()})
    summary["run_dir"] = str(run.finish(summary))
    return summary
