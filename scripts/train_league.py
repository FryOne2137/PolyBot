"""PFSP population league with deterministic, regression-safe promotion."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_local_source_root

add_local_source_root()

from polybot.infrastructure.device import select_device
from polybot.training.cli import (
    add_environment_options,
    add_run_options,
    mcts_config_from_args,
    print_summary,
    training_config_from_args,
    training_run_from_args,
)
from polybot.training.league import train_promotion_league


def _simulation_ladder(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "simulation ladder must be comma-separated integers"
        ) from error
    if not result:
        raise argparse.ArgumentTypeError("simulation ladder must not be empty")
    if any(item < 1 for item in result) or tuple(sorted(set(result))) != result:
        raise argparse.ArgumentTypeError(
            "simulation ladder must be positive and strictly increasing"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train against a historical population and save only robust promotions"
    )
    add_environment_options(parser)
    # The league adds a separate geometry curriculum on top of policy/value
    # learning. A conservative default keeps that auxiliary head from
    # dominating the shared representation of an already competent champion.
    parser.set_defaults(bridge_constraint_weight=0.10)
    add_run_options(parser)
    parser.add_argument("--champion-in", type=Path, required=True, help="Initial frozen champion checkpoint")
    parser.add_argument(
        "--challenger-in", type=Path,
        help="Optional retained challenger to continue training against the frozen champion",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/league"))
    parser.add_argument("--checkpoint-prefix", default="champion")
    parser.add_argument(
        "--league-db", type=Path,
        help="SQLite history (default: <checkpoint-dir>/league.sqlite3)",
    )
    parser.add_argument(
        "--population-model", type=Path, action="append", default=[],
        help="Additional protected teacher/historical checkpoint; repeat as needed",
    )
    parser.add_argument("--population-size", type=int, default=6)
    parser.add_argument(
        "--population-games", type=int, default=0,
        help="MCTS training games; keep 0 while --champion-mode flat",
    )
    parser.add_argument("--population-updates-per-game", type=int, default=2)
    parser.add_argument(
        "--population-workers", type=int, default=16,
        help="Persistent native SelfPlayPool worker threads for population games",
    )
    parser.add_argument("--pfsp-alpha", type=float, default=2.0)
    parser.add_argument("--champion-probability", type=float, default=0.40)
    parser.add_argument("--minimum-sampling-probability", type=float, default=0.05)
    parser.add_argument(
        "--champion-anchor-weight", type=float, default=0.30,
        help="Frozen champion fraction blended into policy targets in both training phases",
    )
    parser.add_argument(
        "--champion-value-anchor-weight", type=float, default=0.25,
        help="Frozen champion value loss used as a trust region during MCTS training",
    )
    parser.add_argument("--direct-teacher-envs", type=int, default=128)
    parser.add_argument(
        "--direct-teacher-steps", type=int, default=2_000,
        help="Fast visible-objective imitation steps before MCTS",
    )
    parser.add_argument(
        "--direct-objective-envs", type=int, default=128,
        help="Parallel native environments for short multi-step objective rollouts",
    )
    parser.add_argument(
        "--direct-objective-steps", type=int, default=1_000,
        help="Short-horizon objective steps before unrestricted flat self-play",
    )
    parser.add_argument("--direct-objective-block-steps", type=int, default=500)
    parser.add_argument("--direct-objective-rollout-horizon", type=int, default=8)
    parser.add_argument("--direct-objective-updates-per-rollout", type=int, default=2)
    parser.add_argument(
        "--direct-objective-teacher-probability", type=float, default=0.35,
        help="Legal objective-teacher fraction while collecting short rollouts",
    )
    parser.add_argument("--direct-actor-envs", type=int, default=128)
    parser.add_argument(
        "--direct-actor-steps", type=int, default=1_000,
        help="On-policy direct-play steps before MCTS",
    )
    parser.add_argument("--direct-actor-block-steps", type=int, default=500)
    parser.add_argument(
        "--direct-actor-max-steps", type=int, default=5_000,
        help="Maximum direct-play steps before a round defers MCTS",
    )
    parser.add_argument("--direct-gate-games", type=int, default=30)
    parser.add_argument(
        "--direct-gate-max-moves", type=int, default=700,
        help="Longer raw-policy games used by direct and population gates",
    )
    parser.add_argument("--direct-gate-score", type=float, default=0.50)
    parser.add_argument("--direct-policy-anchor-weight", type=float, default=0.75)
    parser.add_argument("--direct-value-anchor-weight", type=float, default=0.50)
    parser.add_argument("--direct-teacher-weight", type=float, default=0.50)
    parser.add_argument("--direct-shaping-weight", type=float, default=0.20)
    parser.add_argument("--direct-readiness-score", type=float, default=0.50)
    parser.add_argument(
        "--direct-readiness-city-captures-per-game", type=float, default=0.05,
    )
    parser.add_argument(
        "--direct-readiness-completion-rate", type=float, default=0.05,
    )
    parser.add_argument("--direct-readiness-patience", type=int, default=2)
    parser.add_argument(
        "--direct-regression-patience", type=int, default=2,
        help="Advance with the best ready checkpoint after this many bad actor blocks",
    )
    parser.add_argument("--direct-population-envs", type=int, default=128)
    parser.add_argument(
        "--direct-population-steps", type=int, default=2_000,
        help="Minimum fast population steps without MCTS",
    )
    parser.add_argument(
        "--direct-population-max-steps", type=int, default=10_000,
        help="Maximum fast population steps before MCTS is deferred",
    )
    parser.add_argument(
        "--direct-population-block-steps", type=int, default=1_000,
    )
    parser.add_argument(
        "--direct-population-gate-games", type=int, default=100,
        help="Paired games per population checkpoint selection gate",
    )
    parser.add_argument(
        "--direct-population-diagnostic-gate-games", type=int, default=20,
        help="Cheap paired games after a PPO block; full gate runs once for its best checkpoint",
    )
    parser.add_argument("--direct-population-gate-score", type=float, default=0.50)
    parser.add_argument(
        "--direct-population-historical-floor", type=float, default=0.45,
    )
    parser.add_argument(
        "--direct-population-readiness-patience", type=int, default=2,
    )
    parser.add_argument(
        "--direct-population-regression-patience", type=int, default=2,
        help="Advance with the best ready checkpoint after persistent population regression",
    )
    parser.add_argument(
        "--direct-population-opponent-policy-weight", type=float, default=0.0,
        help="Deprecated: PPO learns only candidate actions",
    )
    parser.add_argument(
        "--direct-population-teacher-weight", type=float, default=0.0,
        help="Deprecated: PPO uses on-policy GAE rather than a heuristic target",
    )
    parser.add_argument(
        "--direct-population-shaping-weight", type=float, default=0.20,
        help="Initial visible-action shaping weight; it linearly decays during PPO",
    )
    parser.add_argument(
        "--direct-population-shaping-final-weight", type=float, default=0.0,
        help="Final visible-action shaping weight after the population phase",
    )
    parser.add_argument(
        "--direct-population-ppo-rollout-steps", type=int, default=32,
    )
    parser.add_argument("--direct-population-ppo-epochs", type=int, default=2)
    parser.add_argument(
        "--direct-population-ppo-minibatch-size", type=int, default=256,
    )
    parser.add_argument(
        "--direct-population-ppo-clip-range", type=float, default=0.15,
    )
    parser.add_argument(
        "--direct-population-ppo-gamma", type=float, default=1.0,
        help="Undiscounted terminal outcome objective for long Polytopia games",
    )
    parser.add_argument(
        "--direct-population-ppo-gae-lambda", type=float, default=0.95,
    )
    parser.add_argument(
        "--direct-population-ppo-terminal-advantage-weight",
        type=float,
        default=0.75,
        help="Fraction of PPO actor advantage taken from the final game outcome",
    )
    parser.add_argument(
        "--direct-population-ppo-top-advantage-fraction",
        type=float,
        default=0.50,
        help="Upper advantage fraction used by the PPO policy loss; value/PPG keep all positions",
    )
    parser.add_argument(
        "--direct-population-ppo-target-kl",
        type=float,
        default=0.010,
        help="Target full-policy behaviour KL per PPO epoch; also scales the soft champion anchor",
    )
    parser.add_argument(
        "--direct-population-ppo-hard-kl",
        type=float,
        default=0.040,
        help="Hard full-policy behaviour KL that reverts only the current PPO epoch",
    )
    parser.add_argument(
        "--direct-population-ppo-rollout-temperature", type=float, default=0.85,
    )
    parser.add_argument(
        "--direct-population-ppo-max-pending-examples", type=int, default=32_768,
        help="RAM safety cap; unfinished trajectories bootstrap only if this is exceeded",
    )
    parser.add_argument(
        "--direct-population-ppo-min-learning-rate", type=float, default=1.0e-5,
    )
    parser.add_argument(
        "--direct-population-ppo-lr-backoff", type=float, default=0.5,
    )
    parser.add_argument(
        "--direct-population-objective-replay-horizon", type=int, default=16,
        help="Recent candidate decisions credited when a visible objective succeeds",
    )
    parser.add_argument(
        "--direct-population-objective-replay-capacity", type=int, default=512,
    )
    parser.add_argument(
        "--direct-population-objective-replay-batch-size", type=int, default=32,
    )
    parser.add_argument(
        "--direct-population-objective-replay-interval", type=int, default=16,
        help="PPO optimiser updates between bounded self-imitation batches",
    )
    parser.add_argument(
        "--direct-population-objective-replay-weight", type=float, default=0.35,
    )
    parser.add_argument(
        "--direct-population-objective-replay-recency", type=float, default=0.92,
    )
    parser.add_argument(
        "--direct-population-objective-replay-max-epochs", type=int, default=4,
        help="Maximum average reuses credited to one successful public decision",
    )
    parser.add_argument(
        "--direct-population-ppg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable single-network PPG auxiliary value/distillation phases",
    )
    parser.add_argument(
        "--direct-population-ppg-interval", type=int, default=8,
        help="PPO rollouts between PPG auxiliary phases",
    )
    parser.add_argument(
        "--direct-population-ppg-buffer-capacity", type=int, default=8_192,
    )
    parser.add_argument(
        "--direct-population-ppg-samples-per-rollout", type=int, default=512,
        help="Bounded diverse positions retained from each PPO rollout",
    )
    parser.add_argument(
        "--direct-population-ppg-phase-samples", type=int, default=4_096,
    )
    parser.add_argument(
        "--direct-population-ppg-epochs", type=int, default=2,
    )
    parser.add_argument(
        "--direct-population-ppg-minibatch-size", type=int, default=256,
    )
    parser.add_argument(
        "--direct-population-ppg-policy-kl-weight", type=float, default=1.0,
        help="Protect the post-PPO policy while auxiliary value trains the encoder",
    )
    parser.add_argument(
        "--direct-population-ppg-value-weight", type=float, default=1.0,
    )
    parser.add_argument(
        "--direct-population-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Track a parameter EMA and select it when it wins the development population gate",
    )
    parser.add_argument(
        "--direct-population-ema-decay", type=float, default=0.999,
    )
    parser.add_argument(
        "--breakthrough-exploiter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train a second, champion-focused PPO branch without policy/value anchoring",
    )
    parser.add_argument(
        "--exploiter-champion-probability", type=float, default=0.75,
        help="Champion share for the free exploiter's opponent mix",
    )
    parser.add_argument(
        "--exploiter-policy-anchor-weight", type=float, default=0.0,
    )
    parser.add_argument(
        "--exploiter-value-anchor-weight", type=float, default=0.0,
    )
    parser.add_argument(
        "--exploiter-ppo-target-kl",
        type=float,
        default=0.010,
        help="Target full-policy behaviour KL per PPO epoch; this does not anchor the exploiter to the champion",
    )
    parser.add_argument(
        "--exploiter-ppo-hard-kl",
        type=float,
        default=0.040,
        help="Hard full-policy behaviour KL that reverts only the current PPO epoch",
    )
    parser.add_argument(
        "--exploiter-ppo-rollout-temperature", type=float, default=1.0,
    )
    parser.add_argument(
        "--post-bootstrap-population-steps", type=int, default=6_000,
        help="Minimum champion-population steps after the first promotion",
    )
    parser.add_argument(
        "--post-bootstrap-population-max-steps", type=int, default=12_000,
        help="Maximum champion-population steps after the first promotion",
    )
    parser.add_argument(
        "--post-bootstrap-objective-steps", type=int, default=0,
        help="Deprecated: established champions use PPO/GAE instead",
    )
    parser.add_argument(
        "--post-bootstrap-champion-probability", type=float, default=0.65,
        help="Current-champion share in post-bootstrap population training",
    )
    parser.add_argument(
        "--post-bootstrap-policy-anchor-weight", type=float, default=0.35,
    )
    parser.add_argument(
        "--post-bootstrap-teacher-weight", type=float, default=0.0,
        help="Deprecated: PPO uses on-policy GAE",
    )
    parser.add_argument(
        "--post-bootstrap-opponent-policy-weight", type=float, default=0.0,
        help="Deprecated: PPO learns only candidate actions",
    )
    parser.add_argument(
        "--champion-mode",
        choices=("flat", "mcts"),
        default="flat",
        help="Flat promotes by raw policy; mcts enables the legacy search ladder",
    )
    parser.add_argument(
        "--flat-mcts-distill-games", type=int, default=0,
        help="MCTS expert games distilled into the raw policy before its flat promotion gate",
    )
    parser.add_argument(
        "--flat-mcts-distill-simulations", type=int, default=32,
    )
    parser.add_argument(
        "--flat-mcts-distill-updates-per-game", type=int, default=2,
    )
    parser.add_argument(
        "--flat-mcts-distill-gate-games", type=int, default=20,
        help="Paired raw-policy games used to keep or roll back MCTS distillation",
    )
    parser.add_argument(
        "--flat-mcts-distill-policy-anchor-weight", type=float, default=0.20,
    )
    parser.add_argument(
        "--flat-mcts-distill-value-anchor-weight", type=float, default=0.25,
    )
    parser.add_argument(
        "--flat-mcts-distill-critical-fraction",
        type=float,
        default=0.25,
        help="Top fraction of candidate MCTS roots eligible as critical decisions",
    )
    parser.add_argument(
        "--flat-mcts-distill-critical-minimum-score",
        type=float,
        default=0.15,
        help="Minimum confident MCTS-vs-policy disagreement score",
    )
    parser.add_argument(
        "--flat-mcts-distill-critical-replay-fraction",
        type=float,
        default=0.50,
        help="Replay-batch share reserved for selected critical MCTS decisions",
    )
    parser.add_argument(
        "--mcts-probe-games", type=int, default=10,
        help="Fast paired diagnostic games; 0 disables the MCTS readiness probe",
    )
    parser.add_argument("--mcts-probe-simulations", type=int, default=32)
    parser.add_argument("--mcts-probe-score", type=float, default=0.55)
    parser.add_argument(
        "--mcts-probe-min-improvement", type=float, default=0.0,
        help="Strict score margin by which search@32 must beat raw policy on the same maps",
    )
    parser.add_argument("--mcts-probe-patience", type=int, default=2)
    parser.add_argument(
        "--mcts-simulation-ladder", type=_simulation_ladder,
        default=(16, 32, 64),
        help="Comma-separated search curriculum; every failed rung is rolled back",
    )
    parser.add_argument("--mcts-gate-games", type=int, default=20)
    parser.add_argument("--mcts-gate-score", type=float, default=0.50)
    parser.add_argument("--rounds", type=int, default=10, help="Candidate train-and-gate cycles")
    parser.add_argument("--promotion-games", type=int, default=100, help="Fixed-seed paired games per promotion gate")
    parser.add_argument("--promotion-score", type=float, default=0.55, help="Win=1 / draw=0.5 score required")
    parser.add_argument("--promotion-lower-bound", type=float, default=0.50)
    parser.add_argument(
        "--historical-games", type=int, default=20,
        help="Paired held-out games against every retained historical model",
    )
    parser.add_argument("--population-score", type=float, default=0.52)
    parser.add_argument("--historical-score-floor", type=float, default=0.45)
    parser.add_argument("--elo-k", type=float, default=32.0)
    parser.add_argument("--evaluation-seed", type=int, default=1_500_000_007, help="Held-out paired-map schedule")
    parser.add_argument("--evaluation-seed-stride", type=int, default=20_000_033)
    parser.add_argument("--round-seed-stride", type=int, default=10_000_019)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument(
        "--steps", type=int, default=0,
        help="Legacy same-network MCTS decisions; keep 0 for regression-safe curriculum",
    )
    parser.add_argument(
        "--updates", type=int, default=0,
        help="Legacy same-network optimiser updates; keep 0 with --steps 0",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        help=(
            "Native worker count for VectorGameEnv gates and SelfPlayPool; "
            "default uses available logical CPU cores"
        ),
    )
    parser.add_argument("--max-pending-leaves", type=int, default=4, help="Virtual-loss leaves per tree, 1-8")
    parser.add_argument(
        "--visible-action-history",
        type=int,
        default=32,
        help=(
            "Fog-safe events remembered by PPO, replay, flat gates and MCTS "
            "(0 disables, default: 32)"
        ),
    )
    parser.add_argument("--belief-retries", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--exploration-rate", type=float, default=0.15,
        help="Sampled candidate-turn fraction during direct population play",
    )
    parser.add_argument("--teacher-policy-target-weight", type=float, default=0.10)
    parser.add_argument("--priority-replay-fraction", type=float, default=0.35)
    parser.add_argument("--td-lambda", type=float, default=0.95)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--particles", type=int, default=2)
    parser.add_argument("--mcts-max-actions", type=int, default=1024)
    parser.add_argument(
        "--mcts-top-actions", type=int, default=0,
        help="PolyBot-only leaf prior filter; 0 disables, 48 is a conservative experiment.",
    )
    parser.add_argument(
        "--mcts-random-actions", type=int, default=0,
        help="Extra random legal candidates per leaf in addition to top-K and mandatory tactics.",
    )
    parser.add_argument("--strategic-prior-weight", type=float, default=1.0)
    parser.add_argument("--teacher-prior-weight", type=float, default=0.20)
    args = parser.parse_args()

    training = training_config_from_args(args)
    print(
        f"device: {select_device(training.device)} | league rounds: "
        f"{args.rounds} | visible action history: "
        f"{args.visible_action_history} | PPG: "
        f"{'on' if args.direct_population_ppg else 'off'}",
        flush=True,
    )
    print_summary(train_promotion_league(
        training,
        champion_in=args.champion_in,
        challenger_in=args.challenger_in,
        checkpoint_dir=args.checkpoint_dir,
        rounds=args.rounds,
        promotion_games=args.promotion_games,
        promotion_score=args.promotion_score,
        promotion_lower_bound=args.promotion_lower_bound,
        evaluation_seed=args.evaluation_seed,
        evaluation_seed_stride=args.evaluation_seed_stride,
        round_seed_stride=args.round_seed_stride,
        checkpoint_prefix=args.checkpoint_prefix,
        mcts=mcts_config_from_args(args),
        num_envs=args.num_envs,
        steps=args.steps,
        updates=args.updates,
        num_threads=args.num_threads,
        max_pending_leaves=args.max_pending_leaves,
        visible_action_history=args.visible_action_history,
        belief_retries=args.belief_retries,
        database_path=args.league_db,
        population_model_paths=args.population_model,
        population_size=args.population_size,
        population_games=args.population_games,
        population_updates_per_game=args.population_updates_per_game,
        population_workers=args.population_workers,
        pfsp_alpha=args.pfsp_alpha,
        champion_probability=args.champion_probability,
        minimum_sampling_probability=args.minimum_sampling_probability,
        champion_anchor_weight=args.champion_anchor_weight,
        champion_value_anchor_weight=args.champion_value_anchor_weight,
        direct_teacher_envs=args.direct_teacher_envs,
        direct_teacher_steps=args.direct_teacher_steps,
        direct_objective_envs=args.direct_objective_envs,
        direct_objective_steps=args.direct_objective_steps,
        direct_objective_block_steps=args.direct_objective_block_steps,
        direct_objective_rollout_horizon=args.direct_objective_rollout_horizon,
        direct_objective_updates_per_rollout=(
            args.direct_objective_updates_per_rollout
        ),
        direct_objective_teacher_probability=(
            args.direct_objective_teacher_probability
        ),
        direct_actor_envs=args.direct_actor_envs,
        direct_actor_steps=args.direct_actor_steps,
        direct_actor_block_steps=args.direct_actor_block_steps,
        direct_actor_max_steps=args.direct_actor_max_steps,
        direct_gate_games=args.direct_gate_games,
        direct_gate_max_moves=args.direct_gate_max_moves,
        direct_gate_score=args.direct_gate_score,
        direct_policy_anchor_weight=args.direct_policy_anchor_weight,
        direct_value_anchor_weight=args.direct_value_anchor_weight,
        direct_teacher_weight=args.direct_teacher_weight,
        direct_shaping_weight=args.direct_shaping_weight,
        direct_readiness_score=args.direct_readiness_score,
        direct_readiness_city_captures_per_game=(
            args.direct_readiness_city_captures_per_game
        ),
        direct_readiness_completion_rate=(
            args.direct_readiness_completion_rate
        ),
        direct_readiness_patience=args.direct_readiness_patience,
        direct_regression_patience=args.direct_regression_patience,
        direct_population_envs=args.direct_population_envs,
        direct_population_steps=args.direct_population_steps,
        direct_population_max_steps=args.direct_population_max_steps,
        direct_population_block_steps=args.direct_population_block_steps,
        direct_population_gate_games=args.direct_population_gate_games,
        direct_population_diagnostic_gate_games=(
            args.direct_population_diagnostic_gate_games
        ),
        direct_population_gate_score=args.direct_population_gate_score,
        direct_population_historical_floor=(
            args.direct_population_historical_floor
        ),
        direct_population_readiness_patience=(
            args.direct_population_readiness_patience
        ),
        direct_population_regression_patience=(
            args.direct_population_regression_patience
        ),
        direct_population_opponent_policy_weight=(
            args.direct_population_opponent_policy_weight
        ),
        direct_population_teacher_weight=(
            args.direct_population_teacher_weight
        ),
        direct_population_shaping_weight=(
            args.direct_population_shaping_weight
        ),
        direct_population_shaping_final_weight=(
            args.direct_population_shaping_final_weight
        ),
        direct_population_ppo_rollout_steps=(
            args.direct_population_ppo_rollout_steps
        ),
        direct_population_ppo_epochs=args.direct_population_ppo_epochs,
        direct_population_ppo_minibatch_size=(
            args.direct_population_ppo_minibatch_size
        ),
        direct_population_ppo_clip_range=(
            args.direct_population_ppo_clip_range
        ),
        direct_population_ppo_gamma=args.direct_population_ppo_gamma,
        direct_population_ppo_gae_lambda=(
            args.direct_population_ppo_gae_lambda
        ),
        direct_population_ppo_terminal_advantage_weight=(
            args.direct_population_ppo_terminal_advantage_weight
        ),
        direct_population_ppo_top_advantage_fraction=(
            args.direct_population_ppo_top_advantage_fraction
        ),
        direct_population_ppo_target_kl=(
            args.direct_population_ppo_target_kl
        ),
        direct_population_ppo_hard_kl=(
            args.direct_population_ppo_hard_kl
        ),
        direct_population_ppo_rollout_temperature=(
            args.direct_population_ppo_rollout_temperature
        ),
        direct_population_ppo_max_pending_examples=(
            args.direct_population_ppo_max_pending_examples
        ),
        direct_population_ppo_min_learning_rate=(
            args.direct_population_ppo_min_learning_rate
        ),
        direct_population_ppo_lr_backoff=(
            args.direct_population_ppo_lr_backoff
        ),
        direct_population_objective_replay_horizon=(
            args.direct_population_objective_replay_horizon
        ),
        direct_population_objective_replay_capacity=(
            args.direct_population_objective_replay_capacity
        ),
        direct_population_objective_replay_batch_size=(
            args.direct_population_objective_replay_batch_size
        ),
        direct_population_objective_replay_interval=(
            args.direct_population_objective_replay_interval
        ),
        direct_population_objective_replay_weight=(
            args.direct_population_objective_replay_weight
        ),
        direct_population_objective_replay_recency=(
            args.direct_population_objective_replay_recency
        ),
        direct_population_objective_replay_max_epochs=(
            args.direct_population_objective_replay_max_epochs
        ),
        direct_population_ppg_enabled=args.direct_population_ppg,
        direct_population_ppg_interval=(
            args.direct_population_ppg_interval
        ),
        direct_population_ppg_buffer_capacity=(
            args.direct_population_ppg_buffer_capacity
        ),
        direct_population_ppg_samples_per_rollout=(
            args.direct_population_ppg_samples_per_rollout
        ),
        direct_population_ppg_phase_samples=(
            args.direct_population_ppg_phase_samples
        ),
        direct_population_ppg_epochs=args.direct_population_ppg_epochs,
        direct_population_ppg_minibatch_size=(
            args.direct_population_ppg_minibatch_size
        ),
        direct_population_ppg_policy_kl_weight=(
            args.direct_population_ppg_policy_kl_weight
        ),
        direct_population_ppg_value_weight=(
            args.direct_population_ppg_value_weight
        ),
        direct_population_ema_enabled=args.direct_population_ema,
        direct_population_ema_decay=args.direct_population_ema_decay,
        breakthrough_exploiter=args.breakthrough_exploiter,
        exploiter_champion_probability=args.exploiter_champion_probability,
        exploiter_policy_anchor_weight=args.exploiter_policy_anchor_weight,
        exploiter_value_anchor_weight=args.exploiter_value_anchor_weight,
        exploiter_ppo_target_kl=args.exploiter_ppo_target_kl,
        exploiter_ppo_hard_kl=args.exploiter_ppo_hard_kl,
        exploiter_ppo_rollout_temperature=(
            args.exploiter_ppo_rollout_temperature
        ),
        post_bootstrap_population_steps=(
            args.post_bootstrap_population_steps
        ),
        post_bootstrap_population_max_steps=(
            args.post_bootstrap_population_max_steps
        ),
        post_bootstrap_objective_steps=args.post_bootstrap_objective_steps,
        post_bootstrap_champion_probability=(
            args.post_bootstrap_champion_probability
        ),
        post_bootstrap_policy_anchor_weight=(
            args.post_bootstrap_policy_anchor_weight
        ),
        post_bootstrap_teacher_weight=args.post_bootstrap_teacher_weight,
        post_bootstrap_opponent_policy_weight=(
            args.post_bootstrap_opponent_policy_weight
        ),
        champion_mode=args.champion_mode,
        flat_mcts_distill_games=args.flat_mcts_distill_games,
        flat_mcts_distill_simulations=(
            args.flat_mcts_distill_simulations
        ),
        flat_mcts_distill_updates_per_game=(
            args.flat_mcts_distill_updates_per_game
        ),
        flat_mcts_distill_gate_games=args.flat_mcts_distill_gate_games,
        flat_mcts_distill_policy_anchor_weight=(
            args.flat_mcts_distill_policy_anchor_weight
        ),
        flat_mcts_distill_value_anchor_weight=(
            args.flat_mcts_distill_value_anchor_weight
        ),
        flat_mcts_distill_critical_fraction=(
            args.flat_mcts_distill_critical_fraction
        ),
        flat_mcts_distill_critical_minimum_score=(
            args.flat_mcts_distill_critical_minimum_score
        ),
        flat_mcts_distill_critical_replay_fraction=(
            args.flat_mcts_distill_critical_replay_fraction
        ),
        mcts_probe_games=args.mcts_probe_games,
        mcts_probe_simulations=args.mcts_probe_simulations,
        mcts_probe_score=args.mcts_probe_score,
        mcts_probe_minimum_improvement=(
            args.mcts_probe_min_improvement
        ),
        mcts_probe_patience=args.mcts_probe_patience,
        mcts_simulation_ladder=args.mcts_simulation_ladder,
        mcts_gate_games=args.mcts_gate_games,
        mcts_gate_score=args.mcts_gate_score,
        historical_games=args.historical_games,
        population_score=args.population_score,
        historical_score_floor=args.historical_score_floor,
        elo_k=args.elo_k,
        run=training_run_from_args(args, "league"),
    ))


if __name__ == "__main__":
    main()
