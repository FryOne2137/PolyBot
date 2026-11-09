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
from polybot.training.flat_experiments import FlatImprovements


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


def main(*, fixed_training_stage: str | None = None) -> None:
    """Run the league trainer.

    Dedicated entry points use ``fixed_training_stage`` so flat, belief and
    MCTS phases cannot accidentally be mixed by a command-line flag. The
    original script keeps its ``auto`` mode for backwards-compatible
    experiments.
    """
    if fixed_training_stage not in {None, "flat", "belief", "mcts"}:
        raise ValueError(
            "fixed_training_stage must be None, 'flat', 'belief' or 'mcts'"
        )
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
    if fixed_training_stage is None:
        parser.add_argument(
            "--training-stage",
            choices=("auto", "flat", "belief", "mcts"),
            default="auto",
            help=(
                "auto keeps the guarded curriculum; flat runs only raw-policy "
                "PPO/population training; belief runs only the frozen belief "
                "calibration stage; mcts requires an accepted belief checkpoint "
                "and runs the MCTS policy/value stage."
            ),
        )
    else:
        parser.set_defaults(training_stage=fixed_training_stage)
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
        help=(
            "MCTS games per absorption generation; use 0 only for raw flat mode "
            "and a positive value for mcts/auto"
        ),
    )
    parser.add_argument("--population-updates-per-game", type=int, default=2)
    parser.add_argument(
        "--population-optimisation-block-updates", type=int, default=32,
        help="Optimiser updates per reported replay block after frozen collection.",
    )
    parser.add_argument(
        "--population-optimisation-screen-games", type=int, default=4,
        help="Paired raw-policy guard games after each MCTS optimisation block; 0 disables it.",
    )
    parser.add_argument(
        "--population-optimisation-screen-max-regression", type=float, default=0.10,
        help="Maximum raw-screen score drop before model and AdamW are restored.",
    )
    parser.add_argument(
        "--value-bootstrap-games", type=int, default=64,
        help=(
            "Raw-policy games used to calibrate only value and belief before "
            "MCTS policy distillation; 0 disables this safety bootstrap"
        ),
    )
    parser.add_argument(
        "--value-bootstrap-updates-per-game", type=int, default=8,
        help="Optimisation updates per raw bootstrap game (policy loss remains zero)",
    )
    parser.add_argument(
        "--population-workers", type=int, default=16,
        help="Persistent native SelfPlayPool worker threads for population games",
    )
    parser.add_argument(
        "--population-max-envs", type=int, default=16,
        help="Maximum concurrent MCTS games during population training; bounds CUDA memory",
    )
    parser.add_argument(
        "--population-offline-collection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze the model while one full MCTS generation is collected, then train from that fixed replay",
    )
    parser.add_argument(
        "--population-belief-search-mode",
        choices=("strict", "root_shield"),
        default="strict",
        help=(
            "strict retries action-mismatched particles; root_shield keeps the "
            "public PolyEnv root actions authoritative and uses mismatch only "
            "for belief supervision/diagnostics during ordinary MCTS training"
        ),
    )
    parser.add_argument(
        "--belief-apprentice-path", type=Path,
        help="Persistent calibration checkpoint; defaults to <checkpoint-dir>/<prefix>-belief-apprentice.pt.",
    )
    parser.add_argument(
        "--belief-action-equivalence-target", type=float, default=0.10,
        help="Maximum initial particle action-mismatch rate before MCTS policy distillation begins.",
    )
    parser.add_argument(
        "--belief-action-equivalence-min-particles", type=int, default=512,
        help="Minimum checked candidate particles required to finish belief calibration.",
    )
    parser.add_argument(
        "--belief-calibration-eval-games", type=int, default=8,
        help="Fixed-seed post-update belief evaluation games; 0 disables only this diagnostic.",
    )
    parser.add_argument(
        "--belief-calibration-replay-path", type=Path,
        help="Persistent supervised belief replay; defaults to <checkpoint-dir>/<prefix>-belief-calibration-replay.pkl.",
    )
    parser.add_argument(
        "--belief-calibration-replay-capacity", type=int, default=50_000,
        help="Maximum number of recent belief calibration positions retained across runs.",
    )
    parser.add_argument(
        "--belief-replay-updates", type=int, default=512,
        help=(
            "Additional belief-only optimiser updates over persistent replay after "
            "each calibration generation; ignored by the MCTS-only stage."
        ),
    )
    parser.add_argument(
        "--population-mcts-critical-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep ordinary MCTS targets too; enable only for a late replay-compression phase",
    )
    parser.add_argument(
        "--population-mcts-critical-fraction", type=float, default=0.25,
        help="Maximum fraction of each MCTS root batch retained as confident targets",
    )
    parser.add_argument(
        "--population-mcts-critical-minimum-score", type=float, default=0.25,
        help="Minimum MCTS-vs-raw disagreement confidence for a retained target",
    )
    parser.add_argument(
        "--population-mcts-critical-replay-fraction", type=float, default=1.0,
        help="Replay share reserved for retained MCTS decisions",
    )
    parser.add_argument(
        "--population-selective-mcts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run native MCTS only on uncertain/audited population slots; direct policy handles the rest.",
    )
    parser.add_argument(
        "--population-selective-mcts-fraction", type=float, default=0.50,
        help="Share of active population slots selected for MCTS before each move.",
    )
    parser.add_argument(
        "--population-selective-mcts-minimum-envs", type=int, default=2,
        help="Minimum MCTS slots per active SelfPlayPool batch.",
    )
    parser.add_argument(
        "--population-selective-mcts-audit-fraction", type=float, default=0.10,
        help="Random audit share inside the selected MCTS budget.",
    )
    parser.add_argument(
        "--population-mcts-reanalyse-path", type=Path,
        help="Persistent critical MCTS positions (default: <checkpoint-dir>/mcts-reanalyse.pkl)",
    )
    parser.add_argument(
        "--population-mcts-reanalyse-capacity", type=int, default=4096,
        help="Maximum durable critical MCTS positions retained between league runs",
    )
    parser.add_argument(
        "--population-mcts-reanalyse-max-mb", type=int, default=64,
        help="Maximum RAM/disk payload for durable critical MCTS positions.",
    )
    parser.add_argument(
        "--population-mcts-reanalyse-positions", type=int, default=8,
        help="Old positions re-searched by the current candidate before each MCTS generation",
    )
    parser.add_argument(
        "--population-mcts-reanalyse-replay-fraction", type=float, default=0.50,
        help="Minimum replay-batch share reserved for freshly reanalysed positions",
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
        "--direct-gate-max-moves", type=int, default=None,
        help=(
            "Raw-policy horizon used by direct and population gates. "
            "train_flat.py inherits --max-moves when this is omitted; "
            "other entry points retain their 700-move default."
        ),
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
        "--direct-population-branch-count", type=int, default=1,
        help="Independent short PPO/PPG candidates started from the champion before the long phase",
    )
    parser.add_argument(
        "--direct-population-branch-steps", type=int, default=2_000,
        help="PPO/PPG steps for each short candidate branch",
    )
    parser.add_argument(
        "--direct-population-branch-gate-games", type=int, default=20,
        help="Paired held-out games against the champion used to select a branch",
    )
    parser.add_argument(
        "--direct-population-branch-min-champion-score", type=float, default=0.50,
        help="Minimum short-gate score versus the champion required before long training",
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
        "--direct-population-plateau-patience", type=int, default=2,
        help="Safe non-improving blocks before increasing rollout exploration.",
    )
    parser.add_argument(
        "--direct-population-plateau-temperature-increment",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--direct-population-plateau-max-temperature", type=float, default=1.20,
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
        "--flat-population-persistent-optimizer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Flat-only: retain AdamW, PPG and auxiliary replay across population blocks.",
    )
    parser.add_argument(
        "--flat-population-terminal-replay-capacity", type=int, default=0,
        help="Flat-only capacity of terminal-outcome value replay; 0 disables it.",
    )
    parser.add_argument(
        "--flat-population-terminal-replay-batch-size", type=int, default=0,
    )
    parser.add_argument(
        "--flat-population-terminal-replay-interval", type=int, default=1,
    )
    parser.add_argument(
        "--flat-population-terminal-replay-value-weight", type=float, default=0.0,
    )
    parser.add_argument(
        "--flat-population-synchronous-rollouts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Flat-only: update PPO on fixed-length fresh fragments while "
            "retaining full-game outcomes separately for value/PPG."
        ),
    )
    parser.add_argument("--flat-curriculum", action=argparse.BooleanOptionalAction, default=False,
                        help="Flat-only: varied regular tribes, board sizes and map types with one learner")
    parser.add_argument("--flat-tribes", default="all", help="all, or comma-separated regular tribe names/IDs")
    parser.add_argument("--flat-map-sizes", default="11,12,13,14,15,16")
    parser.add_argument("--flat-map-types", default="lakes,drylands,water_world")
    parser.add_argument("--flat-map-type-weights", default="0.60,0.25,0.15")
    parser.add_argument("--flat-small-map-probability", type=float, default=0.70)
    parser.add_argument("--flat-curriculum-intro-steps", type=int, default=8000)
    parser.add_argument("--flat-curriculum-expansion-steps", type=int, default=24000)
    parser.add_argument("--flat-domain-pools", type=int, default=3)
    parser.add_argument("--flat-domain-segment-steps", type=int, default=512)
    parser.add_argument("--flat-domain-max-moves", type=int, default=3200,
                        help="Mixed collector episode horizon; unfinished games bootstrap and never create terminal value labels")
    parser.add_argument("--flat-domain-evaluation-groups", type=int, default=6)
    parser.add_argument("--flat-evaluation-batch-games", type=int, default=64,
                        help="Mixed flat evaluation: maximum resident games sharing inference; positive and even.")
    parser.add_argument("--flat-evaluation-size-bucketing", choices=("nearby", "exact"), default="nearby",
                        help="Mixed evaluation: nearby pads board storage; exact uses the homogeneous map attention path.")
    parser.add_argument("--flat-domain-score-floor", type=float, default=0.40)
    parser.add_argument("--flat-domain-min-games", type=int, default=20)
    parser.add_argument("--flat-improvements", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--flat-value-updates", type=int, default=8)
    parser.add_argument("--flat-value-learning-rate", type=float, default=1e-4)
    parser.add_argument("--flat-value-batch-size", type=int, default=128)
    parser.add_argument("--flat-value-capacity", type=int, default=16384)
    parser.add_argument("--flat-value-validation-games", type=int, default=8)
    parser.add_argument("--flat-value-fit-backoffs", type=int, default=2,
                        help="Retry a rejected head-only fit with halved learning rates.")
    parser.add_argument("--flat-diagnostic-interval", type=int, default=2,
                        help="Full flat population benchmark every N blocks; champion screens still run each block.")
    parser.add_argument("--flat-diagnostic-improvement", type=float, default=0.05,
                        help="Champion-screen improvement triggering an earlier full benchmark.")
    parser.add_argument("--flat-schedule-steps", type=int, default=0,
                        help="Annealing budget; 0 reaches endpoints by the minimum flat curriculum budget.")
    parser.add_argument("--flat-resume-state", type=Path, default=None,
                        help="Resume flat weights/AdamW/RNG; native in-flight games restart.")
    parser.add_argument("--flat-paired-rollback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flat-cache-evaluations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flat-scientific-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flat-exploiter-objective", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--flat-rollout-steps-final", type=int, default=256,
        help="Final flat rollout horizon; 0 disables horizon annealing.",
    )
    parser.add_argument(
        "--flat-population-ppo-max-policy-age", type=int, default=None,
        help="Flat-only: actor ignores samples older than this many accepted PPO epochs/PPG phases.",
    )
    parser.add_argument(
        "--flat-population-policy-anchor-final-weight", type=float, default=None,
        help="Flat-only final champion-policy anchor after linear decay.",
    )
    parser.add_argument(
        "--flat-population-snapshot-archive-size", type=int, default=0,
        help="Flat-only number of accepted candidate snapshots used as PFSP opponents.",
    )
    parser.add_argument("--flat-population-pfsp-alpha", type=float, default=2.0)
    parser.add_argument(
        "--flat-population-pfsp-probability-floor", type=float, default=0.05,
    )
    parser.add_argument(
        "--flat-population-pfsp-score-ema-decay", type=float, default=0.80,
        help="Flat-only smoothing for opponent difficulty before PFSP reweights.",
    )
    parser.add_argument(
        "--flat-population-hard-counter-threshold", type=float, default=0.0,
        help=(
            "Flat-only: scores below this value are hard counters; 0 disables "
            "their dedicated collection quota."
        ),
    )
    parser.add_argument(
        "--flat-population-hard-counter-probability", type=float, default=0.0,
        help=(
            "Flat-only total collection share reserved for currently weak "
            "hard-counter opponents."
        ),
    )
    parser.add_argument(
        "--flat-population-regression-confirm-games", type=int, default=0,
        help=(
            "Flat-only paired games used to confirm a diagnostic regression "
            "against its weakest opponent; 0 preserves immediate rollback."
        ),
    )
    parser.add_argument(
        "--flat-population-champion-screen",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Flat-only: screen each PPO block against the champion before "
            "running the full population benchmark; clearly losing blocks "
            "skip the expensive historical evaluations."
        ),
    )
    parser.add_argument(
        "--flat-main-exploiter-probability", type=float, default=0.70,
        help="Flat-only main-exploiter share reserved for the current main snapshot.",
    )
    parser.add_argument(
        "--flat-main-exploiter-archive-score", type=float, default=0.55,
        help="Flat-only minimum held-out score versus current main to archive a counterstrategy.",
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
        "--refresh-promoted-curriculum",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Deprecated compatibility option; train_league always uses MCTS "
            "self-play and ignores legacy direct curricula"
        ),
    )
    parser.add_argument(
        "--champion-mode",
        choices=("flat", "mcts", "auto"),
        default="flat",
        help=(
            "flat promotes by raw policy; mcts enables the search ladder; "
            "auto switches from flat to frozen MCTS generations after persisted probes"
        ),
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
        "--flat-mcts-distill-max-envs",
        type=int,
        default=16,
        help="Maximum concurrent MCTS games during distillation; bounds CUDA memory",
    )
    parser.add_argument(
        "--flat-mcts-distill-max-moves",
        type=int,
        default=600,
        help="MCTS-teacher horizon; independent from the longer flat self-play horizon",
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
        default=(32, 64),
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
    parser.add_argument("--temperature", type=float, default=1.0, help="MCTS visit-sampling temperature before the drop move.")
    parser.add_argument(
        "--belief-particle-temperature", type=float, default=0.15,
        help="Completed belief-map sampling temperature; independent of MCTS action exploration.",
    )
    parser.add_argument("--temperature-drop-move", type=int, default=20, help="Use --late-temperature from this move onward; 0 keeps one temperature.")
    parser.add_argument("--late-temperature", type=float, default=0.0, help="MCTS visit-sampling temperature after --temperature-drop-move.")
    parser.add_argument("--root-dirichlet-epsilon", type=float, default=0.25, help="Self-play-only root-prior noise share; 0 disables it.")
    parser.add_argument("--root-dirichlet-concentration", type=float, default=10.0, help="Total Dirichlet concentration across legal root actions.")
    parser.add_argument("--mcts-max-actions", type=int, default=1024,
                        help="Capacity hint; exact native adapters reserve the full ActionSpace and never truncate legal observations.")
    parser.add_argument(
        "--mcts-top-actions", type=int, default=0,
        help="PolyBot-only leaf prior filter; 0 disables, 48 is a conservative experiment.",
    )
    parser.add_argument(
        "--mcts-random-actions", type=int, default=0,
        help="Extra random legal candidates per leaf in addition to top-K and mandatory tactics.",
    )
    parser.add_argument(
        "--mcts-min-prior-mass", type=float, default=0.0,
        help="Expand top-K until this neural prior mass is retained; 0 disables adaptive expansion.",
    )
    parser.add_argument("--strategic-prior-weight", type=float, default=1.0)
    parser.add_argument("--teacher-prior-weight", type=float, default=0.20)
    # A dedicated flat job must never inherit the league's MCTS-oriented CLI
    # defaults. Arguments explicitly supplied by the user are still validated
    # by ``train_promotion_league`` below.
    if fixed_training_stage == "flat":
        parser.set_defaults(
            champion_mode="flat",
            population_games=0,
            value_bootstrap_games=0,
            mcts_probe_games=0,
            flat_mcts_distill_games=0,
            # A resumed flat model already has a policy/value starting point.
            # Do not silently spend hours in the legacy teacher/objective
            # stages before the population PPO, PFSP and exploiter begin.
            direct_teacher_steps=0,
            direct_objective_steps=0,
            direct_actor_steps=0,
            # A sparse terminal game cannot improve a mature policy from a
            # few hundred completed episodes.  These are vector steps (not
            # individual actions): with the usual 16 environments this gives
            # the main line roughly 0.4--0.8M fresh transitions per round,
            # while still checking and rolling back every 4k steps.
            direct_population_steps=24_000,
            direct_population_max_steps=48_000,
            direct_population_block_steps=4_000,
            # Flat PPO can safely use every on-policy advantage.  The
            # terminal value replay is critic-only, so it cannot make the
            # actor off-policy.  These defaults are intentionally scoped to
            # train_flat.py and leave train_league.py unchanged.
            direct_population_ppo_top_advantage_fraction=1.0,
            direct_policy_anchor_weight=0.20,
            direct_value_anchor_weight=0.10,
            flat_population_persistent_optimizer=True,
            flat_population_terminal_replay_capacity=16_384,
            flat_population_terminal_replay_batch_size=256,
            flat_population_terminal_replay_interval=1,
            flat_population_terminal_replay_value_weight=0.25,
            flat_population_ppo_max_policy_age=1,
            flat_population_synchronous_rollouts=True,
            flat_improvements=True,
            flat_population_policy_anchor_final_weight=0.05,
            flat_population_snapshot_archive_size=6,
            flat_population_pfsp_alpha=2.0,
            flat_population_pfsp_probability_floor=0.05,
            flat_population_pfsp_score_ema_decay=0.80,
            # A narrow policy that merely beats the champion must spend a
            # substantial fraction of its own data fixing the counter that
            # defeats it.  The generic defaults stay unchanged for all other
            # entry points.
            flat_population_hard_counter_threshold=0.45,
            flat_population_hard_counter_probability=0.30,
            flat_population_regression_confirm_games=60,
            flat_population_champion_screen=True,
            flat_main_exploiter_probability=0.70,
            flat_main_exploiter_archive_score=0.55,
            breakthrough_exploiter=True,
        )
    args = parser.parse_args()

    if args.flat_curriculum and args.training_stage != "flat":
        parser.error("--flat-curriculum is available only for the flat training stage")
    if args.flat_curriculum and (not args.flat_population_persistent_optimizer or not args.flat_population_synchronous_rollouts):
        parser.error("--flat-curriculum requires persistent optimizer and synchronous rollouts")
    if args.flat_curriculum and (args.flat_evaluation_batch_games < 2 or args.flat_evaluation_batch_games % 2):
        parser.error("--flat-evaluation-batch-games must be positive, even and at least 2")

    # ``train_direct_population_curriculum`` deliberately uses one horizon
    # for both collection and its raw-policy gates.  Historically train_flat
    # silently replaced ``--max-moves`` with the generic 700-move direct
    # default here, so a flat run requested at 1200 moves still collected and
    # evaluated truncated 700-move games.  Keep the legacy league default,
    # but make a dedicated flat invocation follow its public --max-moves
    # unless the user explicitly supplies the direct override.
    resolved_direct_gate_max_moves = (
        int(args.direct_gate_max_moves)
        if args.direct_gate_max_moves is not None
        else (
            int(args.max_moves)
            if fixed_training_stage == "flat"
            else 700
        )
    )

    training = training_config_from_args(args)
    print(
        f"device: {select_device(training.device)} | league rounds: "
        f"{args.rounds} | visible action history: "
        f"{args.visible_action_history} | stage: {args.training_stage} | PPG: "
        f"{'on' if args.direct_population_ppg else 'off'}",
        flush=True,
    )
    print_summary(train_promotion_league(
        training,
        champion_in=args.champion_in,
        challenger_in=args.challenger_in,
        checkpoint_dir=args.checkpoint_dir,
        training_stage=args.training_stage,
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
        belief_apprentice_path=args.belief_apprentice_path,
        belief_action_equivalence_target=(
            args.belief_action_equivalence_target
        ),
        belief_action_equivalence_min_particles=(
            args.belief_action_equivalence_min_particles
        ),
        belief_calibration_eval_games=args.belief_calibration_eval_games,
        belief_calibration_replay_path=args.belief_calibration_replay_path,
        belief_calibration_replay_capacity=(
            args.belief_calibration_replay_capacity
        ),
        belief_replay_updates=args.belief_replay_updates,
        database_path=args.league_db,
        population_model_paths=args.population_model,
        population_size=args.population_size,
        population_games=args.population_games,
        population_updates_per_game=args.population_updates_per_game,
        population_optimisation_block_updates=(
            args.population_optimisation_block_updates
        ),
        population_optimisation_screen_games=(
            args.population_optimisation_screen_games
        ),
        population_optimisation_screen_max_regression=(
            args.population_optimisation_screen_max_regression
        ),
        value_bootstrap_games=args.value_bootstrap_games,
        value_bootstrap_updates_per_game=(
            args.value_bootstrap_updates_per_game
        ),
        population_workers=args.population_workers,
        population_max_envs=args.population_max_envs,
        population_offline_collection=args.population_offline_collection,
        population_belief_search_mode=args.population_belief_search_mode,
        population_mcts_critical_only=args.population_mcts_critical_only,
        population_mcts_critical_fraction=(
            args.population_mcts_critical_fraction
        ),
        population_mcts_critical_minimum_score=(
            args.population_mcts_critical_minimum_score
        ),
        population_mcts_critical_replay_fraction=(
            args.population_mcts_critical_replay_fraction
        ),
        population_selective_mcts=args.population_selective_mcts,
        population_selective_mcts_fraction=(
            args.population_selective_mcts_fraction
        ),
        population_selective_mcts_minimum_envs=(
            args.population_selective_mcts_minimum_envs
        ),
        population_selective_mcts_audit_fraction=(
            args.population_selective_mcts_audit_fraction
        ),
        population_mcts_reanalyse_path=args.population_mcts_reanalyse_path,
        population_mcts_reanalyse_capacity=(
            args.population_mcts_reanalyse_capacity
        ),
        population_mcts_reanalyse_max_bytes=(
            args.population_mcts_reanalyse_max_mb * 1024 * 1024
        ),
        population_mcts_reanalyse_positions=(
            args.population_mcts_reanalyse_positions
        ),
        population_mcts_reanalyse_replay_fraction=(
            args.population_mcts_reanalyse_replay_fraction
        ),
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
        direct_gate_max_moves=resolved_direct_gate_max_moves,
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
        direct_population_branch_count=args.direct_population_branch_count,
        direct_population_branch_steps=args.direct_population_branch_steps,
        direct_population_branch_gate_games=(
            args.direct_population_branch_gate_games
        ),
        direct_population_branch_min_champion_score=(
            args.direct_population_branch_min_champion_score
        ),
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
        direct_population_plateau_patience=(
            args.direct_population_plateau_patience
        ),
        direct_population_plateau_temperature_increment=(
            args.direct_population_plateau_temperature_increment
        ),
        direct_population_plateau_max_temperature=(
            args.direct_population_plateau_max_temperature
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
        flat_population_persistent_optimizer=(
            args.flat_population_persistent_optimizer
        ),
        flat_population_terminal_replay_capacity=(
            args.flat_population_terminal_replay_capacity
        ),
        flat_population_terminal_replay_batch_size=(
            args.flat_population_terminal_replay_batch_size
        ),
        flat_population_terminal_replay_interval=(
            args.flat_population_terminal_replay_interval
        ),
        flat_population_terminal_replay_value_weight=(
            args.flat_population_terminal_replay_value_weight
        ),
        flat_population_ppo_max_policy_age=(
            args.flat_population_ppo_max_policy_age
        ),
        flat_population_synchronous_rollouts=(
            args.flat_population_synchronous_rollouts
        ),
        flat_resume_state=args.flat_resume_state,
        flat_improvements=(FlatImprovements(
            value_updates=args.flat_value_updates,
            value_batch_size=args.flat_value_batch_size,
            value_learning_rate=args.flat_value_learning_rate,
            value_capacity=args.flat_value_capacity,
            value_validation_games=args.flat_value_validation_games,
            value_fit_backoffs=args.flat_value_fit_backoffs,
            diagnostic_interval=args.flat_diagnostic_interval,
            diagnostic_improvement=args.flat_diagnostic_improvement,
            schedule_steps=args.flat_schedule_steps,
            paired_rollback=args.flat_paired_rollback,
            cache_evaluations=args.flat_cache_evaluations,
            scientific_artifacts=args.flat_scientific_artifacts,
            exploiter_objective=args.flat_exploiter_objective,
            rollout_steps_final=args.flat_rollout_steps_final,
        ) if args.flat_improvements else None),
        flat_population_policy_anchor_final_weight=(
            args.flat_population_policy_anchor_final_weight
        ),
        flat_population_snapshot_archive_size=(
            args.flat_population_snapshot_archive_size
        ),
        flat_population_pfsp_alpha=args.flat_population_pfsp_alpha,
        flat_population_pfsp_probability_floor=(
            args.flat_population_pfsp_probability_floor
        ),
        flat_population_pfsp_score_ema_decay=(
            args.flat_population_pfsp_score_ema_decay
        ),
        flat_population_hard_counter_threshold=(
            args.flat_population_hard_counter_threshold
        ),
        flat_population_hard_counter_probability=(
            args.flat_population_hard_counter_probability
        ),
        flat_population_regression_confirm_games=(
            args.flat_population_regression_confirm_games
        ),
        flat_population_champion_screen=(
            args.flat_population_champion_screen
        ),
        flat_main_exploiter_probability=(
            args.flat_main_exploiter_probability
        ),
        flat_main_exploiter_archive_score=(
            args.flat_main_exploiter_archive_score
        ),
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
        refresh_promoted_curriculum=args.refresh_promoted_curriculum,
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
        flat_mcts_distill_max_envs=args.flat_mcts_distill_max_envs,
        flat_mcts_distill_max_moves=args.flat_mcts_distill_max_moves,
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
