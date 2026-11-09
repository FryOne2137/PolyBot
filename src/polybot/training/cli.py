"""Shared command-line options for repository-local training scripts."""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path
from typing import Dict

from ..config import MCTSConfig, TrainingConfig
from .run_artifacts import TrainingRun


def add_environment_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=2137)
    parser.add_argument(
        "--random-seed", action="store_true",
        help="Draw a fresh base seed for this run and print it in the final summary.",
    )
    parser.add_argument("--map-size", type=int, default=11)
    parser.add_argument("--map-type", default="lakes", choices=["lakes", "drylands", "water_world"])
    parser.add_argument("--max-moves", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--replay-capacity", type=int, default=20_000)
    parser.add_argument("--training-policy-weight", type=float, default=1.0)
    parser.add_argument("--training-value-weight", type=float, default=1.0)
    parser.add_argument("--belief-weight", type=float, default=0.35)
    parser.add_argument("--legal-action-belief-weight", type=float, default=2.0)
    parser.add_argument("--engine-action-constraint-weight", type=float, default=0.25)
    parser.add_argument("--bridge-constraint-weight", type=float, default=1.0)
    parser.add_argument("--belief-presence-focal-gamma", type=float, default=1.5)
    parser.add_argument("--belief-entity-consistency-weight", type=float, default=0.35)
    parser.add_argument("--bridge-replay-fraction", type=float, default=0.25)
    parser.add_argument("--engine-action-constraint-replay-fraction", type=float, default=0.25)
    parser.add_argument("--max-replay-epochs", type=float, default=6.0)
    parser.add_argument("--capture-city-prior-bonus", type=float, default=2.0)
    parser.add_argument("--capture-capital-prior-bonus", type=float, default=4.0)
    parser.add_argument("--found-city-prior-bonus", type=float, default=3.0)
    parser.add_argument("--attack-prior-bonus", type=float, default=0.8)
    parser.add_argument("--end-turn-prior-penalty", type=float, default=2.0)


def add_run_options(parser: argparse.ArgumentParser) -> None:
    """Add Ultralytics-style output naming shared by every trainer."""
    parser.add_argument("--project", type=Path, default=Path("runs"), help="Root directory for training artifacts")
    parser.add_argument("--name", default="train", help="Run name; existing names increment to train2, train3, ...")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse --project/TYPE/--name instead of incrementing")


def training_run_from_args(args: argparse.Namespace, stage: str) -> TrainingRun:
    return TrainingRun.create(
        args.project,
        stage,
        args.name,
        exist_ok=args.exist_ok,
        arguments=vars(args),
    )


def training_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    seed = secrets.randbelow(2**31 - 1) + 1 if getattr(args, "random_seed", False) else args.seed
    seed = (int(seed) - 1) % (2**31 - 1) + 1
    curriculum = None
    if getattr(args, "flat_curriculum", False):
        from ..config import FlatCurriculumConfig
        from PolyEnv import NAME_TO_TRIBE
        tribe_text = args.flat_tribes.strip().lower()
        tribes = tuple(range(1, 13)) if tribe_text == "all" else tuple(
            int(value) if value.isdigit() else int(NAME_TO_TRIBE[value])
            for value in (item.strip().lower() for item in tribe_text.split(","))
        )
        curriculum = FlatCurriculumConfig(
            tribes=tribes, map_sizes=tuple(int(v) for v in args.flat_map_sizes.split(",")),
            map_types=tuple(v.strip().lower() for v in args.flat_map_types.split(",")),
            map_type_weights=tuple(float(v) for v in args.flat_map_type_weights.split(",")),
            small_map_probability=args.flat_small_map_probability,
            intro_steps=args.flat_curriculum_intro_steps, expansion_steps=args.flat_curriculum_expansion_steps,
            pools=args.flat_domain_pools, segment_steps=args.flat_domain_segment_steps,
            episode_max_moves=args.flat_domain_max_moves,
            evaluation_groups=args.flat_domain_evaluation_groups,
            domain_score_floor=args.flat_domain_score_floor, domain_min_games=args.flat_domain_min_games,
        )
    return TrainingConfig(
        seed=seed, flat_curriculum=curriculum,
        flat_evaluation_batch_games=getattr(args, "flat_evaluation_batch_games", 64),
        device=args.device,
        map_size=args.map_size,
        map_type=args.map_type,
        max_moves=args.max_moves,
        replay_capacity=args.replay_capacity,
        batch_size=getattr(args, "batch_size", 16),
        exploration_rate=getattr(args, "exploration_rate", 0.05),
        priority_replay_fraction=getattr(args, "priority_replay_fraction", 0.35),
        teacher_policy_target_weight=getattr(args, "teacher_policy_target_weight", 0.10),
        td_lambda=getattr(args, "td_lambda", 1.0),
        learning_rate=args.learning_rate,
        policy_weight=args.training_policy_weight,
        value_weight=args.training_value_weight,
        belief_weight=args.belief_weight,
        legal_action_belief_weight=getattr(args, "legal_action_belief_weight", 2.0),
        engine_action_constraint_weight=getattr(args, "engine_action_constraint_weight", 0.25),
        bridge_constraint_weight=getattr(args, "bridge_constraint_weight", 1.0),
        belief_presence_focal_gamma=getattr(args, "belief_presence_focal_gamma", 1.5),
        belief_entity_consistency_weight=getattr(args, "belief_entity_consistency_weight", 0.35),
        bridge_replay_fraction=getattr(args, "bridge_replay_fraction", 0.25),
        engine_action_constraint_replay_fraction=getattr(args, "engine_action_constraint_replay_fraction", 0.25),
        max_replay_epochs=getattr(args, "max_replay_epochs", 6.0),
    )


def mcts_config_from_args(args: argparse.Namespace) -> MCTSConfig:
    return MCTSConfig(
        simulations=args.simulations,
        particles=args.particles,
        inference_batch_size=getattr(args, "inference_batch_size", 32),
        temperature=getattr(args, "temperature", 1.0),
        belief_particle_temperature=getattr(args, "belief_particle_temperature", 0.15),
        root_dirichlet_epsilon=getattr(args, "root_dirichlet_epsilon", 0.25),
        root_dirichlet_concentration=getattr(args, "root_dirichlet_concentration", 10.0),
        temperature_drop_move=getattr(args, "temperature_drop_move", 20),
        late_temperature=getattr(args, "late_temperature", 0.0),
        native_threads=getattr(args, "mcts_threads", None),
        native_max_actions=getattr(args, "mcts_max_actions", 1024),
        search_top_actions=getattr(args, "mcts_top_actions", 0),
        search_random_actions=getattr(args, "mcts_random_actions", 0),
        search_min_prior_mass=getattr(args, "mcts_min_prior_mass", 0.0),
        strategic_prior_weight=getattr(args, "strategic_prior_weight", 1.0),
        teacher_prior_weight=getattr(args, "teacher_prior_weight", 0.20),
        capture_city_prior_bonus=getattr(args, "capture_city_prior_bonus", 2.0),
        capture_capital_prior_bonus=getattr(args, "capture_capital_prior_bonus", 4.0),
        found_city_prior_bonus=getattr(args, "found_city_prior_bonus", 3.0),
        attack_prior_bonus=getattr(args, "attack_prior_bonus", 0.8),
        end_turn_prior_penalty=getattr(args, "end_turn_prior_penalty", 2.0),
    )


def print_summary(summary: Dict[str, object]) -> None:
    """Console output only; deliberately no metrics file is created."""
    print(json.dumps(summary, indent=2, sort_keys=True))
