"""Population-based self-play league with regression-safe promotion gates."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from ..config import MCTSConfig, TrainingConfig
from ..infrastructure.device import select_device
from ..infrastructure.model_io import load_model, save_model
from .league_db import LeagueDatabase
from .belief_recovery import BeliefQualityError, BeliefRecoveryRequired
from .direct_curriculum import (
    model_state_on_cpu,
    restore_model_state,
    train_direct_policy_curriculum,
)
from .routed_self_play_pool import (
    evaluate_matches_pool,
    require_routed_self_play_pool,
    train_population_pool,
)
from .run_artifacts import TrainingRun
from .self_play_pool_train import train_self_play_pool
from .vector_population import (
    evaluate_direct_population_matches,
    train_direct_population_curriculum,
)


_MAX_SEED = 2**31 - 1


def _offset_seed(seed: int, offset: int) -> int:
    return (int(seed) - 1 + int(offset)) % _MAX_SEED + 1


def _next_checkpoint_path(directory: Path, prefix: str) -> Path:
    """Return an unused, monotonically numbered champion checkpoint path."""
    index = 1
    while True:
        path = directory / f"{prefix}-{index:03d}.pt"
        if not path.exists():
            return path
        index += 1


def _validate_options(
    *,
    rounds: int,
    promotion_games: int,
    historical_games: int,
    promotion_score: float,
    promotion_lower_bound: float,
    population_score: float,
    historical_score_floor: float,
    evaluation_seed_stride: int,
    round_seed_stride: int,
    population_size: int,
    population_games: int,
    population_updates_per_game: int,
    population_workers: int,
    pfsp_alpha: float,
    champion_probability: float,
    minimum_sampling_probability: float,
    champion_anchor_weight: float,
    elo_k: float,
) -> None:
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if promotion_games < 2 or promotion_games % 2:
        raise ValueError("promotion_games must be positive and even")
    if historical_games < 2 or historical_games % 2:
        raise ValueError("historical_games must be positive and even")
    for name, value in {
        "promotion_score": promotion_score,
        "promotion_lower_bound": promotion_lower_bound,
        "population_score": population_score,
        "historical_score_floor": historical_score_floor,
        "champion_probability": champion_probability,
        "minimum_sampling_probability": minimum_sampling_probability,
        "champion_anchor_weight": champion_anchor_weight,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")
    if population_size < 1:
        raise ValueError("population_size must be positive")
    if population_games < 0 or population_updates_per_game < 0:
        raise ValueError("population games and updates must not be negative")
    if population_games and population_workers < 1:
        raise ValueError("population_workers must be positive")
    if pfsp_alpha <= 0.0:
        raise ValueError("pfsp_alpha must be positive")
    if elo_k <= 0.0:
        raise ValueError("elo_k must be positive")
    if round_seed_stride == 0:
        raise ValueError("round_seed_stride must not be zero")
    if rounds > 1 and evaluation_seed_stride == 0:
        raise ValueError("evaluation_seed_stride must rotate held-out maps")


def train_promotion_league(
    training: TrainingConfig,
    *,
    champion_in: Path,
    challenger_in: Path | None = None,
    checkpoint_dir: Path,
    rounds: int,
    promotion_games: int,
    promotion_score: float,
    promotion_lower_bound: float,
    evaluation_seed: int,
    evaluation_seed_stride: int,
    round_seed_stride: int,
    checkpoint_prefix: str,
    mcts: MCTSConfig,
    num_envs: int,
    steps: int,
    updates: int,
    num_threads: int | None,
    max_pending_leaves: int,
    visible_action_history: int,
    belief_retries: int,
    database_path: Path | None = None,
    population_model_paths: Sequence[Path] = (),
    population_size: int = 6,
    population_games: int = 64,
    population_updates_per_game: int = 2,
    population_workers: int = 16,
    pfsp_alpha: float = 2.0,
    champion_probability: float = 0.40,
    minimum_sampling_probability: float = 0.05,
    historical_games: int = 20,
    population_score: float = 0.52,
    historical_score_floor: float = 0.45,
    elo_k: float = 32.0,
    champion_anchor_weight: float = 0.30,
    champion_value_anchor_weight: float = 0.25,
    direct_teacher_envs: int = 128,
    direct_teacher_steps: int = 0,
    direct_actor_envs: int = 128,
    direct_actor_steps: int = 0,
    direct_actor_block_steps: int = 250,
    direct_actor_max_steps: int = 3_000,
    direct_gate_games: int = 20,
    direct_gate_max_moves: int = 700,
    direct_gate_score: float = 0.50,
    direct_policy_anchor_weight: float = 0.75,
    direct_value_anchor_weight: float = 0.50,
    direct_teacher_weight: float = 0.50,
    direct_shaping_weight: float = 0.20,
    direct_readiness_score: float = 0.50,
    direct_readiness_city_captures_per_game: float = 0.05,
    direct_readiness_completion_rate: float = 0.05,
    direct_readiness_patience: int = 2,
    direct_population_envs: int = 128,
    direct_population_steps: int = 0,
    direct_population_max_steps: int = 10_000,
    direct_population_block_steps: int = 1_000,
    direct_population_gate_games: int = 4,
    direct_population_gate_score: float = 0.50,
    direct_population_historical_floor: float = 0.45,
    direct_population_readiness_patience: int = 2,
    direct_population_opponent_policy_weight: float = 0.50,
    direct_population_teacher_weight: float = 0.25,
    direct_population_shaping_weight: float = 0.20,
    mcts_simulation_ladder: Sequence[int] = (),
    mcts_gate_games: int = 20,
    mcts_gate_score: float = 0.50,
    run: TrainingRun | None = None,
) -> Dict[str, float | str]:
    """Train candidates against a PFSP population and promote robust winners.

    Every round combines high-throughput same-network ISMCTS with games against
    a weighted mixture of the champion and historical checkpoints. Promotion
    requires improvement over the champion and no serious population-wide
    regression. SQLite stores game/metric lineage; tensors remain in ``.pt``
    files referenced by path and SHA-256 rather than database BLOBs.
    """
    _validate_options(
        rounds=rounds,
        promotion_games=promotion_games,
        historical_games=historical_games,
        promotion_score=promotion_score,
        promotion_lower_bound=promotion_lower_bound,
        population_score=population_score,
        historical_score_floor=historical_score_floor,
        evaluation_seed_stride=evaluation_seed_stride,
        round_seed_stride=round_seed_stride,
        population_size=population_size,
        population_games=population_games,
        population_updates_per_game=population_updates_per_game,
        population_workers=population_workers,
        pfsp_alpha=pfsp_alpha,
        champion_probability=champion_probability,
        minimum_sampling_probability=minimum_sampling_probability,
        champion_anchor_weight=champion_anchor_weight,
        elo_k=elo_k,
    )
    if not checkpoint_prefix:
        raise ValueError("checkpoint_prefix must not be empty")
    if num_envs < 2:
        raise ValueError("num_envs must be at least two for paired SelfPlayPool gates")
    if steps < 0 or updates < 0:
        raise ValueError("same-network MCTS steps and updates must not be negative")
    if steps == 0 and updates:
        raise ValueError("updates require a positive same-network MCTS step count")
    if direct_gate_games < 2 or direct_gate_games % 2:
        raise ValueError("direct_gate_games must be positive and even")
    if direct_population_gate_games < 2 or direct_population_gate_games % 2:
        raise ValueError(
            "direct_population_gate_games must be positive and even"
        )
    if direct_gate_max_moves < 1:
        raise ValueError("direct_gate_max_moves must be positive")
    if mcts_gate_games < 2 or mcts_gate_games % 2:
        raise ValueError("mcts_gate_games must be positive and even")
    for name, value in {
        "direct_gate_score": direct_gate_score,
        "direct_policy_anchor_weight": direct_policy_anchor_weight,
        "direct_value_anchor_weight": direct_value_anchor_weight,
        "direct_teacher_weight": direct_teacher_weight,
        "direct_shaping_weight": direct_shaping_weight,
        "direct_readiness_score": direct_readiness_score,
        "direct_readiness_completion_rate": direct_readiness_completion_rate,
        "direct_population_gate_score": direct_population_gate_score,
        "direct_population_historical_floor": (
            direct_population_historical_floor
        ),
        "direct_population_opponent_policy_weight": (
            direct_population_opponent_policy_weight
        ),
        "direct_population_teacher_weight": direct_population_teacher_weight,
        "direct_population_shaping_weight": direct_population_shaping_weight,
        "mcts_gate_score": mcts_gate_score,
        "champion_value_anchor_weight": champion_value_anchor_weight,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")
    if direct_readiness_city_captures_per_game < 0.0:
        raise ValueError(
            "direct_readiness_city_captures_per_game must not be negative"
        )
    if direct_actor_max_steps < direct_actor_steps:
        raise ValueError(
            "direct_actor_max_steps must not be smaller than direct_actor_steps"
        )
    if direct_readiness_patience < 1:
        raise ValueError("direct_readiness_patience must be positive")
    if direct_population_steps < 0:
        raise ValueError("direct_population_steps must not be negative")
    if direct_population_max_steps < direct_population_steps:
        raise ValueError(
            "direct_population_max_steps must not be smaller than its minimum"
        )
    if direct_population_steps and (
        direct_population_envs < 2 or direct_population_block_steps < 1
    ):
        raise ValueError(
            "direct population envs and block steps must be positive"
        )
    if direct_population_readiness_patience < 1:
        raise ValueError(
            "direct_population_readiness_patience must be positive"
        )
    ladder = tuple(int(value) for value in mcts_simulation_ladder)
    if any(value < 1 for value in ladder) or tuple(sorted(set(ladder))) != ladder:
        raise ValueError("mcts_simulation_ladder must be strictly increasing")
    population_rungs = ladder or ((int(mcts.simulations),) if population_games else ())
    population_game_allocations = tuple(
        population_games // max(1, len(population_rungs))
        + int(index < population_games % max(1, len(population_rungs)))
        for index in range(len(population_rungs))
    )
    # Check the multi-agent binding before spending hours on the first
    # same-network rollout phase.
    require_routed_self_play_pool()
    if not champion_in.is_file():
        raise FileNotFoundError(f"champion checkpoint does not exist: {champion_in}")
    if challenger_in is not None and not challenger_in.is_file():
        raise FileNotFoundError(
            f"challenger checkpoint does not exist: {challenger_in}"
        )
    for path in population_model_paths:
        if not path.is_file():
            raise FileNotFoundError(f"population checkpoint does not exist: {path}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    database_path = database_path or checkpoint_dir / "league.sqlite3"
    resolved_evaluation_seed = _offset_seed(evaluation_seed, 0)
    device = select_device(training.device)
    run = run or TrainingRun.create("runs", "league")
    evaluation_mcts = replace(
        mcts,
        temperature=0.0,
        strategic_prior_weight=0.0,
        teacher_prior_weight=0.0,
    )
    league_config: Dict[str, Any] = {
        "training": training,
        "mcts": mcts,
        "champion_in": champion_in,
        "challenger_in": challenger_in,
        "checkpoint_dir": checkpoint_dir,
        "database_path": database_path,
        "population_model_paths": list(population_model_paths),
        "rounds": rounds,
        "promotion_games": promotion_games,
        "historical_games": historical_games,
        "promotion_score": promotion_score,
        "promotion_lower_bound": promotion_lower_bound,
        "population_score": population_score,
        "historical_score_floor": historical_score_floor,
        "evaluation_seed": resolved_evaluation_seed,
        "evaluation_seed_stride": evaluation_seed_stride,
        "round_seed_stride": round_seed_stride,
        "population_size": population_size,
        "population_games": population_games,
        "population_updates_per_game": population_updates_per_game,
        "population_workers": population_workers,
        "pfsp_alpha": pfsp_alpha,
        "champion_probability": champion_probability,
        "minimum_sampling_probability": minimum_sampling_probability,
        "champion_anchor_weight": champion_anchor_weight,
        "champion_value_anchor_weight": champion_value_anchor_weight,
        "elo_k": elo_k,
        "checkpoint_prefix": checkpoint_prefix,
        "num_envs": num_envs,
        "steps": steps,
        "updates": updates,
        "num_threads": num_threads,
        "max_pending_leaves": max_pending_leaves,
        "visible_action_history": visible_action_history,
        "belief_retries": belief_retries,
        "direct_teacher_envs": direct_teacher_envs,
        "direct_teacher_steps": direct_teacher_steps,
        "direct_actor_envs": direct_actor_envs,
        "direct_actor_steps": direct_actor_steps,
        "direct_actor_block_steps": direct_actor_block_steps,
        "direct_actor_max_steps": direct_actor_max_steps,
        "direct_gate_games": direct_gate_games,
        "direct_gate_max_moves": direct_gate_max_moves,
        "direct_gate_score": direct_gate_score,
        "direct_policy_anchor_weight": direct_policy_anchor_weight,
        "direct_value_anchor_weight": direct_value_anchor_weight,
        "direct_teacher_weight": direct_teacher_weight,
        "direct_shaping_weight": direct_shaping_weight,
        "direct_readiness_score": direct_readiness_score,
        "direct_readiness_city_captures_per_game": (
            direct_readiness_city_captures_per_game
        ),
        "direct_readiness_completion_rate": direct_readiness_completion_rate,
        "direct_readiness_patience": direct_readiness_patience,
        "direct_population_envs": direct_population_envs,
        "direct_population_steps": direct_population_steps,
        "direct_population_max_steps": direct_population_max_steps,
        "direct_population_block_steps": direct_population_block_steps,
        "direct_population_gate_games": direct_population_gate_games,
        "direct_population_gate_score": direct_population_gate_score,
        "direct_population_historical_floor": (
            direct_population_historical_floor
        ),
        "direct_population_readiness_patience": (
            direct_population_readiness_patience
        ),
        "direct_population_opponent_policy_weight": (
            direct_population_opponent_policy_weight
        ),
        "direct_population_teacher_weight": direct_population_teacher_weight,
        "direct_population_shaping_weight": direct_population_shaping_weight,
        "mcts_simulation_ladder": list(ladder),
        "population_game_allocations": list(population_game_allocations),
        "mcts_gate_games": mcts_gate_games,
        "mcts_gate_score": mcts_gate_score,
    }
    run.write_config(**league_config)

    champion_path = champion_in
    challenger_path = challenger_in
    challenger_id: int | None = None
    promotions = 0
    last_candidate_score = 0.0
    last_population_score = 0.0
    last_training: Dict[str, Any] = {}
    current_round_id: list[int | None] = [None]

    with LeagueDatabase(database_path) as database:
        database_run_id = database.start_run(run.directory, league_config)

        def metric_sink(stage_path: Path, row: Mapping[str, Any]) -> None:
            try:
                relative_stage = stage_path.relative_to(run.directory)
                stage = "league" if relative_stage == Path(".") else str(relative_stage)
            except ValueError:
                stage = str(stage_path)
            database.record_metric_row(database_run_id, current_round_id[0], stage, row)

        run.set_metric_sink(metric_sink)
        try:
            initial_model, initial_config = load_model(champion_path, device)
            champion_id = database.register_checkpoint(
                champion_path,
                initial_config,
                run_id=database_run_id,
                status="initial",
                protected=True,
            )
            del initial_model
            if challenger_path is not None:
                challenger_model, challenger_config = load_model(
                    challenger_path, device
                )
                if challenger_config != initial_config:
                    raise RuntimeError(
                        "challenger checkpoint has a different ModelConfig"
                    )
                challenger_id = database.register_checkpoint(
                    challenger_path,
                    challenger_config,
                    run_id=database_run_id,
                    parent_model_id=champion_id,
                    status="challenger",
                    protected=False,
                    in_population=False,
                )
                if challenger_id == champion_id:
                    challenger_path = None
                    challenger_id = None
                del challenger_model
            for population_path in population_model_paths:
                population_model, population_config = load_model(population_path, device)
                if population_config != initial_config:
                    raise RuntimeError(
                        f"population checkpoint has a different ModelConfig: {population_path}"
                    )
                database.register_checkpoint(
                    population_path,
                    population_config,
                    run_id=database_run_id,
                    status="initial",
                    protected=True,
                    protect_existing=True,
                )
                del population_model
        except BaseException as error:
            database.finish_run(
                database_run_id,
                "failed",
                {"error": repr(error), "champion_path": str(champion_path)},
            )
            raise

        try:
            for round_index in range(1, rounds + 1):
                round_training = replace(
                    training,
                    seed=_offset_seed(training.seed, (round_index - 1) * round_seed_stride),
                )
                round_evaluation_seed = _offset_seed(
                    resolved_evaluation_seed,
                    (round_index - 1) * evaluation_seed_stride,
                )
                round_id = database.start_round(
                    database_run_id,
                    round_index,
                    champion_id,
                    round_training.seed,
                    round_evaluation_seed,
                )
                current_round_id[0] = round_id
                candidate_id: int | None = None
                round_training_summary: Dict[str, Any] = {}
                try:
                    print(
                        f"league round {round_index}/{rounds} | champion: {champion_path} | "
                        f"collection seed: {round_training.seed}",
                        flush=True,
                    )
                    candidate_source = challenger_path or champion_path
                    candidate, model_config = load_model(candidate_source, device)
                    champion, champion_config = load_model(champion_path, device)
                    if champion_config != model_config:
                        raise RuntimeError("candidate and champion ModelConfig differ")
                    candidate_id = database.create_candidate(
                        database_run_id,
                        challenger_id or champion_id,
                        f"candidate-round-{round_index:03d}",
                        model_config,
                    )
                    database.attach_candidate(round_id, candidate_id)

                    population_rows = database.select_population(champion_id, population_size)
                    population_ids = [int(row["id"]) for row in population_rows]
                    population_paths = [Path(str(row["checkpoint_path"])) for row in population_rows]
                    probabilities, reference_scores = database.opponent_probabilities(
                        champion_id,
                        population_ids,
                        pfsp_alpha=pfsp_alpha,
                        champion_probability=champion_probability,
                        probability_floor=minimum_sampling_probability,
                    )
                    database.record_opponent_selection(
                        round_id, population_ids, probabilities, reference_scores
                    )
                    population_training_seed = _offset_seed(
                        round_training.seed, 5_000_003
                    )
                    direct_population_training_seed = _offset_seed(
                        round_training.seed, 2_500_009
                    )
                    evaluation_map_seeds: set[int] = set()
                    for opponent_index, opponent_id in enumerate(population_ids):
                        games_for_opponent = (
                            promotion_games if opponent_id == champion_id else historical_games
                        )
                        match_seed = _offset_seed(
                            round_evaluation_seed, opponent_index * 1_000_003
                        )
                        opponent_map_seeds = {
                            match_seed + game_index // 2
                            for game_index in range(games_for_opponent)
                        }
                        if evaluation_map_seeds & opponent_map_seeds:
                            raise ValueError("held-out evaluation schedules overlap")
                        evaluation_map_seeds.update(opponent_map_seeds)
                    direct_gate_seed = _offset_seed(
                        round_evaluation_seed, 300_000_007
                    )
                    evaluation_map_seeds.update(
                        direct_gate_seed + game_index // 2
                        for game_index in range(direct_gate_games)
                    )
                    direct_population_gate_seed = _offset_seed(
                        round_evaluation_seed, 350_000_009
                    )
                    for opponent_index in range(len(population_ids)):
                        opponent_gate_seed = _offset_seed(
                            direct_population_gate_seed,
                            opponent_index * 1_000_003,
                        )
                        evaluation_map_seeds.update(
                            opponent_gate_seed + game_index // 2
                            for game_index in range(
                                direct_population_gate_games
                            )
                        )
                    # Reuse one development arena across search depths so a
                    # later rung cannot silently erase a score earned by an
                    # earlier rung. The final promotion arena remains separate.
                    mcts_gate_seeds = [
                        _offset_seed(round_evaluation_seed, 400_000_007)
                    ] * len(population_rungs)
                    for seed in mcts_gate_seeds:
                        evaluation_map_seeds.update(
                            seed + game_index // 2
                            for game_index in range(mcts_gate_games)
                        )
                    known_training_map_seeds: set[int] = set()
                    for rung_index, rung_games in enumerate(
                        population_game_allocations
                    ):
                        rung_seed = _offset_seed(
                            population_training_seed,
                            rung_index * 2_000_003,
                        )
                        known_training_map_seeds.update(
                            rung_seed + game_index
                            for game_index in range(rung_games)
                        )
                    if evaluation_map_seeds & known_training_map_seeds:
                        raise ValueError(
                            "population training maps overlap held-out evaluation maps"
                        )
                    if round_training.seed in evaluation_map_seeds:
                        raise ValueError(
                            "SelfPlayPool collection seed overlaps held-out evaluation maps"
                        )
                    if direct_population_training_seed in evaluation_map_seeds:
                        raise ValueError(
                            "direct population seed overlaps held-out evaluation maps"
                        )
                    print(
                        "league population | "
                        + ", ".join(
                            f"{path.stem}: {probability:.1%}"
                            for path, probability in zip(population_paths, probabilities)
                        ),
                        flush=True,
                    )

                    direct_summary: Dict[str, Any] = {}
                    if direct_teacher_steps or direct_actor_steps:
                        def record_direct_gate_game(
                            stage: str,
                            block: int,
                            game: Mapping[str, Any],
                        ) -> None:
                            row = dict(game)
                            row["curriculum_stage"] = stage
                            row["curriculum_block"] = int(block)
                            database.record_match(
                                run_id=database_run_id,
                                round_id=round_id,
                                phase=f"direct_curriculum_gate_{stage}_{block}",
                                model_a_id=candidate_id,
                                model_b_id=champion_id,
                                game=row,
                            )

                        direct_summary = train_direct_policy_curriculum(
                            round_training,
                            model_config,
                            candidate,
                            champion,
                            teacher_envs=direct_teacher_envs,
                            teacher_steps=direct_teacher_steps,
                            actor_envs=direct_actor_envs,
                            actor_steps=direct_actor_steps,
                            actor_block_steps=direct_actor_block_steps,
                            actor_max_steps=direct_actor_max_steps,
                            num_threads=num_threads,
                            max_actions=mcts.native_max_actions,
                            gate_games=direct_gate_games,
                            gate_seed=direct_gate_seed,
                            gate_max_moves=direct_gate_max_moves,
                            gate_score=direct_gate_score,
                            policy_anchor_weight=direct_policy_anchor_weight,
                            value_anchor_weight=direct_value_anchor_weight,
                            teacher_weight=direct_teacher_weight,
                            shaping_weight=direct_shaping_weight,
                            readiness_score=direct_readiness_score,
                            readiness_city_captures_per_game=(
                                direct_readiness_city_captures_per_game
                            ),
                            readiness_completion_rate=(
                                direct_readiness_completion_rate
                            ),
                            readiness_patience=direct_readiness_patience,
                            gate_game_callback=record_direct_gate_game,
                            run=run.child(
                                f"round_{round_index:03d}/direct_curriculum"
                            ),
                        )
                    round_training_summary["direct"] = direct_summary

                    direct_enabled = bool(
                        direct_teacher_steps or direct_actor_steps
                    )
                    direct_ready = (
                        not direct_enabled
                        or bool(direct_summary.get("direct_ready", 0.0))
                    )
                    if not direct_ready:
                        # MCTS is deliberately unavailable until the raw
                        # network can expand and finish games on held-out maps.
                        # Preserve a non-regressing challenger so the next
                        # round continues the direct curriculum instead of
                        # restarting from the teacher.
                        best_direct_score = float(
                            direct_summary.get("direct_best_score", 0.0)
                        )
                        retain_direct_challenger = best_direct_score >= 0.50
                        if retain_direct_challenger:
                            retained_path = _next_checkpoint_path(
                                checkpoint_dir,
                                f"{checkpoint_prefix}-challenger",
                            )
                            save_model(retained_path, candidate, model_config)
                            database.retain_candidate(candidate_id, retained_path)
                            challenger_path = retained_path
                            challenger_id = candidate_id
                        else:
                            database.finalise_candidate(
                                candidate_id, promoted_path=None
                            )
                            challenger_path = None
                            challenger_id = None
                        round_training_summary["pool"] = {}
                        round_training_summary["direct_population"] = {}
                        round_training_summary["population"] = {
                            "mcts_skipped_not_ready": 1.0,
                            "mcts_ladder": [],
                        }
                        last_training = dict(round_training_summary)
                        last_candidate_score = 0.0
                        last_population_score = 0.0
                        direct_gate_summary = {
                            "promoted": 0.0,
                            "valid": 1.0,
                            "direct_ready": 0.0,
                            "mcts_started": 0.0,
                            "retained_challenger": float(
                                retain_direct_challenger
                            ),
                            "direct_best_score": best_direct_score,
                            "direct_best_city_captures_per_game": float(
                                direct_summary.get(
                                    "direct_best_city_captures_per_game", 0.0
                                )
                            ),
                            "direct_best_completion_rate": float(
                                direct_summary.get(
                                    "direct_best_completion_rate", 0.0
                                )
                            ),
                            "reason": "direct_policy_not_ready_for_mcts",
                        }
                        database.finish_round(
                            round_id,
                            champion_id,
                            False,
                            round_training_summary,
                            direct_gate_summary,
                        )
                        run.log(
                            round=round_index,
                            direct_ready=0.0,
                            mcts_started=0.0,
                            retained_challenger=float(
                                retain_direct_challenger
                            ),
                            promotions=promotions,
                        )
                        database.flush()
                        print(
                            "league direct policy not ready | MCTS skipped | "
                            f"score={best_direct_score:.4f} | "
                            + (
                                "training will continue from the retained "
                                "challenger in the next round"
                                if retain_direct_challenger
                                else "candidate rejected; next round returns "
                                "to the champion"
                            ),
                            flush=True,
                        )
                        del candidate, champion
                        continue

                    frozen_models = []
                    for population_path in population_paths:
                        if population_path.resolve() == champion_path.resolve():
                            frozen_model, frozen_config = champion, champion_config
                        else:
                            frozen_model, frozen_config = load_model(
                                population_path, device
                            )
                        if frozen_config != model_config:
                            raise RuntimeError(
                                "population checkpoint has a different "
                                f"ModelConfig: {population_path}"
                            )
                        frozen_models.append(frozen_model.eval())

                    direct_population_summary: Dict[str, Any] = {}
                    if direct_population_steps:
                        def record_direct_population_gate_game(
                            stage: str,
                            block: int,
                            opponent_index: int,
                            game: Mapping[str, Any],
                        ) -> None:
                            row = dict(game)
                            row["curriculum_stage"] = stage
                            row["curriculum_block"] = int(block)
                            database.record_match(
                                run_id=database_run_id,
                                round_id=round_id,
                                phase=(
                                    "direct_population_gate_"
                                    f"{stage}_{block}"
                                ),
                                model_a_id=candidate_id,
                                model_b_id=population_ids[opponent_index],
                                game=row,
                            )

                        def record_direct_population_training_game(
                            block: int,
                            game: Mapping[str, Any],
                        ) -> None:
                            opponent_index = int(
                                game.get("opponent_model_index", 0)
                            )
                            row = dict(game)
                            row["curriculum_block"] = int(block)
                            database.record_match(
                                run_id=database_run_id,
                                round_id=round_id,
                                phase=f"direct_population_training_{block}",
                                model_a_id=candidate_id,
                                model_b_id=population_ids[opponent_index],
                                game=row,
                            )

                        direct_population_summary = (
                            train_direct_population_curriculum(
                                replace(
                                    round_training,
                                    seed=direct_population_training_seed,
                                    max_moves=direct_gate_max_moves,
                                ),
                                model_config,
                                candidate,
                                champion,
                                frozen_models,
                                probabilities,
                                num_envs=direct_population_envs,
                                minimum_steps=direct_population_steps,
                                maximum_steps=direct_population_max_steps,
                                block_steps=direct_population_block_steps,
                                num_threads=num_threads,
                                max_actions=mcts.native_max_actions,
                                gate_games=direct_population_gate_games,
                                gate_seed=direct_population_gate_seed,
                                gate_score=direct_population_gate_score,
                                historical_score_floor=(
                                    direct_population_historical_floor
                                ),
                                readiness_city_captures_per_game=(
                                    direct_readiness_city_captures_per_game
                                ),
                                readiness_completion_rate=(
                                    direct_readiness_completion_rate
                                ),
                                readiness_patience=(
                                    direct_population_readiness_patience
                                ),
                                policy_anchor_weight=(
                                    direct_policy_anchor_weight
                                ),
                                value_anchor_weight=(
                                    direct_value_anchor_weight
                                ),
                                opponent_policy_weight=(
                                    direct_population_opponent_policy_weight
                                ),
                                teacher_weight=(
                                    direct_population_teacher_weight
                                ),
                                shaping_weight=(
                                    direct_population_shaping_weight
                                ),
                                gate_game_callback=(
                                    record_direct_population_gate_game
                                ),
                                training_game_callback=(
                                    record_direct_population_training_game
                                ),
                                run=run.child(
                                    f"round_{round_index:03d}/"
                                    "direct_population"
                                ),
                            )
                        )
                    round_training_summary["direct_population"] = (
                        direct_population_summary
                    )
                    direct_population_ready = (
                        not bool(direct_population_steps)
                        or bool(
                            direct_population_summary.get(
                                "direct_population_ready", 0.0
                            )
                        )
                    )
                    if not direct_population_ready:
                        best_population_score = float(
                            direct_population_summary.get(
                                "direct_population_best_score", 0.0
                            )
                        )
                        best_population_worst = float(
                            direct_population_summary.get(
                                "direct_population_best_worst_score", 0.0
                            )
                        )
                        retain_population_challenger = (
                            best_population_score >= 0.50
                            and best_population_worst
                            >= direct_population_historical_floor
                        )
                        if retain_population_challenger:
                            retained_path = _next_checkpoint_path(
                                checkpoint_dir,
                                f"{checkpoint_prefix}-challenger",
                            )
                            save_model(retained_path, candidate, model_config)
                            database.retain_candidate(candidate_id, retained_path)
                            challenger_path = retained_path
                            challenger_id = candidate_id
                        else:
                            database.finalise_candidate(
                                candidate_id, promoted_path=None
                            )
                            challenger_path = None
                            challenger_id = None
                        round_training_summary["pool"] = {}
                        round_training_summary["population"] = {
                            "mcts_skipped_not_ready": 1.0,
                            "mcts_ladder": [],
                        }
                        last_training = dict(round_training_summary)
                        last_candidate_score = 0.0
                        last_population_score = 0.0
                        population_gate_summary = {
                            "promoted": 0.0,
                            "valid": 1.0,
                            "direct_ready": 1.0,
                            "direct_population_ready": 0.0,
                            "mcts_started": 0.0,
                            "retained_challenger": float(
                                retain_population_challenger
                            ),
                            "direct_population_best_score": (
                                best_population_score
                            ),
                            "direct_population_best_worst_score": (
                                best_population_worst
                            ),
                            "reason": (
                                "direct_population_not_ready_for_mcts"
                            ),
                        }
                        database.finish_round(
                            round_id,
                            champion_id,
                            False,
                            round_training_summary,
                            population_gate_summary,
                        )
                        run.log(
                            round=round_index,
                            direct_ready=1.0,
                            direct_population_ready=0.0,
                            mcts_started=0.0,
                            retained_challenger=float(
                                retain_population_challenger
                            ),
                            promotions=promotions,
                        )
                        database.flush()
                        print(
                            "league direct population not ready | "
                            "MCTS skipped | "
                            f"score={best_population_score:.4f} | "
                            f"worst={best_population_worst:.4f}",
                            flush=True,
                        )
                        del candidate, champion, frozen_models
                        continue

                    direct_population_state = model_state_on_cpu(candidate)
                    pool_summary: Dict[str, Any] = {}
                    if steps:
                        # This legacy same-network phase is now optional. It is
                        # disabled in the recommended curriculum because weak
                        # candidate-vs-candidate MCTS was the main regression
                        # source observed in long league runs.
                        pool_summary = train_self_play_pool(
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
                            model_out=None,
                            model=candidate,
                            anchor_model=champion,
                            anchor_policy_weight=champion_anchor_weight,
                            anchor_value_weight=champion_value_anchor_weight,
                            reserved_seeds=evaluation_map_seeds,
                            run=run.child(
                                f"round_{round_index:03d}/pool_self_play"
                            ),
                        )
                        # Any failed ladder rung still returns to the raw
                        # population checkpoint captured before this phase.
                        pool_summary["rollback_state_available"] = 1.0
                    round_training_summary["pool"] = pool_summary

                    population_summary: Dict[str, Any] = {
                        "mcts_ladder": [],
                        "mcts_ladder_accepted": 0.0,
                        "mcts_ladder_rollbacks": 0.0,
                    }
                    # A failed first MCTS rung must return to the last state
                    # proven safe against the raw-policy population gate, not
                    # to an optional ungated legacy self-play snapshot.
                    best_mcts_state = direct_population_state
                    best_mcts_score = 0.50
                    if direct_population_summary:
                        raw_population_baseline = {
                            "model_a_score": float(
                                direct_population_summary.get(
                                    "direct_population_best_score", 0.0
                                )
                            ),
                            "worst_score": float(
                                direct_population_summary.get(
                                    "direct_population_best_worst_score", 0.0
                                )
                            ),
                            "model_a_city_captures_per_game": float(
                                direct_population_summary.get(
                                    "direct_population_best_city_captures_per_game",
                                    0.0,
                                )
                            ),
                            "completion_rate": float(
                                direct_population_summary.get(
                                    "direct_population_best_completion_rate", 0.0
                                )
                            ),
                        }
                    else:
                        def record_raw_population_baseline_game(
                            opponent_index: int,
                            game: Mapping[str, Any],
                        ) -> None:
                            database.record_match(
                                run_id=database_run_id,
                                round_id=round_id,
                                phase="mcts_absorption_baseline",
                                model_a_id=candidate_id,
                                model_b_id=population_ids[opponent_index],
                                game=game,
                            )

                        raw_population_baseline = (
                            evaluate_direct_population_matches(
                                replace(
                                    training,
                                    max_moves=direct_gate_max_moves,
                                ),
                                candidate,
                                frozen_models,
                                probabilities,
                                games=direct_population_gate_games,
                                gate_seed=direct_population_gate_seed,
                                game_callback=(
                                    record_raw_population_baseline_game
                                ),
                            )
                        )
                    population_summary["raw_population_baseline"] = (
                        raw_population_baseline
                    )
                    best_raw_policy_score = float(
                        raw_population_baseline["model_a_score"]
                    )
                    best_raw_worst_score = float(
                        raw_population_baseline["worst_score"]
                    )
                    best_raw_city_capture_rate = float(
                        raw_population_baseline[
                            "model_a_city_captures_per_game"
                        ]
                    )
                    best_raw_completion_rate = float(
                        raw_population_baseline["completion_rate"]
                    )
                    if population_games:
                        for rung_index, (simulations, rung_games) in enumerate(
                            zip(population_rungs, population_game_allocations)
                        ):
                            if rung_games <= 0:
                                continue
                            rung_training = replace(
                                round_training,
                                seed=_offset_seed(
                                    population_training_seed,
                                    rung_index * 2_000_003,
                                ),
                            )
                            rung_mcts = replace(
                                mcts,
                                simulations=int(simulations),
                                # The direct curriculum must first prove that
                                # the neural policy/value help search. Manual
                                # priors are therefore reduced rung by rung.
                                strategic_prior_weight=min(
                                    mcts.strategic_prior_weight,
                                    0.50 if rung_index == 0 else 0.20,
                                ),
                                teacher_prior_weight=min(
                                    mcts.teacher_prior_weight,
                                    0.10,
                                ),
                            )

                            def record_training_game(
                                game: Mapping[str, Any],
                                *,
                                selected_rung: int = rung_index,
                                selected_simulations: int = int(simulations),
                            ) -> None:
                                opponent_index = game.get("opponent_model_index")
                                opponent_id = (
                                    population_ids[int(opponent_index)]
                                    if opponent_index is not None else None
                                )
                                row = dict(game)
                                row["mcts_rung"] = selected_rung
                                row["mcts_simulations"] = selected_simulations
                                database.record_match(
                                    run_id=database_run_id,
                                    round_id=round_id,
                                    phase=f"population_training_rung_{selected_rung + 1}",
                                    model_a_id=candidate_id,
                                    model_b_id=opponent_id,
                                    game=row,
                                )

                            rung_summary = train_population_pool(
                                rung_training,
                                model_config,
                                rung_mcts,
                                games=rung_games,
                                updates_per_game=population_updates_per_game,
                                candidate=candidate,
                                opponent_models=frozen_models,
                                opponent_weights=probabilities,
                                anchor_model=champion,
                                anchor_policy_weight=champion_anchor_weight,
                                anchor_value_weight=champion_value_anchor_weight,
                                num_threads=population_workers,
                                max_pending_leaves=max_pending_leaves,
                                visible_action_history=visible_action_history,
                                belief_retries=belief_retries,
                                max_envs=num_envs,
                                game_callback=record_training_game,
                                reserved_seeds=evaluation_map_seeds,
                                run=run.child(
                                    f"round_{round_index:03d}/mcts_rung_{rung_index + 1:02d}"
                                ),
                            )
                            gate_mcts = replace(
                                evaluation_mcts,
                                simulations=int(simulations),
                            )

                            def record_rung_gate_game(
                                game: Mapping[str, Any],
                                *,
                                selected_rung: int = rung_index,
                                selected_simulations: int = int(simulations),
                            ) -> None:
                                row = dict(game)
                                row["mcts_rung"] = selected_rung
                                row["mcts_simulations"] = selected_simulations
                                database.record_match(
                                    run_id=database_run_id,
                                    round_id=round_id,
                                    phase=(
                                        f"mcts_curriculum_gate_rung_"
                                        f"{selected_rung + 1}"
                                    ),
                                    model_a_id=candidate_id,
                                    model_b_id=champion_id,
                                    game=row,
                                )

                            rung_gate = evaluate_matches_pool(
                                candidate,
                                champion,
                                replace(
                                    training,
                                    seed=mcts_gate_seeds[rung_index],
                                ),
                                gate_mcts,
                                mcts_gate_games,
                                num_threads=num_threads,
                                max_pending_leaves=max_pending_leaves,
                                visible_action_history=visible_action_history,
                                belief_retries=belief_retries,
                                max_envs=num_envs,
                                game_callback=record_rung_gate_game,
                            )
                            rung_score = float(rung_gate.get("model_a_score", 0.0))

                            def record_absorption_gate_game(
                                opponent_index: int,
                                game: Mapping[str, Any],
                                *,
                                selected_rung: int = rung_index,
                                selected_simulations: int = int(simulations),
                            ) -> None:
                                row = dict(game)
                                row["mcts_rung"] = selected_rung
                                row["mcts_simulations"] = selected_simulations
                                database.record_match(
                                    run_id=database_run_id,
                                    round_id=round_id,
                                    phase=(
                                        f"mcts_absorption_gate_rung_"
                                        f"{selected_rung + 1}"
                                    ),
                                    model_a_id=candidate_id,
                                    model_b_id=population_ids[opponent_index],
                                    game=row,
                                )

                            absorption_gate = evaluate_direct_population_matches(
                                replace(
                                    training,
                                    max_moves=direct_gate_max_moves,
                                ),
                                candidate,
                                frozen_models,
                                probabilities,
                                games=direct_population_gate_games,
                                gate_seed=direct_population_gate_seed,
                                game_callback=record_absorption_gate_game,
                            )
                            raw_policy_score = float(
                                absorption_gate.get("model_a_score", 0.0)
                            )
                            raw_worst_score = float(
                                absorption_gate.get("worst_score", 0.0)
                            )
                            raw_city_capture_rate = float(
                                absorption_gate.get(
                                    "model_a_city_captures_per_game", 0.0
                                )
                            )
                            raw_completion_rate = float(
                                absorption_gate.get("completion_rate", 0.0)
                            )
                            search_ready = bool(
                                rung_gate.get("valid", 1.0)
                            ) and (
                                rung_score
                                >= max(float(mcts_gate_score), best_mcts_score)
                            )
                            network_absorbed_search = (
                                raw_policy_score >= best_raw_policy_score
                                and raw_worst_score >= best_raw_worst_score
                                and raw_city_capture_rate
                                >= best_raw_city_capture_rate
                                and raw_completion_rate
                                >= best_raw_completion_rate
                            )
                            accepted = search_ready and network_absorbed_search
                            if accepted:
                                best_mcts_state = model_state_on_cpu(candidate)
                                best_mcts_score = rung_score
                                best_raw_policy_score = raw_policy_score
                                best_raw_worst_score = raw_worst_score
                                best_raw_city_capture_rate = (
                                    raw_city_capture_rate
                                )
                                best_raw_completion_rate = raw_completion_rate
                                population_summary["mcts_ladder_accepted"] += 1.0
                            else:
                                restore_model_state(candidate, best_mcts_state)
                                population_summary["mcts_ladder_rollbacks"] += 1.0
                            rung_record = {
                                "simulations": float(simulations),
                                "games": float(rung_games),
                                "accepted": float(accepted),
                                "gate_score": rung_score,
                                "best_gate_score": best_mcts_score,
                                "gate_lower_95": float(
                                    rung_gate.get("score_lower_95", 0.0)
                                ),
                                "search_ready": float(search_ready),
                                "network_absorbed_search": float(
                                    network_absorbed_search
                                ),
                                "raw_policy_score": raw_policy_score,
                                "raw_worst_score": raw_worst_score,
                                "raw_city_captures_per_game": (
                                    raw_city_capture_rate
                                ),
                                "raw_completion_rate": raw_completion_rate,
                                "training": rung_summary,
                                "gate": rung_gate,
                                "absorption_gate": absorption_gate,
                            }
                            population_summary["mcts_ladder"].append(rung_record)
                            run.log(
                                round=round_index,
                                mcts_rung=rung_index + 1,
                                mcts_simulations=simulations,
                                mcts_rung_score=rung_score,
                                mcts_raw_policy_score=raw_policy_score,
                                mcts_network_absorbed_search=float(
                                    network_absorbed_search
                                ),
                                mcts_rung_accepted=float(accepted),
                            )
                            print(
                                f"league MCTS rung {simulations} | "
                                f"search={rung_score:.4f} | "
                                f"raw={raw_policy_score:.4f} | "
                                f"absorbed={int(network_absorbed_search)} | "
                                f"accepted={int(accepted)}",
                                flush=True,
                            )
                            if not accepted:
                                break
                    expected_mcts_rungs = sum(
                        int(rung_games > 0)
                        for rung_games in population_game_allocations
                    )
                    completed_mcts_curriculum = (
                        not bool(population_games)
                        or int(population_summary["mcts_ladder_accepted"])
                        == expected_mcts_rungs
                    )
                    population_summary["mcts_ladder_required"] = float(
                        expected_mcts_rungs
                    )
                    population_summary["mcts_ready_for_promotion_gate"] = float(
                        completed_mcts_curriculum
                    )
                    round_training_summary["population"] = population_summary
                    database.flush()
                    last_training = dict(round_training_summary)

                    if not completed_mcts_curriculum:
                        retain_mcts_challenger = (
                            best_raw_policy_score >= 0.50
                            and best_raw_worst_score
                            >= direct_population_historical_floor
                            and best_raw_city_capture_rate
                            >= direct_readiness_city_captures_per_game
                            and best_raw_completion_rate
                            >= direct_readiness_completion_rate
                        )
                        if retain_mcts_challenger:
                            retained_path = _next_checkpoint_path(
                                checkpoint_dir,
                                f"{checkpoint_prefix}-challenger",
                            )
                            save_model(retained_path, candidate, model_config)
                            database.retain_candidate(candidate_id, retained_path)
                            challenger_path = retained_path
                            challenger_id = candidate_id
                        else:
                            database.finalise_candidate(
                                candidate_id, promoted_path=None
                            )
                            challenger_path = None
                            challenger_id = None
                        last_candidate_score = 0.0
                        last_population_score = 0.0
                        mcts_gate_summary = {
                            "promoted": 0.0,
                            "valid": 1.0,
                            "direct_ready": 1.0,
                            "direct_population_ready": 1.0,
                            "mcts_started": 1.0,
                            "mcts_ready_for_promotion_gate": 0.0,
                            "mcts_ladder_accepted": float(
                                population_summary["mcts_ladder_accepted"]
                            ),
                            "mcts_ladder_required": float(
                                expected_mcts_rungs
                            ),
                            "retained_challenger": float(
                                retain_mcts_challenger
                            ),
                            "reason": "mcts_curriculum_incomplete",
                        }
                        database.finish_round(
                            round_id,
                            champion_id,
                            False,
                            round_training_summary,
                            mcts_gate_summary,
                        )
                        run.log(
                            round=round_index,
                            mcts_ready_for_promotion_gate=0.0,
                            retained_challenger=float(
                                retain_mcts_challenger
                            ),
                            promoted=0.0,
                            promotions=promotions,
                        )
                        database.flush()
                        print(
                            "league MCTS curriculum incomplete | final gate "
                            "skipped | accepted="
                            f"{int(population_summary['mcts_ladder_accepted'])}/"
                            f"{expected_mcts_rungs}",
                            flush=True,
                        )
                        del candidate, champion, frozen_models
                        continue

                    evaluation_results: list[Dict[str, Any]] = []
                    invalid_evaluation = False
                    for opponent_index, (
                        opponent_id,
                        opponent_path,
                        opponent_model,
                        probability,
                    ) in enumerate(zip(
                        population_ids, population_paths, frozen_models, probabilities
                    )):
                        is_champion = opponent_id == champion_id
                        games = promotion_games if is_champion else historical_games
                        purpose = "champion_gate" if is_champion else "historical_gate"
                        match_seed = _offset_seed(
                            round_evaluation_seed,
                            opponent_index * 1_000_003,
                        )
                        evaluation_id = database.start_evaluation(
                            round_id,
                            candidate_id,
                            opponent_id,
                            purpose,
                            match_seed,
                            games,
                        )

                        def record_evaluation_game(
                            game: Mapping[str, Any],
                            *,
                            selected_evaluation_id: int = evaluation_id,
                            selected_opponent_id: int = opponent_id,
                        ) -> None:
                            database.record_match(
                                run_id=database_run_id,
                                round_id=round_id,
                                phase="evaluation",
                                model_a_id=candidate_id,
                                model_b_id=selected_opponent_id,
                                game=game,
                                evaluation_id=selected_evaluation_id,
                            )

                        match = evaluate_matches_pool(
                            candidate,
                            opponent_model,
                            replace(training, seed=match_seed),
                            evaluation_mcts,
                            games,
                            num_threads=num_threads,
                            max_pending_leaves=max_pending_leaves,
                            visible_action_history=visible_action_history,
                            belief_retries=belief_retries,
                            max_envs=num_envs,
                            game_callback=record_evaluation_game,
                        )
                        result = {
                            "opponent_id": opponent_id,
                            "opponent_path": str(opponent_path),
                            "sampling_probability": probability,
                            "purpose": purpose,
                            **match,
                        }
                        evaluation_results.append(result)
                        if not bool(match.get("valid", 1.0)):
                            invalid_evaluation = True
                            database.invalidate_evaluation(evaluation_id, match)
                            run.log(
                                round=round_index,
                                invalid_evaluations=1,
                                invalid_evaluation_envs=match.get("invalid_envs", 0),
                                promoted=0,
                            )
                            print(
                                f"league gate vs {opponent_path.stem} invalid | "
                                f"candidate rejected | reason={match.get('invalid_reason', 'belief recovery exhausted')}",
                                flush=True,
                            )
                            break
                        database.finish_evaluation(evaluation_id, match, elo_k=elo_k)
                        print(
                            f"league gate vs {opponent_path.stem} | "
                            f"score={float(match['model_a_score']):.4f} | "
                            f"lower95={float(match['score_lower_95']):.4f}",
                            flush=True,
                        )

                    if invalid_evaluation:
                        last_candidate_score = 0.0
                        last_population_score = 0.0
                        minimum_historical_score = 0.0
                        champion_lower_95 = 0.0
                        promoted = False
                    else:
                        champion_result = next(
                            result for result in evaluation_results
                            if int(result["opponent_id"]) == champion_id
                        )
                        last_candidate_score = float(champion_result["model_a_score"])
                        last_population_score = sum(
                            float(result["sampling_probability"]) * float(result["model_a_score"])
                            for result in evaluation_results
                        )
                        minimum_historical_score = min(
                            float(result["model_a_score"]) for result in evaluation_results
                        )
                        champion_lower_95 = float(champion_result["score_lower_95"])
                        promoted = (
                            last_candidate_score >= promotion_score
                            and champion_lower_95 >= promotion_lower_bound
                            and last_population_score >= population_score
                            and minimum_historical_score >= historical_score_floor
                        )
                    gate_summary = {
                        "promoted": float(promoted),
                        "valid": float(not invalid_evaluation),
                        "invalid_evaluations": float(invalid_evaluation),
                        "candidate_score": last_candidate_score,
                        "champion_lower_95": champion_lower_95,
                        "population_score": last_population_score,
                        "minimum_historical_score": minimum_historical_score,
                        "evaluations": evaluation_results,
                    }
                    print(
                        f"league decision {round_index}/{rounds} | "
                        f"champion={last_candidate_score:.4f} | "
                        f"population={last_population_score:.4f} | "
                        f"worst={minimum_historical_score:.4f} | promoted={int(promoted)}",
                        flush=True,
                    )

                    if promoted:
                        promotions += 1
                        promoted_path = _next_checkpoint_path(checkpoint_dir, checkpoint_prefix)
                        save_model(promoted_path, candidate, model_config)
                        database.finalise_candidate(candidate_id, promoted_path=promoted_path)
                        champion_path = promoted_path
                        champion_id = candidate_id
                        challenger_path = None
                        challenger_id = None
                        print(
                            f"league promotion {promotions}: saved {champion_path}",
                            flush=True,
                        )
                    else:
                        # A statistically inconclusive 0.50-0.55 challenger is
                        # useful learning progress, not a reason to restart
                        # from the teacher. Keep it only when every completed
                        # held-out gate is non-regressing; the protected
                        # champion and league population remain unchanged.
                        retain_challenger = (
                            not invalid_evaluation
                            and last_candidate_score >= 0.50
                            and last_population_score >= 0.50
                            and minimum_historical_score
                            >= historical_score_floor
                        )
                        if retain_challenger:
                            retained_path = _next_checkpoint_path(
                                checkpoint_dir,
                                f"{checkpoint_prefix}-challenger",
                            )
                            save_model(retained_path, candidate, model_config)
                            database.retain_candidate(candidate_id, retained_path)
                            challenger_path = retained_path
                            challenger_id = candidate_id
                            print(
                                "league challenger retained for the next "
                                f"round: {challenger_path}",
                                flush=True,
                            )
                        else:
                            database.finalise_candidate(
                                candidate_id, promoted_path=None
                            )
                        gate_summary["retained_challenger"] = float(
                            retain_challenger
                        )

                    round_summary = {
                        "direct": direct_summary,
                        "direct_population": direct_population_summary,
                        "pool": pool_summary,
                        "population": population_summary,
                    }
                    database.finish_round(
                        round_id,
                        champion_id,
                        promoted,
                        round_summary,
                        gate_summary,
                    )
                    run.log(
                        round=round_index,
                        candidate_score=last_candidate_score,
                        population_score=last_population_score,
                        minimum_historical_score=minimum_historical_score,
                        invalid_evaluations=float(invalid_evaluation),
                        promoted=float(promoted),
                        promotions=promotions,
                    )
                    database.flush()
                    del candidate, champion, frozen_models
                except (BeliefQualityError, BeliefRecoveryRequired) as error:
                    if candidate_id is not None:
                        database.finalise_candidate(candidate_id, promoted_path=None)
                    diagnostics = (
                        error.metrics
                        if isinstance(error, BeliefQualityError)
                        else {
                            "belief_recovery_env_ids": error.env_ids.tolist(),
                            "belief_recovery_diagnostics": list(error.diagnostics),
                        }
                    )
                    rejection_summary = {
                        "promoted": 0.0,
                        "valid": 0.0,
                        "belief_quality_failed": 1.0,
                        "error": str(error),
                        "diagnostics": diagnostics,
                    }
                    database.finish_round(
                        round_id,
                        champion_id,
                        False,
                        round_training_summary,
                        rejection_summary,
                    )
                    run.log(
                        round=round_index,
                        promoted=0.0,
                        belief_quality_failed=1.0,
                        promotions=promotions,
                    )
                    database.flush()
                    print(
                        f"league round {round_index} rejected by belief circuit breaker: {error}",
                        flush=True,
                    )
                except BaseException as error:
                    if candidate_id is not None:
                        database.finalise_candidate(candidate_id, promoted_path=None)
                    database.fail_round(round_id, error)
                    raise
                finally:
                    current_round_id[0] = None

            summary: Dict[str, float | str] = {
                "league_rounds": float(rounds),
                "league_promotions": float(promotions),
                "league_last_candidate_score": last_candidate_score,
                "league_last_population_score": last_population_score,
                "league_promotion_threshold": promotion_score,
                "league_population_threshold": population_score,
                "league_historical_score_floor": historical_score_floor,
                "league_champion_path": str(champion_path),
                "league_challenger_path": (
                    str(challenger_path) if challenger_path is not None else ""
                ),
                "league_database_path": str(database_path.resolve()),
                "league_population_size": float(population_size),
                "league_champion_anchor_weight": float(champion_anchor_weight),
                "league_champion_value_anchor_weight": float(
                    champion_value_anchor_weight
                ),
                "league_evaluation_seed": float(resolved_evaluation_seed),
                "league_evaluation_seed_stride": float(evaluation_seed_stride),
            }
            if last_training:
                summary["league_last_pool_updates"] = float(
                    last_training.get("pool", {}).get(
                        "self_play_pool_optimisation_updates", 0.0
                    )
                )
                population_rung_rows = last_training.get("population", {}).get(
                    "mcts_ladder", []
                )
                summary["league_last_population_games"] = float(
                    sum(
                        float(row.get("training", {}).get("population_pool_games", 0.0))
                        for row in population_rung_rows
                    )
                )
                summary["league_last_mcts_rungs_accepted"] = float(
                    last_training.get("population", {}).get(
                        "mcts_ladder_accepted", 0.0
                    )
                )
                summary["league_last_mcts_rung_rollbacks"] = float(
                    last_training.get("population", {}).get(
                        "mcts_ladder_rollbacks", 0.0
                    )
                )
                summary["league_last_direct_best_score"] = float(
                    last_training.get("direct", {}).get("direct_best_score", 0.0)
                )
                summary["league_last_direct_ready"] = float(
                    last_training.get("direct", {}).get("direct_ready", 0.0)
                )
                summary["league_last_direct_population_steps"] = float(
                    last_training.get("direct_population", {}).get(
                        "direct_population_steps", 0.0
                    )
                )
                summary["league_last_direct_population_ready"] = float(
                    last_training.get("direct_population", {}).get(
                        "direct_population_ready", 0.0
                    )
                )
                summary["league_last_direct_population_score"] = float(
                    last_training.get("direct_population", {}).get(
                        "direct_population_best_score", 0.0
                    )
                )
                summary["league_last_direct_population_worst"] = float(
                    last_training.get("direct_population", {}).get(
                        "direct_population_best_worst_score", 0.0
                    )
                )
                summary["league_last_mcts_started"] = float(
                    bool(population_rung_rows)
                )
                summary["league_last_pool_anchor_mean_l1"] = float(
                    last_training.get("pool", {}).get(
                        "self_play_pool_anchor_mean_l1", 0.0
                    )
                )
                summary["league_last_population_anchor_mean_l1"] = float(
                    population_rung_rows[-1]
                    .get("training", {})
                    .get("population_pool_anchor_mean_l1", 0.0)
                    if population_rung_rows
                    else 0.0
                )
            run_directory = run.finish(summary)
            summary["run_dir"] = str(run_directory)
            database.finish_run(database_run_id, "complete", summary)
            return summary
        except BaseException as error:
            database.finish_run(
                database_run_id,
                "failed",
                {"error": repr(error), "champion_path": str(champion_path)},
            )
            raise
