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
from .routed_self_play_pool import (
    evaluate_matches_pool,
    require_routed_self_play_pool,
    train_population_pool,
)
from .run_artifacts import TrainingRun
from .self_play_pool_train import train_self_play_pool


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
    # Check the multi-agent binding before spending hours on the first
    # same-network rollout phase.
    require_routed_self_play_pool()
    if not champion_in.is_file():
        raise FileNotFoundError(f"champion checkpoint does not exist: {champion_in}")
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
        "elo_k": elo_k,
        "checkpoint_prefix": checkpoint_prefix,
        "num_envs": num_envs,
        "steps": steps,
        "updates": updates,
        "num_threads": num_threads,
        "max_pending_leaves": max_pending_leaves,
        "visible_action_history": visible_action_history,
        "belief_retries": belief_retries,
    }
    run.write_config(**league_config)

    champion_path = champion_in
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
                    candidate, model_config = load_model(champion_path, device)
                    champion, champion_config = load_model(champion_path, device)
                    if champion_config != model_config:
                        raise RuntimeError("candidate and champion ModelConfig differ")
                    candidate_id = database.create_candidate(
                        database_run_id,
                        champion_id,
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
                    known_training_map_seeds = {
                        population_training_seed + game_index
                        for game_index in range(population_games)
                    }
                    if evaluation_map_seeds & known_training_map_seeds:
                        raise ValueError(
                            "population training maps overlap held-out evaluation maps"
                        )
                    if round_training.seed in evaluation_map_seeds:
                        raise ValueError(
                            "SelfPlayPool collection seed overlaps held-out evaluation maps"
                        )
                    print(
                        "league population | "
                        + ", ".join(
                            f"{path.stem}: {probability:.1%}"
                            for path, probability in zip(population_paths, probabilities)
                        ),
                        flush=True,
                    )

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
                        # Shallow MCTS can be much noisier than the teacher on
                        # wide roots. Keep a frozen champion contribution in
                        # every policy target to prevent catastrophic drift.
                        anchor_model=champion,
                        anchor_policy_weight=champion_anchor_weight,
                        reserved_seeds=evaluation_map_seeds,
                        run=run.child(f"round_{round_index:03d}/pool_self_play"),
                    )
                    round_training_summary["pool"] = pool_summary

                    frozen_models = []
                    for population_path in population_paths:
                        if population_path.resolve() == champion_path.resolve():
                            frozen_model, frozen_config = champion, champion_config
                        else:
                            frozen_model, frozen_config = load_model(population_path, device)
                        if frozen_config != model_config:
                            raise RuntimeError(
                                f"population checkpoint has a different ModelConfig: {population_path}"
                            )
                        frozen_models.append(frozen_model.eval())

                    population_summary: Dict[str, Any] = {}
                    if population_games:
                        population_training = replace(
                            round_training,
                            seed=population_training_seed,
                        )

                        def record_training_game(game: Mapping[str, Any]) -> None:
                            opponent_index = game.get("opponent_model_index")
                            opponent_id = (
                                population_ids[int(opponent_index)]
                                if opponent_index is not None else None
                            )
                            database.record_match(
                                run_id=database_run_id,
                                round_id=round_id,
                                phase="population_training",
                                model_a_id=candidate_id,
                                model_b_id=opponent_id,
                                game=game,
                            )

                        population_summary = train_population_pool(
                            population_training,
                            model_config,
                            mcts,
                            games=population_games,
                            updates_per_game=population_updates_per_game,
                            candidate=candidate,
                            opponent_models=frozen_models,
                            opponent_weights=probabilities,
                            anchor_model=champion,
                            anchor_policy_weight=champion_anchor_weight,
                            num_threads=population_workers,
                            max_pending_leaves=max_pending_leaves,
                            visible_action_history=visible_action_history,
                            belief_retries=belief_retries,
                            max_envs=num_envs,
                            game_callback=record_training_game,
                            reserved_seeds=evaluation_map_seeds,
                            run=run.child(f"round_{round_index:03d}/population_training"),
                        )
                    round_training_summary["population"] = population_summary
                    database.flush()
                    last_training = dict(round_training_summary)

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
                        print(
                            f"league promotion {promotions}: saved {champion_path}",
                            flush=True,
                        )
                    else:
                        database.finalise_candidate(candidate_id, promoted_path=None)

                    round_summary = {"pool": pool_summary, "population": population_summary}
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
                "league_database_path": str(database_path.resolve()),
                "league_population_size": float(population_size),
                "league_champion_anchor_weight": float(champion_anchor_weight),
                "league_evaluation_seed": float(resolved_evaluation_seed),
                "league_evaluation_seed_stride": float(evaluation_seed_stride),
            }
            if last_training:
                summary["league_last_pool_updates"] = float(
                    last_training.get("pool", {}).get(
                        "self_play_pool_optimisation_updates", 0.0
                    )
                )
                summary["league_last_population_games"] = float(
                    last_training.get("population", {}).get("population_pool_games", 0.0)
                )
                summary["league_last_pool_anchor_mean_l1"] = float(
                    last_training.get("pool", {}).get(
                        "self_play_pool_anchor_mean_l1", 0.0
                    )
                )
                summary["league_last_population_anchor_mean_l1"] = float(
                    last_training.get("population", {}).get(
                        "population_pool_anchor_mean_l1", 0.0
                    )
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
