from __future__ import annotations

from collections import deque
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
import runpy
import sqlite3
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

from polybot.config import MCTSConfig, ModelConfig, TrainingConfig
from polybot.evaluation.matches import evaluate_direct_matches
from polybot.models.policy_value_belief import PolicyValueBeliefNet
from polybot.infrastructure.model_io import save_model
from polybot.training.league import train_promotion_league
from polybot.training.flat_experiments import (
    FlatImprovements, model_fingerprint, paired_regression,
)
from polybot.training.flat_value import (
    ValuePosition, ValueValidationCollector, fit_value_head,
)
from polybot.training.run_artifacts import TrainingRun
from polybot.training.vector_population import (
    _ppo_snapshot, evaluate_direct_population_matches,
    train_direct_population_curriculum,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model(architecture="factorized"):
    return PolicyValueBeliefNet(ModelConfig(
        d_model=16, transformer_layers=1, attention_heads=2, dropout=0.0,
        policy_architecture=architecture,
    ))


def packet(turn=0):
    actions = np.zeros((1, 2, 17), dtype=np.int32)
    actions[:, :, 0] = 3
    state = np.zeros((1, 11), dtype=np.int32)
    state[:, 0] = turn
    return {
        "map_tokens": np.zeros((1, 121, 23), dtype=np.int32),
        "state": state,
        "action_features": actions,
        "action_mask": np.ones((1, 2), dtype=np.uint8),
        "visible_event_features": np.zeros((1, 0, 26), dtype=np.int32),
        "visible_event_mask": np.zeros((1, 0), dtype=np.uint8),
    }


def value_positions(target):
    return [ValuePosition(_ppo_snapshot(packet(turn), 0, 0)[0], target)
            for turn in (0, 5, 10, 20, 30, 40, 60, 80)]


def paired_rows(scores, start=100):
    return [{"map_seed": start + index, "model_a_player": side, "score_a": score}
            for index, score in enumerate(scores) for side in (0, 1)]


def test_fingerprint_includes_scalar_buffers_and_updated_weights():
    model = torch.nn.Linear(2, 1)
    model.register_buffer("scalar", torch.tensor(0.5))
    initial = model_fingerprint(model)
    assert initial == model_fingerprint(deepcopy(model))
    model.scalar.add_(0.1)
    assert initial != model_fingerprint(model)


def test_paired_rollback_distinguishes_regression_from_uncertain_difference():
    reference = paired_rows([0.5] * 40)
    strong = paired_regression(paired_rows([0.0] * 40), reference)
    assert strong["confirmed"] and strong["upper95"] < -0.02
    uncertain = paired_regression(paired_rows([0.0, 1.0] * 20), reference)
    assert uncertain["valid"] and not uncertain["confirmed"]
    assert not paired_regression(paired_rows([0.0] * 2), paired_rows([1.0] * 2))["valid"]
    assert not paired_regression(paired_rows([0.0] * 40, 101), reference)["valid"]
    assert not paired_regression(paired_rows([0.0] * 40)[1:], reference)["valid"]


def test_evaluation_cache_tracks_weights_maps_inputs_and_preserves_rng():
    candidate, a, b = tiny_model(), tiny_model(), tiny_model()
    cache = {}
    observed = []
    def evaluation(*args, **kwargs):
        random.random()
        np.random.random()
        torch.rand(1)
        kwargs["game_callback"]({"map_seed": args[3].seed, "score_a": 0.5})
        return {"model_a_score": 0.25 if args[1] is a else 0.75,
                "model_a_city_captures_per_game": 0.0, "completion_rate": 1.0}
    def evaluate(weights=(0.5, 0.5), **changes):
        return evaluate_direct_population_matches(
            TrainingConfig(device="cpu"), candidate, [a, b], weights,
            games=2, gate_seed=changes.pop("gate_seed", 100),
            evaluation_cache=cache,
            game_callback=lambda index, row: observed.append((index, row)),
            **changes,
        )
    torch.manual_seed(71)
    before = torch.get_rng_state().clone()
    with patch("polybot.training.vector_population.evaluate_direct_matches", side_effect=evaluation) as mock:
        first = evaluate()
        second = evaluate((0.8, 0.2))
        assert mock.call_count == 2
        assert len(observed) == 4  # Cached games still reach the audit log.
        assert first["model_a_score"] == 0.5
        assert second["model_a_score"] == pytest.approx(0.35)
        assert torch.equal(before, torch.get_rng_state())
        with torch.no_grad():
            next(candidate.parameters()).add_(0.001)
        evaluate()
        assert mock.call_count == 4
        evaluate(visible_action_history=64)
        assert mock.call_count == 6
        evaluate(gate_seed=101)
        assert mock.call_count == 8


def test_validation_collector_copies_inputs_and_excludes_truncated_labels():
    collector = ValueValidationCollector(2, 16, 17)
    data = packet()
    collector.add(0, data, 0)
    data["map_tokens"].fill(10)
    collector.finish({"game_index": 0, "score_a": 1.0, "truncated": 0})
    assert collector.examples[0].return_target == 1.0
    assert not collector.examples[0].observation["map_tokens"].any()
    collector.add(1, data, 0)
    collector.finish({"game_index": 1, "score_a": 0.5, "truncated": 1})
    assert len(collector.examples) == 1
    assert collector.completed_games == collector.truncated_games == 1


@pytest.mark.parametrize("validation_target,kept", [(1.0, True), (-1.0, False)])
@pytest.mark.parametrize("architecture", ["factorized", "relational"])
def test_terminal_critic_updates_only_head_and_rolls_back_bad_validation(validation_target, kept, architecture):
    model = tiny_model(architecture)
    with torch.no_grad():
        for parameter in model.value_head.parameters():
            parameter.zero_()
    before = deepcopy(model.state_dict())
    settings = FlatImprovements(value_updates=8, value_batch_size=8, value_learning_rate=0.01)
    result = fit_value_head(
        model, TrainingConfig(device="cpu"), value_positions(1.0),
        value_positions(validation_target), settings, seed=19,
    )
    assert result["updates"] == 8 and result["kept"] is kept
    for name, tensor in model.state_dict().items():
        if not name.startswith("value_head.") or not kept:
            assert torch.equal(tensor, before[name]), name
    if kept:
        assert result["after"]["rmse"] < result["before"]["rmse"]
        assert result["after"]["opening"]["count"] == 3


def test_native_arena_position_callback_supplies_candidate_public_states():
    collector = ValueValidationCollector(2, 4, 13)
    model = tiny_model()
    rows = []
    result = evaluate_direct_matches(
        model, deepcopy(model), torch.device("cpu"),
        TrainingConfig(device="cpu", seed=287, max_moves=4), games=2,
        num_threads=1, max_actions=1024, visible_action_history=32,
        position_callback=collector.add,
        game_callback=lambda row: (rows.append(row), collector.finish(row)),
    )
    assert result["games"] == 2 and len(rows) == 2
    assert collector.seen and max(collector.seen) <= 1
    assert not collector.examples  # Move limit does not invent a terminal label.
    assert collector.truncated_games == 2


def test_scientific_state_roundtrip_restores_working_weights_and_adamw(tmp_path):
    candidate = tiny_model()
    champion = deepcopy(candidate)
    settings = FlatImprovements(value_updates=0, rollout_steps_final=0)
    seen_states = []
    def train(training, model_config, live, *args, **kwargs):
        seen_states.append(deepcopy(live.state_dict()))
        continuation = kwargs["continuation_state"]
        optimizer = continuation.setdefault("optimiser", torch.optim.AdamW(live.parameters(), lr=1e-5))
        continuation.setdefault("scaler", torch.amp.GradScaler("cuda", enabled=False))
        continuation.setdefault("rng", np.random.default_rng(19))
        optimizer.zero_grad(set_to_none=True)
        sum(p.sum() for p in live.parameters()).backward()
        optimizer.step()
        continuation["policy_version"] = continuation.get("policy_version", 0) + 1
        return {"steps": 1.0}
    def evaluate(*args, **kwargs):
        return {"model_a_score": 0.5, "worst_score": 0.5,
                "model_a_city_captures_per_game": 0.0, "completion_rate": 1.0,
                "opponents": [{"model_a_score": 0.5}]}
    def curriculum(name, resume=None):
        return train_direct_population_curriculum(
            TrainingConfig(device="cpu", seed=11), candidate.config, candidate,
            champion, [champion], [1.0], num_envs=2,
            minimum_steps=1, maximum_steps=1, block_steps=1,
            num_threads=1, max_actions=16, gate_games=2, gate_seed=127,
            visible_action_history=0, gate_score=0.5, historical_score_floor=0.4,
            readiness_city_captures_per_game=0.0, readiness_completion_rate=0.0,
            readiness_patience=1, policy_anchor_weight=0.0, value_anchor_weight=0.0,
            opponent_policy_weight=0.0, teacher_weight=0.0, shaping_weight=0.0,
            ema_enabled=False, flat_persistent_optimizer=True,
            flat_improvements=settings, flat_resume_state=resume,
            run=TrainingRun.create(tmp_path, name),
        )
    with patch("polybot.training.vector_population.train_vector_population", side_effect=train), \
         patch("polybot.training.vector_population.evaluate_direct_population_matches", side_effect=evaluate), \
         patch.object(TrainingRun, "_write_plots"):
        first = curriculum("first")
        file = Path(first["run_dir"]) / "latest_state.pt"
        saved = torch.load(file, weights_only=False)
        # A pre-relational optimizer checkpoint has no rank setting. That
        # missing default must not prevent continuation of the unchanged net.
        saved["model_config"].pop("action_interaction_dim")
        torch.save(saved, file)
        with torch.no_grad():
            next(candidate.parameters()).add_(10)
        second = curriculum("second", str(file))
        restored = seen_states[1]
        for name, tensor in saved["model_state"].items():
            assert torch.equal(restored[name], tensor)
        again = torch.load(Path(second["run_dir"]) / "latest_state.pt", weights_only=False)
        assert again["policy_version"] == saved["policy_version"] + 1
        assert max(float(s["step"]) for s in again["optimizer"]["state"].values()) == 2
        assert (Path(first["run_dir"]) / "source_snapshot.json").is_file()
        manifest = json.loads((Path(first["run_dir"]) / "experiment.json").read_text())
        assert manifest["source_sha256"] and manifest["champion"]
        with torch.no_grad():
            next(champion.parameters()).add_(0.1)
        with pytest.raises(ValueError, match="resume state does not match"):
            curriculum("mismatch", str(file))


def test_ablation_plan_uses_paired_seeds_and_rejects_optimizer_warmstart():
    path = Path(__file__).resolve().parents[1] / "scripts/flat_experiment_matrix.py"
    spec = importlib.util.spec_from_file_location("flat_matrix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = ["python", "scripts/train_flat.py", "--random-seed", "--seed", "5",
            "--champion-in", "models/flat.pt", "--flat-value-updates=90"]
    plan = module.build_plan(base, [11, 12], ["control", "full"], 100)
    assert len(plan) == 4
    for trial in plan:
        command = trial["command"]
        assert "--random-seed" not in command
        assert command[command.index("--evaluation-seed") + 1] == "100"
        assert command[command.index("--champion-in") + 1] == "models/flat.pt"
        assert command.count("--flat-value-updates") == 1
        assert command[command.index("--flat-value-updates") + 1] == ("0" if trial["variant"] == "control" else "8")
    with pytest.raises(ValueError, match="optimizer states"):
        module.build_plan(base + ["--flat-resume-state", "state.pt"], [11], ["full"], 100)


def test_flat_league_wires_main_critic_and_target_only_exploiter(tmp_path):
    model = tiny_model()
    champion = tmp_path / "champion.pt"
    save_model(champion, model, model.config)
    settings = FlatImprovements(scientific_artifacts=False, value_validation_games=2)
    calls = []
    def curriculum(*args, **kwargs):
        calls.append((args, kwargs))
        return {"direct_population_ready": 0.0, "direct_population_full_gate_selected": 1.0,
                "direct_population_full_gate_score": 0.5,
                "direct_population_best_score": 0.5}
    def evaluate(*args, **kwargs):
        return {"games": 2.0, "model_a_wins": 1.0, "model_a_losses": 1.0,
                "draws": 0.0, "model_a_score": 0.5, "score_lower_95": 0.0,
                "evaluated_map_seeds": [args[3].seed, args[3].seed]}
    with patch("polybot.training.league.train_direct_population_curriculum", side_effect=curriculum), \
         patch("polybot.training.league.evaluate_direct_matches", side_effect=evaluate), \
         patch.object(TrainingRun, "_write_plots"):
        train_promotion_league(
            TrainingConfig(device="cpu", seed=11), training_stage="flat",
            champion_in=champion, checkpoint_dir=tmp_path / "checkpoints",
            rounds=1, promotion_games=2, promotion_score=0.55, promotion_lower_bound=0.5,
            evaluation_seed=1_000, evaluation_seed_stride=101, round_seed_stride=103,
            checkpoint_prefix="flat", mcts=MCTSConfig(simulations=1, particles=1),
            num_envs=2, steps=0, updates=0, num_threads=1, max_pending_leaves=1,
            visible_action_history=0, belief_retries=1,
            database_path=tmp_path / "league.sqlite", champion_mode="flat",
            value_bootstrap_games=0, population_games=0, historical_games=2,
            direct_population_envs=2, direct_population_steps=1,
            direct_population_max_steps=1, direct_population_block_steps=1,
            direct_population_gate_games=2, direct_population_diagnostic_gate_games=2,
            direct_population_branch_count=1, mcts_probe_games=0,
            flat_improvements=settings, flat_population_regression_confirm_games=16,
            flat_population_policy_anchor_final_weight=0.05,
            breakthrough_exploiter=True, exploiter_policy_anchor_weight=0.0,
            run=TrainingRun.create(tmp_path, "league"),
        )
    assert len(calls) == 2
    main, exploiter = calls[0][1], calls[1][1]
    assert main["flat_improvements"] is settings and isinstance(main["flat_value_replay"], deque)
    assert exploiter["historical_score_floor"] == 0.0
    assert exploiter["flat_policy_anchor_final_weight"] == 0.0
    assert len(exploiter["benchmark_opponent_models"]) == 1
    assert exploiter["benchmark_opponent_ids"] == ["flat-current-main"]


@pytest.mark.parametrize("architecture", ["factorized", "relational"])
def test_train_flat_cli_runs_a_native_ppo_block_and_saves_artifacts(tmp_path, architecture):
    model = tiny_model(architecture)
    champion = tmp_path / "champion.pt"
    save_model(champion, model, model.config)
    script = Path(__file__).resolve().parents[1] / "scripts/train_flat.py"
    arguments = [
        str(script), "--champion-in", str(champion), "--rounds", "1",
        "--checkpoint-dir", str(tmp_path / "models"),
        "--league-db", str(tmp_path / "league.sqlite"),
        "--project", str(tmp_path / "runs"), "--device", "cpu",
        "--seed", "3187", "--evaluation-seed", "1503187", "--num-threads", "1",
        "--population-size", "1", "--promotion-games", "2", "--historical-games", "2",
        "--direct-gate-games", "2", "--direct-population-gate-games", "2",
        "--direct-population-diagnostic-gate-games", "2",
        "--direct-population-envs", "2", "--direct-population-steps", "16",
        "--direct-population-max-steps", "16", "--direct-population-block-steps", "16",
        "--post-bootstrap-population-steps", "16", "--post-bootstrap-population-max-steps", "16",
        "--direct-population-ppo-rollout-steps", "4",
        "--direct-population-ppo-minibatch-size", "4",
        "--direct-population-ppo-epochs", "1",
        "--max-moves", "4", "--direct-gate-max-moves", "4", "--visible-action-history", "0",
        "--no-breakthrough-exploiter", "--no-direct-population-ppg",
        "--flat-population-terminal-replay-value-weight", "0.0",
        "--flat-population-regression-confirm-games", "0",
        "--flat-value-updates", "1", "--flat-value-validation-games", "2",
        "--flat-rollout-steps-final", "4",
    ]
    with patch.object(sys, "argv", arguments), \
         patch.object(sys, "path", [str(script.parent), *sys.path]), \
         patch.object(TrainingRun, "_write_plots"):
        runpy.run_path(str(script), run_name="__main__")
    checkpoints = list((tmp_path / "runs").rglob("latest_state.pt"))
    assert len(checkpoints) == 1
    state = torch.load(checkpoints[0], weights_only=False)
    assert state["optimizer"]["state"]
    assert state["policy_version"] > 0
    assert all(torch.isfinite(tensor).all() for tensor in state["model_state"].values())
    assert (checkpoints[0].parent / "selected_model.pt").is_file()
    validation = torch.load(checkpoints[0].parent / "value_validation.pt", weights_only=False)
    assert not validation["examples"] and validation["truncated_games"] == 2


def test_ablation_dry_run_isolates_paths_without_copying_or_executing_training(tmp_path):
    source_db = tmp_path / "original.sqlite"
    with sqlite3.connect(source_db) as connection:
        connection.execute("CREATE TABLE evidence (score REAL)")
        connection.execute("INSERT INTO evidence VALUES (0.5)")
    original = source_db.read_bytes()
    output = tmp_path / "plan"
    script = Path(__file__).resolve().parents[1] / "scripts/flat_experiment_matrix.py"
    arguments = [
        str(script), "--output", str(output), "--seeds", "11,12",
        "--variants", "control,full", "--", "python", "scripts/train_flat.py",
        "--champion-in", "models/initial.pt", "--league-db=" + str(source_db),
    ]
    with patch.object(sys, "argv", arguments):
        runpy.run_path(str(script), run_name="__main__")
    plan = json.loads((output / "plan.json").read_text())
    assert len(plan) == 4 and {p["status"] for p in plan} == {"planned"}
    assert not list(output.rglob("*.sqlite"))
    assert source_db.read_bytes() == original
    paths = [p["command"][p["command"].index("--checkpoint-dir") + 1] for p in plan]
    assert len(set(paths)) == 4


@pytest.mark.parametrize("confirmed", [False, True])
def test_curriculum_confirms_regression_on_fresh_paired_maps_and_protects_best(tmp_path, confirmed):
    candidate = tiny_model()
    champion = deepcopy(candidate)
    initial = deepcopy(candidate.state_dict())
    scores = [0.5, 0.3, 0.0 if confirmed else 0.5, 1.0 if confirmed else 0.5]
    scores += [0.5] if confirmed else [0.4, 0.5]
    measurements = []
    def evaluate(*args, **kwargs):
        score = scores[len(measurements)]
        measurements.append(kwargs)
        values = [score] * 40
        return {"model_a_score": score, "worst_score": score,
                "model_a_city_captures_per_game": 0.0, "completion_rate": 1.0,
                "opponents": [{"model_a_score": score, "game_rows": paired_rows(values)}]}
    def update(*args, **kwargs):
        with torch.no_grad():
            next(candidate.parameters()).add_(0.01)
        return {"steps": 1.0}
    settings = FlatImprovements(value_updates=0, scientific_artifacts=False,
                                cache_evaluations=False, rollout_steps_final=0)
    with patch("polybot.training.vector_population.evaluate_direct_population_matches", side_effect=evaluate), \
         patch("polybot.training.vector_population.train_vector_population", side_effect=update), \
         patch.object(TrainingRun, "_write_plots"):
        result = train_direct_population_curriculum(
            TrainingConfig(device="cpu"), candidate.config, candidate, champion,
            [champion], [1.0], num_envs=2, minimum_steps=1, maximum_steps=1,
            block_steps=1, num_threads=1, max_actions=16, gate_games=2, gate_seed=127,
            gate_score=0.5, historical_score_floor=0.4,
            readiness_city_captures_per_game=0.0, readiness_completion_rate=0.0,
            readiness_patience=1, policy_anchor_weight=0.0, value_anchor_weight=0.0,
            opponent_policy_weight=0.0, teacher_weight=0.0, shaping_weight=0.0,
            ema_enabled=False, flat_improvements=settings,
            flat_regression_confirmation_games=80,
            run=TrainingRun.create(tmp_path, "paired"),
        )
    assert measurements[2]["gate_seed"] == measurements[3]["gate_seed"]
    assert measurements[2]["gate_seed"] > measurements[0]["gate_seed"]
    block = result["direct_population_gate_rows"][1]
    assert bool(block["accepted"]) is not confirmed
    for name, tensor in candidate.state_dict().items():
        assert torch.equal(tensor, initial[name]), name
