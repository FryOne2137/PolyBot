import importlib.util
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from polybot.config import ModelConfig, TrainingConfig
from polybot.evaluation.mcts_diagnostics import _ReplayPosition
from polybot.evaluation.public_rollouts import PublicRolloutPool
from polybot.evaluation.value_disagreements import Disagreement, _public_row, _save_supervised_pairs
from polybot.infrastructure.model_io import load_model
from polybot.models.policy_value_belief import PolicyValueBeliefNet
from polybot.training.flat_experiments import FlatImprovements
from polybot.training.flat_value import EncodedValuePosition, fit_paired_value_heads, paired_outcome_weights
from polybot.training.mcts_value import _encode_value_rows
from polybot.training.run_artifacts import TrainingRun
from polybot.training.saved_value_pairs import load_saved_value_pairs, split_value_pairs


@pytest.fixture
def saved_pairs(tmp_path):
    model = PolicyValueBeliefNet(ModelConfig(
        d_model=16, transformer_layers=1, attention_heads=4, belief_decoder_layers=1,
        action_encoder_dim=16, action_encoder_heads=4, action_latent_count=4, dropout=0,
    )).eval()
    training = TrainingConfig(device="cpu", seed=2137)
    pool = PublicRolloutPool(training, (model, model), num_envs=1, visible_action_history=0)
    metadata, rollouts, pairs, scores = [], [], [], []
    for index, seed in enumerate(range(2137, 2141)):
        pair_id = index + 1
        position = _ReplayPosition(index, seed, 0, 0, "public")
        request = pool.reset(np.array([seed], dtype=np.uint32), np.array([[0, 1]], dtype=np.int32))
        actions = tuple(map(int, request["action_id"][0, :2]))
        pairs.append(Disagreement(pair_id, position, actions))
        for arm, action in zip(("zero", "soft_z"), actions, strict=True):
            request = pool.reset(np.array([seed], dtype=np.uint32), np.array([[0, 1]], dtype=np.int32))
            pool.step_public_policy(np.array([action], dtype=np.int32), request["state_id"], request)
            features = _encode_value_rows(model, training, pool.device, request, np.array([0]))[0]
            np.savez_compressed(tmp_path / f"pair-{pair_id:03d}-{arm}-successor.npz",
                                **_public_row(request, 0), features=features)
        metadata.append({
            "pair_id": pair_id, "map_seed": seed, "player": 0, "phase": ("opening", "mid", "late")[index % 3],
            "choices": {"zero": {"action_id": actions[0]}, "soft_z": {"action_id": actions[1]}},
        })
        scores.append({
            "pair_id": pair_id, "map_seed": seed, "completed_paired_repeats": 4,
            "mean_scores": {"zero": 0.0, "soft_z": 1.0}, "soft_z_minus_zero": 1.0,
        })
        rollouts.extend({
            "pair_id": pair_id, "repeat": repeat,
            "outcomes": {"zero": {"score_a": 0.0, "truncated": False}, "soft_z": {"score_a": 1.0, "truncated": False}},
        } for repeat in range(1, 5))
    (tmp_path / "config.json").write_text(json.dumps({
        "schema_version": 1, "training": asdict(training), "source_games": 8, "visible_action_history": 0,
    }))
    for filename, rows in (("pairs.jsonl", metadata), ("rollouts.jsonl", rollouts)):
        (tmp_path / filename).write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert _save_supervised_pairs(tmp_path, pairs, scores) == 4
    return model, training, tmp_path


def test_saved_pairs_reencode_real_successors_and_split_whole_maps(saved_pairs):
    model, training, path = saved_pairs
    original = {key: value.clone() for key, value in model.state_dict().items()}
    examples, source = load_saved_value_pairs(model, training, path, history=0)
    assert source["labelled_pairs"] == 4 and len(examples) == 8
    assert source["reserved_map_seeds"] == [2137, 2138, 2139, 2140]
    # Action zero ends the turn. Both stored targets must be converted back
    # to the original player before comparing the two actions.
    assert all(example.to_root_sign == -1 for example in examples[::2])
    assert all(example.return_target == 1 for example in examples)
    assert all(pair["root_gap"] == 2 for pair in source["pairs"])
    datasets, split = split_value_pairs(examples, seed=81, selection_fraction=0.25)
    assert len(split["train"]["map_seeds"]) == 3 and len(split["selection"]["map_seeds"]) == 1
    assert not set(split["train"]["map_seeds"]) & set(split["selection"]["map_seeds"])
    assert split_value_pairs(examples, seed=81, selection_fraction=0.25)[1] == split
    assert len(datasets["train"]) == 6 and len(datasets["selection"]) == 2
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in original.items())


def test_saved_pairs_reserve_every_attempted_adaptive_source_map(saved_pairs):
    model, training, path = saved_pairs
    config = json.loads((path / "config.json").read_text())
    config.update(source_games=0, reserved_map_seeds=[2137, 2138, 2139, 2140, 9001])
    (path / "config.json").write_text(json.dumps(config))
    _, source = load_saved_value_pairs(model, training, path, history=0)
    assert source["reserved_map_seeds"] == [2137, 2138, 2139, 2140, 9001]


def test_soft_labels_recompute_uncertainty_and_preserve_terminal_perspective(saved_pairs):
    model, training, path = saved_pairs
    original_labels = (path / "labelled_pairs.jsonl").read_bytes()
    config = json.loads((path / "config.json").read_text())
    config.update(label_mode="weighted_soft", labels_file="weighted_pairs.jsonl", pair_weighting_version=1)
    (path / "config.json").write_text(json.dumps(config))
    labels = [json.loads(line) for line in original_labels.splitlines()]
    reliability = paired_outcome_weights([[0, 1]] * 4)
    for label in labels:
        label.update(reliability)
    target = path / "weighted_pairs.jsonl"
    target.write_text("".join(json.dumps(row) + "\n" for row in labels))
    rows, source = load_saved_value_pairs(model, training, path, history=0)
    assert source["labelled_pairs"] == 4
    assert all(row.supervision_weight == reliability["supervision_weight"] and row.gap_weight == reliability["gap_weight"]
               for row in rows)
    assert all(row.return_target == 1 for row in rows)
    assert all(row.to_root_sign == -1 for row in rows[::2])
    assert (path / "labelled_pairs.jsonl").read_bytes() == original_labels
    labels[0]["gap_weight"] = 1.0
    target.write_text("".join(json.dumps(row) + "\n" for row in labels))
    with pytest.raises(ValueError, match="uncertainty"):
        load_saved_value_pairs(model, training, path, history=0)


def test_later_paired_repeat_saves_a_missing_successor_without_overwriting_existing_data(saved_pairs, monkeypatch):
    from polybot.evaluation import value_disagreements as diagnostic

    model, training, path = saved_pairs
    pool = PublicRolloutPool(training, (model, model), num_envs=2, visible_action_history=0)
    packet = pool.reset(np.array([2137, 2137], dtype=np.uint32), np.array([[0, 1], [0, 1]], dtype=np.int32))
    pair = Disagreement(99, _ReplayPosition(0, 2137, 0, 0, "root"), (0, 1))
    search = SimpleNamespace(models=(model, model), training=training, device=torch.device("cpu"))
    monkeypatch.setattr(diagnostic, "_replay_positions", lambda *args, **kwargs: packet)

    def continuations(*args, **kwargs):
        kwargs["after_first_action"](packet)
        return []

    monkeypatch.setattr(diagnostic, "_play_replayed_continuations", continuations)
    diagnostic._branches(search, [pair], {0: []}, 1, 0.5, path, save_missing_successors=True)
    target = path / "pair-099-zero-successor.npz"
    with np.load(target, allow_pickle=False) as data:
        assert data["features"].shape == (16,)
    original = target.read_bytes()
    diagnostic._branches(search, [pair], {0: []}, 2, 0.5, path, save_missing_successors=True)
    assert target.read_bytes() == original


def test_mixed_repeat_batch_keeps_opponent_rngs_paired_per_position(saved_pairs, monkeypatch):
    from polybot.evaluation import value_disagreements as diagnostic

    model, training, path = saved_pairs
    pool = PublicRolloutPool(training, (model, model), num_envs=4, visible_action_history=0)
    packet = pool.reset(np.array([2137, 2137, 2138, 2138], dtype=np.uint32), np.array([[0, 1]] * 4, dtype=np.int32))
    pairs = [Disagreement(key, _ReplayPosition(key, seed, 0, 0, "root"), (0, 1))
             for key, seed in ((99, 2137), (100, 2138))]
    search = SimpleNamespace(models=(model, model), training=training, device=torch.device("cpu"))
    monkeypatch.setattr(diagnostic, "_replay_positions", lambda *args, **kwargs: packet)

    def continuations(*args, **kwargs):
        kwargs["after_first_action"](packet)
        actual = [generator.integers(0, 2**31, size=8) for generator in kwargs["opponent_rngs"]]
        expected = [np.random.default_rng((training.seed, seed, 0, 0, repeat, 1)).integers(0, 2**31, size=8)
                    for seed, repeat in ((2137, 3), (2138, 7))]
        for index in range(4):
            np.testing.assert_array_equal(actual[index], expected[index // 2])
        assert not np.array_equal(actual[0], actual[2])
        return []

    monkeypatch.setattr(diagnostic, "_play_replayed_continuations", continuations)
    diagnostic._branches(search, pairs, {99: [], 100: []}, 0, 0.5, path,
                         save_missing_successors=True, repeat_by_pair={99: 3, 100: 7})


@pytest.mark.parametrize("corruption", ["target", "encoder", "map_type", "rollout"])
def test_saved_pairs_reject_incompatible_or_unpaired_labels(saved_pairs, corruption):
    model, training, path = saved_pairs
    if corruption == "target":
        labels = [json.loads(line) for line in (path / "labelled_pairs.jsonl").read_text().splitlines()]
        labels[0]["branches"]["zero"]["return_target"] = -1
        (path / "labelled_pairs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in labels))
    elif corruption == "encoder":
        with torch.no_grad():
            model.global_encoder[0].bias[0].add_(3.0)
    elif corruption == "map_type":
        training = replace(training, map_type="drylands")
    else:
        rows = [json.loads(line) for line in (path / "rollouts.jsonl").read_text().splitlines()]
        rows[0]["outcomes"]["soft_z"]["truncated"] = True
        (path / "rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        load_saved_value_pairs(model, training, path, history=0)


def test_saved_pair_fit_keeps_reference_if_updates_hurt_selection(saved_pairs):
    model, _, _ = saved_pairs
    model.requires_grad_(False)
    model.value_head.requires_grad_(True)
    with torch.no_grad():
        model.value_head[-2].weight.zero_()
        model.value_head[-2].bias.zero_()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    features = np.ones(16, dtype=np.float32)
    replay = [EncodedValuePosition(features, 0, 1, (1, pair), "opening") for pair in range(2) for _ in range(2)]
    selection = [EncodedValuePosition(features, 0, -1, (2, pair), "opening") for pair in range(2) for _ in range(2)]
    history = []
    heads, report = fit_paired_value_heads(
        model, replay, selection, FlatImprovements(value_updates=4, value_batch_size=4, value_learning_rate=0.01),
        seed=2137, difference_weight=1.0, loss_variants=("mse_delta",), preserve_reference=True,
        update_callback=lambda name, step, metrics: history.append((name, step, metrics)),
    )
    assert set(heads) == {"mse_delta"}
    assert not report["mse_delta"]["kept_update"] and report["mse_delta"]["selected_updates"] == 0
    assert report["mse_delta"]["updates"] == 4
    assert [step for _, step, _ in history] == [0, 1, 2, 3, 4]
    assert history[-1][2]["train"]["selection_loss"] < history[0][2]["train"]["selection_loss"]
    assert history[-1][2]["selection"]["selection_loss"] > history[0][2]["selection"]["selection_loss"]
    assert report["mse_delta"]["train_after"] == report["mse_delta"]["train_before"]
    assert report["mse_delta"]["last_update"] == history[-1][2]
    assert history[-1][2]["gradient_norm"] > 0 and history[-1][2]["best_update"] == 0
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert all(torch.equal(value, heads["mse_delta"][key.removeprefix("value_head.")])
               for key, value in before.items() if key.startswith("value_head."))


def _trainer(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("saved_pair_trainer", scripts / "train_value.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_saved_pair_trainer_freezes_weights_and_uses_new_paired_maps(saved_pairs, monkeypatch):
    model, training, path = saved_pairs
    trainer = _trainer(monkeypatch)
    model.requires_grad_(False)
    model.value_head.requires_grad_(True)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    calls = []

    def arena(candidate, opponent, configuration, mcts, games, **kwargs):
        calls.append(kwargs)
        assert configuration.seed == 5000 and games == 4 and mcts.simulations == 32
        assert kwargs["isolate_failed_pairs"] and kwargs["opponent_value_scale"] == 1.0
        assert set(kwargs["fitted_heads"]) == {"mse_delta"}
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
        return {
            "valid": True, "beats_zero": False, "compared_games_per_arm": 4,
            "paired_scores": {"mse_delta": 0.5, "zero": 0.5},
        }

    monkeypatch.setattr("polybot.training.mcts_value.evaluate_mcts_value_heads", arena)
    args = SimpleNamespace(
        saved_pairs=path, visible_action_history=0, selection_seed=81, pair_selection_fraction=0.25,
        simulations=32, particles=4, difference_loss_weight=1.0, updates=4, batch_size=4, learning_rate=1e-5,
        model_out=path / "new.pt", num_envs=4, num_threads=1, max_pending_leaves=4, opponent_value_scale=1.0,
        benchmark_unchanged_head=True, fit_log_interval=2,
    )
    run = TrainingRun(path / "run")
    schedules = (("train", 1000, 4), ("selection", 3000, 4), ("validation", 5000, 4))
    report = trainer.fit_saved_mcts_pairs(args, model, model, model.config, training, schedules, run)
    assert len(calls) == 1 and report["saved_checkpoint"] == str(args.model_out)
    fitted, _ = load_model(args.model_out, torch.device("cpu"))
    assert all(torch.equal(value, fitted.state_dict()[key]) for key, value in before.items()
               if not key.startswith("value_head."))
    assert report["reason"] == "mcts_benchmark_complete"
    assert report["benchmark_performed"] and report["beats_zero"] is False
    history = [json.loads(line) for line in (run.directory / "metrics.jsonl").read_text().splitlines()]
    updates = [row for row in history if "value_update" in row]
    assert [row["value_update"] for row in updates] == [0, 1, 2, 3, 4]
    assert all("train_loss" in row and "selection_loss" in row for row in updates)
    with pytest.raises(ValueError, match="overlaps"):
        trainer.fit_saved_mcts_pairs(args, model, model, model.config, training,
                                     (*schedules[:2], ("validation", 2137, 4)), run)


def test_saved_pair_split_preserves_phase_and_decisive_coverage_when_possible():
    features = np.ones(16, dtype=np.float32)
    specifications = {
        1: [("opening", 0), ("mid", 0), ("late", 1)],
        2: [("opening", 1)], 3: [("mid", 0)], 4: [("late", 0)],
    }
    examples = [
        EncodedValuePosition(features, 0, target, (seed, 0, pair_id), phase)
        for seed, pairs in specifications.items() for pair_id, (phase, gap) in enumerate(pairs)
        for target in (0, gap)
    ]
    datasets, report = split_value_pairs(examples, seed=81, selection_fraction=0.25)
    assert report["selection"]["map_seeds"] == [1]
    assert report["missing_categories"] == 0
    for split in ("train", "selection"):
        assert all(report[split]["phase_pairs"].values())
        assert report[split]["informative_pairs"] == 1 and not report[split]["coverage_gaps"]
    assert len(datasets["selection"]) == len(datasets["train"]) == 6


@pytest.mark.parametrize("force_benchmark", [False, True])
def test_unchanged_head_skips_arena_unless_requested(saved_pairs, monkeypatch, force_benchmark):
    from copy import deepcopy

    model, training, path = saved_pairs
    trainer = _trainer(monkeypatch)
    calls = []

    def fit(candidate, replay, selection, settings, **kwargs):
        metrics = {"rmse": 1.0, "gap_rmse": 0.5, "preference_accuracy": 0.0, "selection_loss": 1.0}
        return {"mse_delta": deepcopy(candidate.value_head.state_dict())}, {
            "mse_delta": {"before": metrics.copy(), "after": metrics.copy(), "kept_update": False,
                          "updates": 4, "selected_updates": 0},
        }

    def arena(*args, **kwargs):
        calls.append(True)
        return {"valid": True, "beats_zero": True, "compared_games_per_arm": 4,
                "paired_scores": {"mse_delta": 0.75, "zero": 0.5}}

    monkeypatch.setattr(trainer, "fit_paired_value_heads", fit)
    monkeypatch.setattr("polybot.training.mcts_value.evaluate_mcts_value_heads", arena)
    args = SimpleNamespace(
        saved_pairs=path, visible_action_history=0, selection_seed=81, pair_selection_fraction=0.25,
        simulations=32, particles=4, difference_loss_weight=1.0, updates=4, batch_size=4, learning_rate=1e-5,
        model_out=path / "new.pt", num_envs=4, num_threads=1, max_pending_leaves=4, opponent_value_scale=1.0,
        benchmark_unchanged_head=force_benchmark, fit_log_interval=2,
    )
    run = TrainingRun(path / "run")
    report = trainer.fit_saved_mcts_pairs(args, model, model, model.config, training,
                                        (("train", 1000, 4), ("selection", 3000, 4), ("validation", 5000, 4)), run)
    assert bool(calls) is force_benchmark and report["benchmark_performed"] is force_benchmark
    assert not report["head_updated"] and Path(report["saved_checkpoint"]).is_file()
    assert "mcts_improved" not in report
    assert report["beats_zero"] is (True if force_benchmark else None)
    assert report["reason"] == ("mcts_benchmark_complete" if force_benchmark else "no_selection_improvement")


def test_offline_comparison_reuses_saved_data_and_restores_network(saved_pairs):
    from polybot.training.value_pair_diagnostics import compare_saved_value_losses

    model, training, path = saved_pairs
    model.requires_grad_(False)
    model.value_head.requires_grad_(True)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    run = TrainingRun(path / "comparison")
    report = compare_saved_value_losses(
        model, training, path, FlatImprovements(value_updates=4, value_batch_size=4),
        run, history=0, split_seed=81, selection_fraction=0.25, difference_weight=1.0, log_interval=2,
    )
    run.finish(report, plots=False)
    assert report["benchmark_performed"] is False
    assert set(report["loss_comparison"]) == {"mse", "delta", "mse_delta"}
    assert not set(report["split"]["train"]["map_seeds"]) & set(report["split"]["selection"]["map_seeds"])
    for name, result in report["loss_comparison"].items():
        history = [json.loads(line) for line in Path(result["history"]).read_text().splitlines()]
        assert [row["update"] for row in history] == [0, 1, 2, 3, 4]
        assert all("train_gap_mse" in row and "selection_gap_mse" in row for row in history)
        assert (run.directory / name / "metrics.csv").is_file()
    pairs = [json.loads(line) for line in Path(report["pair_diagnostics"]).read_text().splitlines()]
    assert {row["stage"] for row in pairs} == {
        "initial_train", "initial_selection", "last_mse_train", "last_delta_train", "last_mse_delta_train",
    }
    assert all("delta_loss_gradient_norm" in row and "to_root_signs" in row for row in pairs)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_memorization_uses_only_informative_training_pairs(saved_pairs):
    from polybot.training.value_pair_diagnostics import compare_saved_value_losses

    model, training, path = saved_pairs
    labels = [json.loads(line) for line in (path / "labelled_pairs.jsonl").read_text().splitlines()]
    labels[0]["branches"]["soft_z"]["return_target"] = -1.0
    labels[0]["mean_scores"]["soft_z"] = 0.0
    labels[0]["soft_z_minus_zero"] = 0.0
    (path / "labelled_pairs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in labels))
    rollouts = [json.loads(line) for line in (path / "rollouts.jsonl").read_text().splitlines()]
    for row in rollouts:
        if row["pair_id"] == 1:
            row["outcomes"]["soft_z"]["score_a"] = 0.0
    (path / "rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rollouts))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    run = TrainingRun(path / "memorization")
    report = compare_saved_value_losses(
        model, training, path, FlatImprovements(value_updates=4, value_batch_size=4),
        run, history=0, split_seed=81, selection_fraction=0.25, difference_weight=1.0, log_interval=2,
        memorization_test=True,
    )
    effective_ids = set(report["memorization"]["pair_ids"])
    assert effective_ids == set(report["split"]["train"]["pair_ids"]) - {1}
    assert not effective_ids & set(report["split"]["selection"]["pair_ids"])
    assert set(report["loss_comparison"]) == {"delta"}
    assert report["initial_diagnostics"]["train"]["informative_pairs"] == len(effective_ids)
    assert report["initial_diagnostics"]["train"]["gradients"]["scope"] == "full_batch_split"
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


@pytest.mark.parametrize("admit", [False, True])
def test_calibrated_pipeline_gates_arena_and_reserves_all_maps(saved_pairs, monkeypatch, admit):
    from copy import deepcopy

    model, training, path = saved_pairs
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("ranked_value", scripts / "train_ranked_value.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    original = deepcopy(model.state_dict())
    arena_calls = []

    def collect(candidate, opponent, configuration, mcts, games, **kwargs):
        seed = configuration.seed
        assert games == 4 and {2137, 2138, 2139, 2140, 7000, 7001} <= set(kwargs["excluded_map_seeds"])
        assert ({6000, 6001} if seed == 5000 else {5000, 5001}) <= set(kwargs["excluded_map_seeds"])
        rows = [EncodedValuePosition(np.ones(16, dtype=np.float32), 0, 1, (seed + i, 0), "opening")
                for i in range(2)]
        return rows, {"valid": True, "labelled_maps": 2, "attempted_map_seeds": [seed, seed + 1], "cache_loaded": False}

    def fit(candidate, *args, **kwargs):
        state = deepcopy(candidate.value_head.state_dict())
        state["2.bias"] += 0.1
        return (state if admit else None), {"selected": {"update": 1} if admit else None}

    def arena(candidate, opponent, configuration, mcts, games, **kwargs):
        arena_calls.append(True)
        assert configuration.seed == 7000 and games == 4
        assert set(kwargs["fitted_heads"]) == {"calibrated", "original"}
        assert all(torch.equal(value, candidate.state_dict()[key]) for key, value in original.items())
        return {"valid": True, "beats_zero": True, "paired_scores": {"calibrated": 0.75, "original": 0.5, "zero": 0.5}}

    monkeypatch.setattr(trainer, "cached_terminal_positions", collect)
    monkeypatch.setattr(trainer, "fit_calibrated_heads", fit)
    monkeypatch.setattr(trainer, "evaluate_mcts_value_heads", arena)
    args = SimpleNamespace(
        saved_pairs=[path], visible_action_history=0, pair_split_seed=81, pair_selection_fraction=0.25,
        calibration_games=4, calibration_selection_games=4, validation_games=4,
        calibration_selection_seed=6000, validation_seed=7000, simulations=32, particles=4,
        num_envs=4, num_threads=1, max_pending_leaves=4, calibration_weights=(0.05,),
        calibration_policy="mcts-zero", calibration_rmse_regression=0.0, calibration_bias_regression=0.02,
        calibration_cache=path / "cache", positions_per_game=3, updates=4, batch_size=4, learning_rate=0.0001,
        model_out=path / "calibrated.pt", log_interval=4,
    )
    report = trainer.run_experiment(args, model, model, model.config, replace(training, seed=5000),
                                     TrainingRun(path / "calibrated-run"))
    assert bool(arena_calls) is admit and args.model_out.is_file() is admit
    assert report["head_updated"] is admit and report["benchmark_performed"] is admit
    if admit:
        assert report["score_delta_vs_original"] == report["score_delta_vs_zero"] == 0.25


@pytest.mark.parametrize("enough_new_pairs", [False, True])
@pytest.mark.parametrize("mode", ["confirmed", "pooled"])
def test_new_pair_pipeline_loads_cache_first_and_trains_only_confirmed_new_maps(saved_pairs, monkeypatch, enough_new_pairs, mode):
    model, training, path = saved_pairs
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("ranked_collector", scripts / "train_ranked_value.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    calls = []
    new_dir = path / "new-pairs"
    old_rows, old_source = load_saved_value_pairs(model, training, path, history=0)
    load_sources = trainer.load_ranking_sources

    def load(candidate, configuration, directories, **kwargs):
        if directories == [new_dir]:
            new_rows = [replace(row, game_key=(row.game_key[0] + 10000, *row.game_key[1:])) for row in old_rows]
            if mode == "pooled":
                new_rows[:2] = [replace(row, return_target=float(row.to_root_sign), supervision_weight=0.5, gap_weight=0.4)
                                for row in new_rows[:2]]
            return new_rows, {
                "sources": [], "reserved_map_seeds": [12137, 12138, 12139, 12140],
            }
        return load_sources(candidate, configuration, directories, **kwargs)

    def cache(candidate, opponent, configuration, mcts, games, **kwargs):
        assert kwargs["require_existing"]
        calls.append("cache")
        seed = configuration.seed
        return [EncodedValuePosition(np.ones(16, dtype=np.float32), 0, 1, (seed + i, 0), "opening")
                for i in range(2)], {"valid": True, "labelled_maps": 2,
                                    "attempted_map_seeds": [seed, seed + 1, seed + 2], "cache_loaded": True}

    def collect(*args, **kwargs):
        assert calls == ["cache", "cache"]
        assert {2137, 2138, 2139, 2140, 5000, 5001, 5002, 6000, 6001, 6002, 7000, 7001} <= kwargs["excluded_map_seeds"]
        calls.append("collect")
        return {"confirmed_pairs": 0 if mode == "pooled" else 4 if enough_new_pairs else 0,
                "usable_pairs": 4 if enough_new_pairs else 0, "reached_target": enough_new_pairs}

    def fit(candidate, ranking, calibration, *args, **kwargs):
        calls.append("fit")
        assert {row.game_key[0] for rows in ranking.values() for row in rows} == {12137, 12138, 12139, 12140}
        assert kwargs["pair_value_weight"] == (1.0 if mode == "pooled" else 0.0)
        assert kwargs["balance_pair_maps"] is (mode == "pooled")
        return None, {"selected": None}

    monkeypatch.setattr(trainer, "load_ranking_sources", load)
    monkeypatch.setattr(trainer, "cached_terminal_positions", cache)
    monkeypatch.setattr(trainer, "collect_decisive_value_pairs", collect)
    monkeypatch.setattr(trainer, "fit_calibrated_heads", fit)
    args = SimpleNamespace(
        saved_pairs=[path], visible_action_history=0, pair_split_seed=81, pair_selection_fraction=0.25,
        calibration_games=4, calibration_selection_games=4, validation_games=4,
        calibration_selection_seed=6000, validation_seed=7000, simulations=32, particles=4,
        num_envs=4, num_threads=1, max_pending_leaves=4, calibration_weights=(0.05,),
        calibration_policy="mcts-zero", calibration_rmse_regression=0.0, calibration_bias_regression=0.02,
        calibration_cache=path / "cache", positions_per_game=3, updates=4, batch_size=4, learning_rate=0.0001,
        model_out=path / "new-head.pt", log_interval=4, require_calibration_cache=True,
        collect_decisive_pairs=4, minimum_collected_pairs=4, collected_pairs_dir=new_dir,
        pair_source_seed=9000, pair_source_games=8, pair_source_batch_games=4, pair_probes_per_game=3,
        pair_min_rollouts=4, pair_max_rollouts=16, pair_rollout_step=4,
        pair_confidence=0.95, pair_rollout_temperature=0.5, pair_max_pairs_per_map=2, pair_collection_hours=8,
        pair_collection_mode=mode,
    )
    report = trainer.run_experiment(args, model, model, model.config, replace(training, seed=5000),
                                    TrainingRun(path / "collect-run"))
    assert ("fit" in calls) is enough_new_pairs
    assert not report["benchmark_performed"]
    assert report["reason"] == ("no_calibrated_ranking_improvement" if enough_new_pairs else
                                "insufficient_usable_new_pairs" if mode == "pooled" else "insufficient_confirmed_new_pairs")


@pytest.fixture
def replayable_value_run(saved_pairs, monkeypatch):
    from copy import deepcopy
    from polybot.infrastructure.model_io import save_model
    from polybot.training.calibrated_value import cached_terminal_positions, fit_calibrated_heads, load_ranking_sources
    from polybot.config import MCTSConfig
    from polybot.training.flat_experiments import model_fingerprint

    model, training, path = saved_pairs
    opponent = deepcopy(model)
    training = replace(training, seed=5000, batch_size=4, learning_rate=0.02)
    dataset = json.loads((path / "config.json").read_text())
    dataset.update(label_mode="weighted_soft", labels_file="weighted_pairs.jsonl", pair_weighting_version=1)
    (path / "config.json").write_text(json.dumps(dataset))
    labels = [json.loads(line) for line in (path / "labelled_pairs.jsonl").read_text().splitlines()]
    for row in labels:
        row.update(paired_outcome_weights([[0, 1]] * 4))
    (path / "weighted_pairs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in labels))
    save_model(path / "input.pt", model, model.config)
    save_model(path / "opponent.pt", opponent, opponent.config)
    rows, source = load_ranking_sources(model, training, [path], history=0)
    ranking, split = split_value_pairs(rows, seed=81, selection_fraction=0.25)
    pool = dict(num_envs=4, num_threads=1, visible_action_history=0, max_pending_leaves=1, opponent_value_scale=1.0)
    mcts = MCTSConfig(simulations=32, particles=4)
    calls = []

    def collect(candidate, opponent, configuration, *args, **kwargs):
        calls.append(configuration.seed)
        rows = [EncodedValuePosition(feature.features.copy(), 0, float(1 if i % 2 else -1),
                                     (configuration.seed + i, i), "opening") for i, feature in enumerate(ranking["train"][:2])]
        return rows, {"valid": True, "labelled_maps": 2, "positions": 2,
                      "map_seeds": [configuration.seed, configuration.seed + 1],
                      "attempted_map_seeds": [configuration.seed, configuration.seed + 1, configuration.seed + 2]}

    monkeypatch.setattr("polybot.training.calibrated_value.collect_mcts_terminal_value_games", collect)
    calibration, cache = {}, {}
    for name, seed in (("train", 5000), ("selection", 6000)):
        calibration[name], cache[name] = cached_terminal_positions(model, opponent, replace(training, seed=seed), mcts, 4,
            cache_directory=path / "cache", excluded_map_seeds={7000, 7001}, positions_per_game=3, pool=pool, policy="mcts-zero")
    arguments = dict(model_in=str(path / "input.pt"), opponent=str(path / "opponent.pt"),
                     updates=12, batch_size=4, learning_rate=0.02, calibration_cache=str(path / "cache"),
                     num_envs=4, num_threads=1, visible_action_history=0, max_pending_leaves=1, positions_per_game=3)
    run = TrainingRun(path / "source-run", arguments=arguments)
    run.write_config(training=training, mcts=mcts, ranking_sources=source, ranking_split=split,
                     checkpoint_fingerprints={"model": model_fingerprint(model), "opponent": model_fingerprint(opponent)},
                     calibration_weights=(0.01, 0.05, 0.2, 1.0), schedules={"train": (5000, 4), "selection": (6000, 4), "arena": (7000, 4)},
                     calibration_policy="mcts-zero", rmse_regression=0.0, bias_regression=0.02,
                     pair_value_loss_weight=1.0, balance_pair_maps=True)
    model.requires_grad_(False)
    model.value_head.requires_grad_(True)
    captured = {}

    def snapshot(weight, step, metrics, state):
        if weight == 1.0 and step == 7:
            captured.update(state=deepcopy(state), metrics=deepcopy(metrics))

    _, report = fit_calibrated_heads(model, ranking, calibration, training,
        FlatImprovements(value_updates=12, value_batch_size=4, value_learning_rate=0.02),
        weights=(0.01, 0.05, 0.2, 1.0), rmse_regression=0.0, bias_regression=0.02,
        run=run, log_interval=12, pair_value_weight=1.0, balance_pair_maps=True, update_callback=snapshot)
    run.finish({"calibration": cache, "fit": report}, plots=False)
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("value_reconstructor", scripts / "reconstruct_ranked_value.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return model, opponent, path, module, captured, calls


@pytest.mark.parametrize("mismatch", [False, True])
def test_reconstructs_exact_requested_step_from_full_sweep_and_gates_arena(replayable_value_run, monkeypatch, mismatch):
    from copy import deepcopy

    model, opponent, path, module, captured, calls = replayable_value_run
    original = deepcopy(model.state_dict())
    if mismatch:
        history = path / "source-run/calibration-1.0/metrics.jsonl"
        rows = [json.loads(line) for line in history.read_text().splitlines()]
        for row in rows:
            if row["update"] == 7:
                row["selection_gap_rmse"] += 1
        history.write_text("".join(json.dumps(row) + "\n" for row in rows))
    arena_calls = []

    def arena(candidate, rival, training, mcts, games, **kwargs):
        arena_calls.append(True)
        assert training.seed == 8000 and games == 4 and mcts.simulations == 32
        assert set(kwargs["fitted_heads"]) == {"original", "reconstructed"}
        assert all(torch.equal(value, kwargs["fitted_heads"]["reconstructed"][key]) for key, value in captured["state"].items())
        assert all(torch.equal(value, candidate.state_dict()[key]) for key, value in original.items())
        return {"valid": True, "paired_scores": {"reconstructed": 0.75, "original": 0.25, "zero": 0.5}}

    monkeypatch.setattr(module, "evaluate_mcts_value_heads", arena)
    args = SimpleNamespace(update=7, calibration_weight=1.0, device="cpu", calibration_cache=None,
                           reproduction_tolerance=1e-7, validation_seed=8000, validation_games=4,
                           log_interval=7, model_out=path / "reconstructed.pt")
    report = module.reconstruct(args, model, opponent, model.config, module.read_source(path / "source-run"),
                                TrainingRun(path / "replay"))
    assert report["reconstruction_matches"] is not mismatch
    assert bool(arena_calls) is not mismatch and args.model_out.is_file() is not mismatch
    assert calls == [5000, 6000]
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in original.items())
    if not mismatch:
        restored, _ = load_model(args.model_out, "cpu")
        assert all(torch.equal(value, restored.value_head.state_dict()[key]) for key, value in captured["state"].items())
        assert report["score_delta_vs_original"] == 0.5 and report["score_delta_vs_zero"] == 0.25
        assert (path / "replay/gap_pairs.jsonl").is_file() and report["diagnostic_only"]
    else:
        assert report["reason"] == "reconstruction_mismatch"


@pytest.mark.parametrize("corruption", ["dataset", "arena", "weights"])
def test_reconstruction_rejects_modified_data_and_calibration_replacement_maps(replayable_value_run, corruption):
    model, opponent, path, module, _, _ = replayable_value_run
    if corruption == "dataset":
        with (path / "pairs.jsonl").open("a") as stream:
            stream.write(" ")
    if corruption == "weights":
        with torch.no_grad():
            model.value_head[-2].bias.add_(0.1)
    args = SimpleNamespace(update=7, calibration_weight=1.0, device="cpu", calibration_cache=None,
                           reproduction_tolerance=1e-7, validation_seed=5002 if corruption == "arena" else 8000,
                           validation_games=4, log_interval=7, model_out=path / "reconstructed.pt")
    with pytest.raises(ValueError, match={"dataset": "dataset changed", "arena": "overlap", "weights": "initial weights"}[corruption]):
        module.reconstruct(args, model, opponent, model.config, module.read_source(path / "source-run"), TrainingRun(path / "replay"))
    assert not args.model_out.is_file()


@pytest.fixture
def audited_value_run(replayable_value_run, monkeypatch):
    model, opponent, path, reconstruction, captured, calls = replayable_value_run
    monkeypatch.setattr(reconstruction, "evaluate_mcts_value_heads", lambda *args, **kwargs: {
        "valid": True, "paired_scores": {"reconstructed": 0.5, "original": 0.25, "zero": 0.5}})
    args = SimpleNamespace(update=7, calibration_weight=1.0, device="cpu", calibration_cache=None,
                           reproduction_tolerance=1e-7, validation_seed=8000, validation_games=4,
                           log_interval=7, model_out=path / "reconstructed.pt")
    run = TrainingRun(path / "audit-run")
    report = reconstruction.reconstruct(args, model, opponent, model.config,
                                        reconstruction.read_source(path / "source-run"), run)
    run.finish(report, plots=False)
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location("value_scaler", scripts / "calibrate_value_scale.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate, _ = load_model(args.model_out, "cpu")
    return model, candidate, opponent, path, module, calls


@pytest.mark.parametrize("selected_scale", [0.0, 0.25])
def test_scale_analysis_uses_training_maps_only_and_tests_one_frozen_scale(audited_value_run, monkeypatch, selected_scale):
    from polybot.training.flat_experiments import model_fingerprint

    original, candidate, opponent, path, module, calls = audited_value_run
    before = model_fingerprint(candidate)
    source, audit = module.read_source(path / "source-run"), module.read_source(path / "audit-run")
    cv_function = module.cross_validate_scale

    def cv(pairs, calibration, **kwargs):
        assert set(pairs["map_seeds"]) == set(source["config"]["ranking_split"]["train"]["map_seeds"])
        assert not set(pairs["map_seeds"]) & set(source["config"]["ranking_split"]["selection"]["map_seeds"])
        assert set(calibration["map_seeds"]) == {5000, 5001}
        result = cv_function(pairs, calibration, **kwargs)
        result["selection"] = {"scale": selected_scale, "method": "fixture", "reason": "test arena routing"}
        return result

    arena_calls = []

    def arena(model, opponent, training, mcts, games, **kwargs):
        arena_calls.append(True)
        assert model_fingerprint(model) == before
        assert kwargs["value_scales"] == (0.0, selected_scale) and kwargs["opponent_value_scale"] == 1
        assert training.seed == 9000 and mcts.simulations == 32 and games == 4
        return {"valid": True, "paired_scores": {"0": 0.5, "0.25": 0.75}}

    monkeypatch.setattr(module, "cross_validate_scale", cv)
    monkeypatch.setattr(module, "evaluate_mcts_value_heads", arena)
    args = SimpleNamespace(device="cpu", calibration_cache=None, verification_tolerance=1e-6,
                           folds=2, fold_seed=91, calibration_weights=(0, 0.2), maximum_scale=1,
                           top_errors=2, validation_seed=9000, validation_games=4,
                           model_in=path / "reconstructed.pt", offline_only=False)
    run = TrainingRun(path / "scale-run")
    report = module.run_analysis(args, original, candidate, opponent, source, audit, run)
    assert bool(arena_calls) is (selected_scale > 0)
    assert model_fingerprint(candidate) == before and calls == [5000, 6000]
    assert not report["model_updated"]
    assert (run.directory / "gap_errors.jsonl").is_file()
    settings = json.loads((run.directory / "value_scale.json").read_text())
    assert settings["leaf_value_scale"] == selected_scale
    if selected_scale:
        assert report["score_delta_vs_zero"] == 0.25
    else:
        assert report["reason"] == "zero_selected" and not report["benchmark_performed"]


@pytest.mark.parametrize("corruption", ["head", "network", "arena"])
def test_scale_analysis_rejects_changed_checkpoint_and_old_test_maps(audited_value_run, corruption):
    original, candidate, opponent, path, module, _ = audited_value_run
    if corruption in ("head", "network"):
        parameter = (candidate.value_head[-2].bias if corruption == "head" else candidate.global_encoder[0].bias)
        with torch.no_grad():
            parameter.add_(0.1)
    args = SimpleNamespace(device="cpu", calibration_cache=None, verification_tolerance=1e-6,
                           folds=2, fold_seed=91, calibration_weights=(0, 0.2), maximum_scale=1,
                           top_errors=2, validation_seed=8000 if corruption == "arena" else 9000,
                           validation_games=4, model_in=path / "reconstructed.pt", offline_only=False)
    expected = {"head": "checkpoint metrics", "network": "frozen network", "arena": "previous audit"}
    with pytest.raises(ValueError, match=expected[corruption]):
        module.run_analysis(args, original, candidate, opponent, module.read_source(path / "source-run"),
                            module.read_source(path / "audit-run"), TrainingRun(path / "scale-run"))


def _pair_mse_sweeper(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("pair_mse_sweeper", scripts / "compare_pair_mse.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sweep_args():
    return SimpleNamespace(device="cpu", calibration_cache=None, pair_mse_weights=(0.0, 0.1, 1.0),
                           calibration_weight=1.0, updates=12, control_update=7, reproduction_tolerance=1e-6,
                           log_interval=12, model_out=None)


def test_pair_mse_sweep_reuses_data_and_initial_weights_with_identical_minibatches(replayable_value_run, monkeypatch):
    from polybot.training.flat_experiments import model_fingerprint

    model, opponent, path, _, _, calls = replayable_value_run
    module = _pair_mse_sweeper(monkeypatch)
    source = module.read_source(path / "source-run")
    fingerprint = model_fingerprint(model)
    fit = module.fit_calibrated_heads
    batches, execution = {}, []

    def compare(candidate, ranking, calibration, training, settings, **kwargs):
        weight = kwargs["pair_value_weight"]
        execution.append(weight)
        assert model_fingerprint(candidate) == fingerprint
        assert kwargs["weights"] == (1.0,) and kwargs["balance_pair_maps"]
        assert {row.game_key[0] for row in ranking["train"]} == set(source["config"]["ranking_split"]["train"]["map_seeds"])
        assert {row.game_key[0] for row in calibration["train"]} == {5000, 5001}
        observed = []

        def observe(head, inputs):
            states = inputs[0]
            if states.ndim == 2 and len(states) == settings.value_batch_size:
                observed.append(states.detach().cpu().clone())

        handle = candidate.value_head.register_forward_pre_hook(observe)
        try:
            result = fit(candidate, ranking, calibration, training, settings, **kwargs)
        finally:
            handle.remove()
        batches[weight] = observed
        return result

    monkeypatch.setattr(module, "fit_calibrated_heads", compare)
    run = TrainingRun(path / "sweep")
    report = module.run_experiment(_sweep_args(), model, opponent, source, run)
    assert execution == [1.0, 0.0, 0.1] and report["control_matches"]
    assert calls == [5000, 6000] and report["new_games_collected"] == 0 and not report["benchmark_performed"]
    assert model_fingerprint(model) == fingerprint and len(batches[1.0]) == 12
    for weight in (0.0, 0.1):
        assert len(batches[weight]) == len(batches[1.0])
        assert all(torch.equal(a, b) for a, b in zip(batches[weight], batches[1.0], strict=True))
    assert report["fixed_loss_weights"] == {"gap": 1, "terminal_cache_mse": 1}
    assert set(report["variants"]) == {"0.0", "0.1", "1.0"}
    assert set(report["gap_diagnostics"]) == {"train", "selection"}
    for weight, variant in report["variants"].items():
        assert variant["attempted_updates"] == 12
        assert variant["best_gap"]["rank"]["gap_rmse"] <= variant["last"]["rank"]["gap_rmse"]
        assert (run.directory / f"pair-mse-{weight}" / "best-gap-head.pt").is_file()
        assert (run.directory / f"pair-mse-{weight}" / "calibration-1.0" / "metrics.csv").is_file()
        groups = report["gap_diagnostics"]["selection"][f"pair-{weight}-last"]
        assert groups["tied"]["pairs"] + groups["informative"]["pairs"] == groups["all"]["pairs"]
    assert (run.directory / "gap_pairs.jsonl").is_file()
    if report["saved_checkpoint"]:
        saved, _ = load_model(Path(report["saved_checkpoint"]), "cpu")
        assert all(torch.equal(value, saved.state_dict()[key]) for key, value in model.state_dict().items()
                   if not key.startswith("value_head."))


def test_pair_mse_sweep_stops_before_other_variants_if_control_does_not_reproduce(replayable_value_run, monkeypatch):
    model, opponent, path, _, _, calls = replayable_value_run
    module = _pair_mse_sweeper(monkeypatch)
    history = path / "source-run/calibration-1.0/metrics.jsonl"
    rows = [json.loads(line) for line in history.read_text().splitlines()]
    for row in rows:
        if row["update"] == 7:
            row["selection_gap_rmse"] += 1
    history.write_text("".join(json.dumps(row) + "\n" for row in rows))
    run = TrainingRun(path / "sweep")
    report = module.run_experiment(_sweep_args(), model, opponent, module.read_source(path / "source-run"), run)
    assert report["reason"] == "control_reproduction_mismatch" and not report["control_matches"]
    assert not report["variants"] and report["saved_checkpoint"] is None
    assert not (run.directory / "pair-mse-0.0").exists() and not (run.directory / "candidate.pt").exists()
    assert calls == [5000, 6000]


def test_pair_mse_sweep_missing_cache_never_collects_games(replayable_value_run, monkeypatch):
    model, opponent, path, _, _, calls = replayable_value_run
    module = _pair_mse_sweeper(monkeypatch)
    args = _sweep_args()
    args.calibration_cache = path / "missing-cache"
    with pytest.raises(FileNotFoundError, match="required calibration cache"):
        module.run_experiment(args, model, opponent, module.read_source(path / "source-run"), TrainingRun(path / "sweep"))
    assert calls == [5000, 6000]


def _gradient_diagnostic(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("gradient_diagnostic", scripts / "diagnose_value_gradients.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_memorization_result(path, model):
    directory = path / "historical-memory"
    (directory / "delta").mkdir(parents=True)
    metadata = {"full_batch": True, "train_pairs": 2, "pair_ids": [1, 2],
                "max_training_preference_accuracy": 1.0, "first_perfect_training_update": 1}
    (directory / "args.json").write_text(json.dumps({"arguments": {
        "memorization_test": True, "model_in": str(path / "input.pt")}}))
    (directory / "config.json").write_text(json.dumps({"model": asdict(model.config),
        "source": {"source_sha256": {"pairs.jsonl": "historical-other-dataset"}}}))
    (directory / "summary.json").write_text(json.dumps({"memorization": metadata, "loss_comparison": {"delta": {
        "updates": 2, "train_before": {"gap_rmse": 1.0}, "train_last": {"gap_rmse": 0.1},
        "selection_last": {"preference_accuracy": 0.5, "rmse": 1.4}}}}))
    (directory / "delta/metrics.jsonl").write_text("".join(json.dumps({
        "update": update, "train_preference_accuracy": value}) + "\n"
        for update, value in ((0, 0.5), (1, 1.0), (2, 1.0))))
    return directory


def test_gradient_diagnostic_reuses_historical_memory_and_audited_head_without_games(audited_value_run, monkeypatch):
    from polybot.training.flat_experiments import model_fingerprint

    original, _, opponent, path, _, calls = audited_value_run
    module = _gradient_diagnostic(monkeypatch)
    source = module.read_source(path / "source-run")
    args = SimpleNamespace(device="cpu", calibration_cache=None,
        memorization_run=_write_memorization_result(path, original), audit_run=path / "audit-run",
        batches=4, probe_batches=2, calibration_weight=1.0, verification_tolerance=1e-6)
    fingerprint = model_fingerprint(original)
    run = TrainingRun(path / "gradient-run")
    report = module.run_diagnostics(args, original, opponent, source, run)
    assert report["weights_unchanged"] and model_fingerprint(original) == fingerprint
    assert calls == [5000, 6000] and report["new_games_collected"] == 0
    assert report["memorization"]["history_verified"] and not report["memorization"]["same_pair_dataset"]
    assert report["memorization"]["first_perfect_training_update"] == 1
    assert set(report["stages"]) == {"initial", "audited-7"}
    assert not list(run.directory.rglob("*.pt"))
    probes = [json.loads(line) for line in (run.directory / "adamw_probes.jsonl").read_text().splitlines()]
    initial = next(row for row in probes if row["stage"] == "initial" and row["mode"] == "full" and row["batch"] == 1)
    history = [json.loads(line) for line in (path / "source-run/calibration-1.0/metrics.jsonl").read_text().splitlines()]
    recorded = next(row for row in history if row["update"] == 1)
    assert initial["after"]["selection_gap_rmse"] == pytest.approx(recorded["selection_gap_rmse"], abs=1e-6)
    assert initial["after"]["cache_selection_rmse"] == pytest.approx(recorded["selection_value_rmse"], abs=1e-6)
    assert initial["after"]["train_gap_rmse"] == pytest.approx(recorded["train_gap_rmse"], abs=1e-6)
    assert len(probes) == 16


@pytest.mark.parametrize("corruption", ["history", "checkpoint", "cache"])
def test_gradient_diagnostic_rejects_incompatible_provenance_and_never_collects(replayable_value_run, monkeypatch, corruption):
    model, opponent, path, _, _, calls = replayable_value_run
    module = _gradient_diagnostic(monkeypatch)
    memory = _write_memorization_result(path, model)
    if corruption == "history":
        target = memory / "summary.json"
        value = json.loads(target.read_text())
        value["memorization"]["first_perfect_training_update"] = 2
        target.write_text(json.dumps(value))
    if corruption == "checkpoint":
        target = memory / "args.json"
        value = json.loads(target.read_text())
        value["arguments"]["model_in"] = str(path / "different.pt")
        target.write_text(json.dumps(value))
    args = SimpleNamespace(device="cpu", calibration_cache=path / "missing" if corruption == "cache" else None,
        memorization_run=memory, audit_run=None, batches=4, probe_batches=2,
        calibration_weight=1.0, verification_tolerance=1e-6)
    with pytest.raises((ValueError, FileNotFoundError)):
        module.run_diagnostics(args, model, opponent, module.read_source(path / "source-run"), TrainingRun(path / "grad-run"))
    assert calls == [5000, 6000]
