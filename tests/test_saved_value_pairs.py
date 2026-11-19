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
from polybot.training.flat_value import EncodedValuePosition, fit_paired_value_heads
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
