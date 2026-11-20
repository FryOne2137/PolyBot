from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from polybot.config import MCTSConfig, TrainingConfig
from polybot.training.calibrated_value import fit_calibrated_heads, paired_training_losses
from polybot.training.flat_experiments import FlatImprovements
from polybot.training.flat_value import EncodedValuePosition
from polybot.training.run_artifacts import TrainingRun
from polybot.training.value_routing import paired_value_admission, routed_pair_predictions


def test_routed_pair_loss_backpropagates_only_through_the_apprentice_in_root_perspective():
    head = nn.Linear(1, 1, bias=False)
    nn.init.zeros_(head.weight)
    states = torch.tensor([[[1.0], [50.0]]])
    frozen = torch.tensor([[0.0, 0.5]], requires_grad=True)
    predictions = routed_pair_predictions(head, states, (torch.tensor([[True, False]]), frozen))
    targets = torch.tensor([[0.5, 0.5]])
    signs = torch.tensor([[1.0, -1.0]])
    gap, mse = paired_training_losses(predictions, targets, signs, torch.ones(1), torch.ones(1))
    (gap + mse).backward()
    # Root targets are +0.5 versus -0.5. The opponent's feature and value are constants.
    assert head.weight.grad.item() == pytest.approx(-0.75)
    assert frozen.grad is None


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(1, 1)
        self.value_head = nn.Sequential(nn.Linear(1, 1), nn.Tanh())
        nn.init.zeros_(self.value_head[0].weight)
        nn.init.zeros_(self.value_head[0].bias)
        self.requires_grad_(False)
        self.value_head.requires_grad_(True)


def data():
    lookup = {}

    def pairs(seed):
        key = (seed, 0, 1)
        rows = [EncodedValuePosition(np.array([value], dtype=np.float32), 0, 0.5,
                    key, "opening", sign) for value, sign in ((1, 1), (50, -1))]
        lookup[key] = [{"agent_id": 0, "opponent_value": None}, {"agent_id": 1, "opponent_value": 0.5}]
        return rows

    calibration = {name: [EncodedValuePosition(np.array([1], dtype=np.float32), 0, 1,
                                              (seed, 0), "opening")]
                   for name, seed in (("train", 300), ("selection", 400))}
    return {"train": pairs(100), "selection": pairs(200)}, calibration, lookup


def fit(model, ranking, calibration, lookup, path, trace):
    return fit_calibrated_heads(model, ranking, calibration, TrainingConfig(device="cpu"),
        FlatImprovements(value_updates=12, value_batch_size=4, value_learning_rate=0.03),
        weights=(0.2,), rmse_regression=0, bias_regression=0.02, run=TrainingRun(path), log_interval=12,
        pair_value_weight=1, balance_pair_maps=True, routing_lookup=lookup,
        update_callback=lambda weight, step, metrics, state: trace.append((deepcopy(state), deepcopy(metrics))))


def test_routed_fit_can_improve_the_correct_ranking_and_restores_the_input_network(tmp_path):
    model = Net()
    ranking, calibration, lookup = data()
    before, trace = deepcopy(model.state_dict()), []
    state, report = fit(model, ranking, calibration, lookup, tmp_path, trace)
    assert state is not None and report["selected"]["rank"]["preference_accuracy"] == 1
    assert report["selected"]["rank"]["gap_rmse"] < report["reference"]["rank"]["gap_rmse"]
    assert report["selected"]["calibration"]["rmse"] < report["reference"]["calibration"]["rmse"]
    assert trace[-1][1]["routing_groups"]["selection"]["mixed_models"]["all"]["pairs"] == 1
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


def test_held_out_observations_and_opponent_values_never_change_training_updates(tmp_path):
    ranking, calibration, lookup = data()
    first, second = [], []
    fit(Net(), ranking, calibration, lookup, tmp_path / "first", first)
    for example in ranking["selection"] + calibration["selection"]:
        example.features[:] = -20
    lookup[(200, 0, 1)][1]["opponent_value"] = -0.8
    fit(Net(), ranking, calibration, lookup, tmp_path / "second", second)
    for (a, _), (b, _) in zip(first, second, strict=True):
        assert all(torch.equal(value, b[name]) for name, value in a.items())


def test_fitting_rejects_pairs_without_any_apprentice_gradient_path(tmp_path):
    ranking, calibration, lookup = data()
    lookup[(100, 0, 1)][0] = {"agent_id": 1, "opponent_value": 0.2}
    with pytest.raises(ValueError, match="no saved training branch"):
        fit(Net(), ranking, calibration, lookup, tmp_path, [])


def arena(count=16, *, candidate=1.0, control=0.0):
    return {"valid": True, "paired_map_scores": [{"map_seed": seed, "scores_by_player": {
        "calibrated": [candidate, candidate], "original": [control, control], "zero": [control, control]}}
        for seed in range(count)]}


def test_arena_admission_requires_complete_maps_and_evidence_against_both_controls():
    result = paired_value_admission(arena(), planned_games=32)
    assert result["accepted"] and result["comparisons"]["zero"]["positive_maps"] == 16
    assert not paired_value_admission(arena(candidate=0.0), planned_games=32)["accepted"]
    assert not paired_value_admission(arena(15), planned_games=32)["accepted"]
    assert not paired_value_admission(arena(1), planned_games=2)["accepted"]
    values = arena()
    for row in values["paired_map_scores"]:
        row["scores_by_player"]["original"] = [1, 1]
    assert not paired_value_admission(values, planned_games=32)["accepted"]
    values["paired_map_scores"][0]["scores_by_player"]["calibrated"] = [None, 1]
    with pytest.raises(ValueError, match="terminal outcomes"):
        paired_value_admission(values, planned_games=32)


def test_greedy_branch_continuations_do_not_sample_the_raw_opponent(tmp_path, monkeypatch):
    from polybot.evaluation import value_disagreements as module
    from polybot.evaluation.mcts_diagnostics import _ReplayPosition

    position = _ReplayPosition(0, 100, 0, 1, "root")
    pair = module.Disagreement(1, position, (1, 2))
    search = SimpleNamespace(training=TrainingConfig(device="cpu"))
    monkeypatch.setattr(module, "_replay_positions", lambda *args, **kwargs: {})
    observed = []
    monkeypatch.setattr(module, "_play_replayed_continuations", lambda *args, **kwargs: observed.append(kwargs))
    module._branches(search, [pair], {0: [0]}, 0, 0, tmp_path)
    module._branches(search, [pair], {0: [0]}, 0, 0.5, tmp_path)
    assert observed[0]["opponent_rngs"] is None and observed[0]["mcts_value_scale"] == 0
    assert len(observed[1]["opponent_rngs"]) == 2


def test_source_collection_uses_the_same_search_and_opponent_value_scales(monkeypatch):
    from polybot.evaluation import value_disagreements as module

    observed = []

    def evaluate(model, opponent, training, mcts, games, **kwargs):
        observed.append(kwargs)
        return {"valid": True, "evaluated_map_seeds": kwargs["map_seeds"]}

    monkeypatch.setattr(module, "evaluate_matches_pool", evaluate)
    module._sample_source_positions(None, None, TrainingConfig(seed=100), MCTSConfig(), 2,
        num_envs=2, num_threads=1, history=0, samples_per_game=1, source_policy="mcts-zero", max_pending_leaves=2)
    assert observed[0]["model_a_uses_mcts"] and not observed[0]["model_b_uses_mcts"]
    assert observed[0]["leaf_value_scale"] == 0 and observed[0]["opponent_leaf_value_scale"] == 1
    assert observed[0]["max_pending_leaves"] == 2


def test_resuming_old_data_cannot_silently_switch_source_policy_or_opponent_sampling():
    from dataclasses import asdict, replace
    from polybot.training.decisive_value_pairs import DecisivePairCollection, _resume_policy

    settings = DecisivePairCollection()
    saved = {"settings": asdict(settings), "model": "original"}
    saved["settings"].pop("source_policy")  # Legacy schema before source-policy selection.
    assert not _resume_policy(saved, {"settings": asdict(settings), "model": "original"})
    for changed in (replace(settings, source_policy="mcts-zero"), replace(settings, temperature=0)):
        with pytest.raises(ValueError, match="other checkpoints or settings"):
            _resume_policy(saved, {"settings": asdict(changed), "model": "original"})
