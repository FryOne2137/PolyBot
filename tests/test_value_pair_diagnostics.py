import numpy as np
import pytest
import torch
from torch import nn

from polybot.training.flat_experiments import FlatImprovements
from polybot.training.flat_value import EncodedValuePosition, fit_paired_value_heads
from polybot.training.value_pair_diagnostics import inspect_value_pairs


class ToyValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.value_head = nn.Sequential(nn.Linear(2, 1), nn.Tanh())
        with torch.no_grad():
            self.value_head[0].weight.zero_()
            self.value_head[0].bias.zero_()


def test_pair_sensitivity_respects_opposite_player_perspectives():
    model = ToyValueNet()
    features = np.array([1.0, 0.0], dtype=np.float32)
    pairs = [
        EncodedValuePosition(features, 0, -1.0, (2137, 0, 1), "opening", 1),
        EncodedValuePosition(features, 0, 1.0, (2137, 0, 1), "opening", 1),
        EncodedValuePosition(features, 0, 1.0, (2138, 0, 2), "opening", 1),
        EncodedValuePosition(features, 0, 1.0, (2138, 0, 2), "opening", -1),
    ]
    before = {key: value.clone() for key, value in model.state_dict().items()}
    summary, rows = inspect_value_pairs(model, pairs, difference_weight=1)
    assert summary["identical_features_with_conflicting_gap"] == 1
    assert summary["informative_pairs_with_zero_sensitivity"] == 1
    assert rows[0]["delta_loss_gradient_norm"] == rows[0]["normalized_gap_sensitivity_gradient_norm"] == 0
    assert rows[1]["delta_loss_gradient_norm"] > 0 and not rows[1]["identical_features_with_conflicting_gap"]
    assert summary["gradients"]["delta_gradient_norm"] > 0
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_delta_only_fit_learns_ranking_and_keeps_shared_minibatches():
    model = ToyValueNet()
    before = {key: value.clone() for key, value in model.state_dict().items()}

    def examples(seed):
        return [
            EncodedValuePosition(np.array([sign, 0.0], dtype=np.float32), 0,
                                 -0.5 * sign, (seed + index, 0, index), "opening")
            for index in range(3) for sign in (1, -1)
        ]

    history = {name: [] for name in ("mse", "delta", "mse_delta")}
    heads, reports = fit_paired_value_heads(
        model, examples(2137), examples(3000),
        FlatImprovements(value_updates=12, value_batch_size=4, value_learning_rate=0.05),
        seed=2137, difference_weight=1.0, loss_variants=tuple(history), preserve_reference=True,
        selection_by_variant=True,
        update_callback=lambda name, step, metrics: history[name].append((step, metrics)),
    )
    assert set(heads) == set(history)
    for name, criterion in (("mse", "value_mse"), ("delta", "gap_mse"), ("mse_delta", "selection_loss")):
        assert reports[name]["selection_metric"] == criterion
        assert reports[name]["last_update"]["train"]["preference_accuracy"] == 1.0
        assert reports[name]["train_after"]["gap_rmse"] < reports[name]["train_before"]["gap_rmse"]
        assert history[name][0][1]["train"] == history["mse"][0][1]["train"]
    assert history["mse"][1][1]["minibatch_loss"] == pytest.approx(0.25)
    assert history["delta"][1][1]["minibatch_loss"] == pytest.approx(0.25)
    assert history["mse_delta"][1][1]["minibatch_loss"] == pytest.approx(0.5)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
