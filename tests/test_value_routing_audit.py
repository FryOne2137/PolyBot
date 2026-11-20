import json

import numpy as np
import torch
from torch import nn

from polybot.training.flat_value import EncodedValuePosition
from polybot.training.value_routing_audit import audit_routing


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.value_head = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.value_head.weight)


def dataset():
    lookup = {}

    def pairs(base):
        result = []
        for index, agents in enumerate(((0, 0), (0, 1), (1, 1))):
            key = (base + index, 0, index)
            lookup[key] = []
            for branch, agent in enumerate(agents):
                # Root values prefer the first branch. Opponent-perspective
                # terminal targets must be sign-flipped back at comparison.
                sign = 1 if agent == 0 else -1
                root_target = 0.5 if branch == 0 else -0.5
                result.append(EncodedValuePosition(np.array([1 if branch == 0 else -1], dtype=np.float32),
                    0, sign * root_target, key, "opening", sign))
                lookup[key].append({"agent_id": agent, "opponent_value": 0.0 if agent else None,
                    "turn_changed": bool(agent), "agent_source": "saved", "root_role_disagrees_with_agent": False})
        return result

    ranking = {"train": pairs(100), "selection": pairs(200)}
    calibration = {name: [EncodedValuePosition(np.array([sign], dtype=np.float32), 0, -float(sign),
                                               (seed, 0), "opening") for sign in (1, -1)]
                   for name, seed in (("train", 300), ("selection", 400))}
    return ranking, calibration, lookup


def inspect(model, ranking, calibration, lookup, **kwargs):
    return audit_routing(model, ranking, calibration, lookup, seed=2137, batch_size=4, batches=4,
        pair_weight=1, cache_weight=1, balance_pair_maps=True, **kwargs)


def test_routing_counts_changes_and_removes_only_frozen_opponent_gradients():
    model = Head()
    ranking, calibration, lookup = dataset()
    before = model.value_head.weight.clone()
    report, rows = inspect(model, ranking, calibration, lookup)
    counts = report["splits"]["train"]["counts"]
    assert counts["pairs"] == 3 and counts["turn_changed_branches"] == 3
    assert counts["both_apprentice"] == counts["mixed_models"] == counts["both_opponent"] == 1
    gradients = report["training_gradients"]
    own = gradients["both_apprentice"]
    assert own["gap_gradient_change_norm"] == 0
    assert own["legacy"]["expected_phase_balanced"] == own["routed"]["expected_phase_balanced"]
    other = gradients["both_opponent"]
    assert other["legacy"]["expected_phase_balanced"]["gap_norm"] > 0
    assert other["routed"]["status"] == "no_apprentice_route"
    assert other["routed"]["expected_phase_balanced"]["gap_norm"] == 0
    assert other["routed"]["sampled_cache_conflict_fraction"] is None
    assert gradients["mixed_models"]["gap_gradient_change_norm"] > 0
    # Identical scalar predictions do not establish identical gradient flow.
    assert all(row["legacy_gap"] == row["routed_gap"] for row in rows)
    assert gradients["all"]["gap_gradient_change_norm"] > 0
    assert gradients["all"]["routed"]["sampled_cache_conflict_fraction"] == 1
    assert torch.equal(model.value_head.weight, before) and model.value_head.weight.grad is None


def test_zero_apprentice_scale_reports_absence_of_signal_instead_of_resolved_conflict():
    model = Head()
    ranking, calibration, lookup = dataset()
    report, rows = inspect(model, ranking, calibration, lookup, apprentice_scale=0)
    routed = report["training_gradients"]["all"]["routed"]
    assert routed["status"] == "no_apprentice_route"
    assert routed["expected_phase_balanced"]["gap_cache_cosine"] is None
    assert not any(row["has_apprentice_gradient_route"] for row in rows)
    json.dumps((report, rows), allow_nan=False)


def test_selection_inputs_do_not_supply_any_routing_gradient():
    model = Head()
    ranking, calibration, lookup = dataset()
    first, _ = inspect(model, ranking, calibration, lookup)
    for example in ranking["selection"] + calibration["selection"]:
        example.features[0] *= 10
    changed, _ = inspect(model, ranking, calibration, lookup)
    assert first["training_gradients"] == changed["training_gradients"]
