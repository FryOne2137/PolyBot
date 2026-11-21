from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from polybot.training.flat_value import EncodedValuePosition
from polybot.training.value_pair_sampling import AdaptivePairSampling, adaptive_rollout_batch, proposal_pairs
from polybot.training.value_pair_split import persistent_pair_split


def test_uncertain_zero_pairs_receive_a_persistent_attempt_budget_and_regular_batches_are_not_starved():
    positions = {key: SimpleNamespace(pair_id=key) for key in range(1, 6)}
    results = {key: [{}] * 4 for key in positions}
    decisions = {key: {"usable": True, "status": "pending", "completed_paired_repeats": 4,
                      "soft_z_minus_zero": 0, "gap_standard_error": 0.5} for key in positions}
    settings = SimpleNamespace(max_rollouts=16, screening_rollouts=4)
    policy = AdaptivePairSampling(zero_budget=8)
    state = {"adaptive_rollout_batches": 0}
    regular = ("screening", [positions[5]])
    assert adaptive_rollout_batch(positions, results, decisions, settings, 2, policy, state, regular) == regular
    state["adaptive_rollout_batches"] = 3
    phase, batch = adaptive_rollout_batch(positions, results, decisions, settings, 2, policy, state, regular)
    assert phase == "zero_recheck" and [pair.pair_id for pair in batch] == [1, 2]
    assert state["zero_recheck_reserved"] == 8
    restored = json.loads(json.dumps(state))
    restored["zero_recheck_attempts"] = {"1": 4, "2": 4}  # Even four failed attempts spend the budget.
    assert adaptive_rollout_batch(positions, results, decisions, settings, 2, policy, restored, regular) == regular
    assert restored["zero_recheck_reserved"] == 8
    state["zero_recheck_attempts"] = {"1": 1}
    decisions[1]["completed_paired_repeats"] = 5
    decisions[1]["soft_z_minus_zero"] = 0.2
    assert adaptive_rollout_batch(positions, results, decisions, settings, 2, policy, state, regular)[1][0].pair_id == 1


def test_diverse_proposals_exist_even_if_both_mcts_variants_choose_the_same_action():
    choices = {"zero": {"action_id": 7}, "soft_z": {"action_id": 7}}
    alternatives = [{"action_id": 7, "action_type": 0, "raw_probability": 0.9},
                    {"action_id": 8, "action_type": 1, "raw_probability": 0.2},
                    {"action_id": 9, "action_type": 0, "raw_probability": 0.8}]
    assert proposal_pairs(choices, alternatives, adaptive=False) == []
    result = proposal_pairs(choices, alternatives, adaptive=True)
    assert result[0][0]["soft_z"]["action_id"] == 8 and result[0][1] == "public_action_diversity"
    choices["soft_z"]["action_id"] = 8
    result = proposal_pairs(choices, alternatives, adaptive=True)
    keys = [tuple(sorted(row[arm]["action_id"] for arm in ("zero", "soft_z"))) for row, _ in result]
    assert keys[0] == (7, 8) and len(keys) == len(set(keys))


def test_persistent_split_does_not_move_old_maps_when_labels_change_or_pairs_are_added(tmp_path):
    def pairs(seeds):
        return [EncodedValuePosition(np.zeros(1, dtype=np.float32), 0, float(branch), (seed, 0, seed), "opening")
                for seed in seeds for branch in (0, 1)]

    rows = pairs(range(100, 108))
    manifest = tmp_path / "value-split.json"
    _, first = persistent_pair_split(rows, seed=81, selection_fraction=0.25, path=manifest)
    old = json.loads(manifest.read_text())["map_roles"].copy()
    changed = [replace(row, return_target=0) for row in rows] + pairs(range(108, 116))
    datasets, second = persistent_pair_split(changed, seed=81, selection_fraction=0.25, path=manifest)
    assert {key: value for key, value in json.loads(manifest.read_text())["map_roles"].items() if key in old} == old
    for name in ("train", "selection"):
        assert set(first[name]["map_seeds"]) <= set(second[name]["map_seeds"])
    assert not {row.game_key[0] for row in datasets["train"]} & {row.game_key[0] for row in datasets["selection"]}
    with pytest.raises(ValueError, match="split settings"):
        persistent_pair_split(changed, seed=82, selection_fraction=0.25, path=manifest)


def test_probe_alternatives_use_public_legal_actions_and_include_both_search_choices(monkeypatch):
    from polybot.evaluation import value_disagreements as module

    request = {"agent_id": np.array([0]), "action_id": np.array([[1, 2, 3, 99]]),
               "action_mask": np.array([[1, 1, 1, 0]]), "action_features": np.zeros((1, 4, 17), dtype=np.int32)}
    request["action_features"][0, :, 0] = [0, 0, 1, 2]
    monkeypatch.setattr(module, "_replay_positions", lambda *args, **kwargs: request)

    def compare(search, packet, **kwargs):
        kwargs["observe_particles"](np.zeros((1, 2, 1, 1, 24)))
        return {"belief_zero": [{"action_id": 2}], "belief_soft_z": [{"action_id": 3}]}

    monkeypatch.setattr(module, "_compare_leaf_scales", compare)
    search = SimpleNamespace(direct_policy_value=lambda *args: (np.array([[0.8, 0.15, 0.05, 1.0]]), np.zeros(1)))
    result, _, _ = module._compare_probes(search, [None], {}, None, capture_leaves=False, action_types=12)
    rows = result["public_action_candidates"][0]
    assert {row["action_id"] for row in rows} == {1, 2, 3}
    assert {row["action_type"] for row in rows} == {0, 1}
