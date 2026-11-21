import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from polybot.config import MCTSConfig, TrainingConfig
from polybot.evaluation.mcts_diagnostics import _ReplayPosition
from polybot.training import decisive_value_pairs as collector


def _outcome(left, right, repeat=1, **extra):
    return {"repeat": repeat, "outcomes": {
        "zero": {"score_a": left, **extra}, "soft_z": {"score_a": right, **extra},
    }}


def test_confidence_needs_repeats_even_when_sample_variance_is_zero():
    settings = collector.DecisivePairCollection()
    assert collector.paired_evidence([_outcome(0, 1)] * 4, settings)["status"] == "pending"
    confirmed = collector.paired_evidence([_outcome(0, 1)] * 8, settings)
    assert confirmed["status"] == "confirmed" and confirmed["sign_test_pvalue"] == 2 / 256
    uncertain = collector.paired_evidence([_outcome(0, 1)] * 7 + [_outcome(1, 0)], settings)
    assert uncertain["status"] == "pending"
    assert uncertain["sign_test_pvalue"] > uncertain["per_look_alpha"]
    assert collector.paired_evidence([_outcome(0, 1)] * 13 + [_outcome(1, 0)] * 3, settings)["status"] == "uncertain_at_limit"
    assert collector.paired_evidence([_outcome(0, 1)] * 14 + [_outcome(1, 0)] * 2, settings)["status"] == "confirmed"
    assert collector.paired_evidence([_outcome(0, 0)] * 8, settings)["status"] == "no_observed_difference"


def test_incomplete_or_truncated_branches_never_count_as_paired_labels():
    settings = collector.DecisivePairCollection()
    rows = [_outcome(0, 1)] * 4 + [_outcome(0, 1, truncated=True)] * 4 + [_outcome(None, 1)] * 4
    evidence = collector.paired_evidence(rows, settings)
    assert evidence["completed_paired_repeats"] == 4 and evidence["status"] == "pending"
    with pytest.raises(ValueError, match="terminal scores"):
        collector.paired_evidence([_outcome(float("nan"), 1)], settings)


@pytest.mark.parametrize("group_failure", [False, True])
def test_collector_resumes_after_deadline_and_enforces_reserved_maps_and_map_quota(tmp_path, monkeypatch, group_failure):
    clock = [0.0]
    advance = [3601.0]
    source_calls, branch_calls = [], []
    model = torch.nn.Linear(1, 1)

    def source(model, opponent, training, mcts, games, **kwargs):
        source_calls.append(training.seed)
        assert training.seed == 1002 and games == 4
        prefixes = {game: [5, 6, 7] for game in range(4)}
        return [_ReplayPosition(game, training.seed + game // 2, game % 2, 1, f"root-{game}")
                for game in range(4)], prefixes, {}

    def compare(search, batch, prefixes, model, **kwargs):
        assert not kwargs["capture_leaves"]
        return {"belief_zero": [{"action_id": 10} for _ in batch],
                "belief_soft_z": [{"action_id": 11} for _ in batch]}, {}, np.zeros((len(batch), 2, 1, 1, 24))

    def branches(search, batch, prefixes, repeat, temperature, output, **kwargs):
        assert kwargs["save_missing_successors"] and temperature == 0.5
        if group_failure and len(batch) > 1:
            raise collector.BeliefRecoveryRequired([0], "isolate the failed group")
        branch_calls.extend((pair.pair_id, repeat) for pair in batch)
        for pair in batch:
            for arm in ("zero", "soft_z"):
                np.savez_compressed(output / f"pair-{pair.pair_id:03d}-{arm}-successor.npz",
                                    to_play=np.array(pair.position.player))
        clock[0] += advance[0]
        return [row for pair in batch for row in ({"score_a": 0, "truncated": False}, {"score_a": 1, "truncated": False})]

    monkeypatch.setattr(collector.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(collector, "_sample_source_positions", source)
    monkeypatch.setattr(collector, "_compare_probes", compare)
    monkeypatch.setattr(collector, "_branches", branches)
    monkeypatch.setattr(collector, "RoutedPoolSearch", lambda *args, **kwargs: SimpleNamespace())
    settings = collector.DecisivePairCollection(target=2, seed=1000, source_games=4,
                                                source_batch_games=4, max_pairs_per_map=1, hours=1)
    options = dict(output=tmp_path, settings=settings, excluded_map_seeds={1000, 1001, 2000},
                   pool=dict(num_envs=4, num_threads=1, max_pending_leaves=1, visible_action_history=0))
    partial = collector.collect_decisive_value_pairs(model, model, TrainingConfig(device="cpu"), MCTSConfig(), **options)
    assert partial["reason"] == "collection_time_limit" and partial["confirmed_pairs"] == 0
    assert partial["attempted_paired_repeats"] == (1 if group_failure else 2)
    # Simulate a terminated writer leaving an incomplete trailing JSON record.
    with (tmp_path / "rollouts.jsonl").open("a") as stream:
        stream.write('{"pair_id":')
    clock[0], advance[0] = 0.0, 0.0
    finished = collector.collect_decisive_value_pairs(model, model, TrainingConfig(device="cpu"), MCTSConfig(), **options)
    assert finished["reason"] == "target_reached" and finished["confirmed_pairs"] == finished["confirmed_maps"] == 2
    assert source_calls == [1002] and finished["reserved_map_seeds"] == [1002, 1003]
    assert len(branch_calls) == len(set(branch_calls))
    labels = [json.loads(line) for line in (tmp_path / "labelled_pairs.jsonl").read_text().splitlines()]
    assert len(labels) == 2 and all(row["completed_paired_repeats"] == 8 for row in labels)
    again = collector.collect_decisive_value_pairs(model, model, TrainingConfig(device="cpu"), MCTSConfig(), **options)
    assert again["attempted_paired_repeats"] == finished["attempted_paired_repeats"]
    with pytest.raises(ValueError, match="reserved"):
        collector.collect_decisive_value_pairs(model, model, TrainingConfig(device="cpu"), MCTSConfig(),
                                              **{**options, "excluded_map_seeds": {1002}})
    with pytest.raises(ValueError, match="other checkpoints or settings"):
        collector.collect_decisive_value_pairs(model, model, TrainingConfig(device="cpu"), MCTSConfig(),
                                              **{**options, "settings": replace(settings, temperature=0.7)})


def test_journal_rejects_corruption_before_last_record(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('not json\n{"pair_id": 1}\n')
    with pytest.raises(ValueError, match="corrupt"):
        collector._journal(path)


@pytest.mark.parametrize("resume_legacy", [False, True])
def test_pooled_scouts_first_fills_batches_and_reuses_uncertain_legacy_rollouts(tmp_path, monkeypatch, resume_legacy):
    model = torch.nn.Linear(1, 1)
    source_calls, played = [], []
    pool = dict(num_envs=4, num_threads=1, max_pending_leaves=1, visible_action_history=0)
    training = TrainingConfig(device="cpu")

    def source(model, opponent, training, mcts, games, **kwargs):
        source_calls.append(training.seed)
        return [_ReplayPosition(game, training.seed + game // 2, game % 2, 1, f"r-{game}")
                for game in range(4)], {game: [5, 6, 7] for game in range(4)}, {}

    def compare(search, batch, prefixes, model, **kwargs):
        return {"belief_zero": [{"action_id": 10} for _ in batch],
                "belief_soft_z": [{"action_id": 11} for _ in batch]}, {}, np.zeros((len(batch), 2, 1, 1, 24))

    def branches(search, batch, prefixes, repeat, temperature, output, **kwargs):
        if "repeat_by_pair" in kwargs:
            assert len(source_calls) >= (1 if resume_legacy else 2)
            assert len(batch) == 2
        for pair in batch:
            index = kwargs.get("repeat_by_pair", {}).get(pair.pair_id, repeat)
            played.append((pair.pair_id, index))
            for arm in ("zero", "soft_z"):
                np.savez_compressed(output / f"pair-{pair.pair_id:03d}-{arm}-successor.npz",
                                    to_play=np.array(pair.position.player))
        return [{"score_a": 1.0, "truncated": False} for _ in range(2 * len(batch))]

    monkeypatch.setattr(collector, "_sample_source_positions", source)
    monkeypatch.setattr(collector, "_compare_probes", compare)
    monkeypatch.setattr(collector, "_branches", branches)
    monkeypatch.setattr(collector, "RoutedPoolSearch", lambda *args, **kwargs: SimpleNamespace())
    settings = collector.DecisivePairCollection(target=4, seed=1000, source_games=4 if resume_legacy else 8,
                                               source_batch_games=4, mode="confirmed" if resume_legacy else "pooled",
                                               candidate_pool=8, screening_rollouts=2)
    options = dict(output=tmp_path, settings=settings, excluded_map_seeds={2000}, pool=pool)
    if resume_legacy:
        previous = collector.collect_decisive_value_pairs(model, model, training, MCTSConfig(), **options)
        assert previous["confirmed_pairs"] == 0 and previous["completed_paired_repeats"] == 32
        original = {name: (tmp_path / name).read_bytes() for name in ("pairs.jsonl", "rollouts.jsonl", "labelled_pairs.jsonl")}
        config = json.loads((tmp_path / "config.json").read_text())
        for name in ("mode", "candidate_pool", "screening_rollouts", "scout_fraction"):
            config["collector_identity"]["settings"].pop(name)
        collector._atomic_json(tmp_path / "config.json", config)
        options["settings"] = replace(settings, mode="pooled")
    report = collector.collect_decisive_value_pairs(model, model, training, MCTSConfig(), **options)
    assert report["usable_pairs"] == 4 and report["confirmed_pairs"] == 0 and report["reached_target"]
    assert report["mean_pairs_per_rollout_batch"] == 2
    assert len(played) == len(set(played))
    labels = [json.loads(line) for line in (tmp_path / "weighted_pairs.jsonl").read_text().splitlines()]
    assert len(labels) == 4 and all(0 < row["gap_weight"] < 1 and row["soft_z_minus_zero"] == 0 for row in labels)
    if resume_legacy:
        assert (tmp_path / "legacy-config.json").is_file()
        assert all((tmp_path / name).read_bytes() == content for name, content in original.items())
        assert report["imported"]["paired_repeats"] == 32
    else:
        assert source_calls == [1000, 1002] and report["completed_paired_repeats"] == 8


def test_failed_source_maps_are_journalled_and_consume_a_bounded_reserved_budget(tmp_path, monkeypatch):
    calls = []
    model = torch.nn.Linear(1, 1)

    def source(model, opponent, training, mcts, games, **kwargs):
        assert kwargs["isolate_failed_pairs"] and kwargs["deadline"] > 0
        calls.append(training.seed)
        report = {"valid": False, "failed_map_seeds": [training.seed], "source_evaluation_attempts": [{
            "map_seeds": [training.seed], "planned_games": games, "valid": False,
            "invalid_reason": "belief failed", "belief_failure_details": {"message": "bad particle", "moves": [58]}}]}
        return [], {}, report

    monkeypatch.setattr(collector, "_sample_source_positions", source)
    settings = collector.DecisivePairCollection(target=2, seed=1000, source_games=4,
        source_batch_games=2, mode="pooled", source_policy="mcts-zero")
    report = collector.collect_decisive_value_pairs(model, model, TrainingConfig(device="cpu"), MCTSConfig(),
        output=tmp_path, settings=settings, excluded_map_seeds={2000},
        pool=dict(num_envs=4, num_threads=1, max_pending_leaves=1, visible_action_history=0))
    assert calls == [1000, 1001] and report["reason"] == "source_game_budget_exhausted"
    assert report["failed_source_batches"] == report["failed_source_maps"] == 2
    assert report["source_evaluation_games"] == 4 and report["usable_pairs"] == 0
    assert report["reserved_map_seeds"] == [1000, 1001]
    journal = [json.loads(line) for line in (tmp_path / "source-evaluations.jsonl").read_text().splitlines()]
    assert len(journal) == 2 and journal[1]["map_seeds"] == [1001]
    assert journal[0]["belief_failure_details"]["moves"] == [58]


def test_rollouts_only_resume_reuses_existing_candidates_without_new_source_games(tmp_path, monkeypatch):
    import hashlib

    model = torch.nn.Linear(1, 1)
    pool = dict(num_envs=4, num_threads=1, max_pending_leaves=1, visible_action_history=0)
    settings = collector.DecisivePairCollection(target=2, seed=1000, source_games=8, source_batch_games=4,
        mode="pooled", candidate_pool=2, min_rollouts=2, max_rollouts=2, screening_rollouts=2)

    def source(model, opponent, training, mcts, games, **kwargs):
        return [_ReplayPosition(game, training.seed + game // 2, game % 2, 1, f"root-{game}")
                for game in range(2)], {game: [1, 2] for game in range(2)}, {}

    monkeypatch.setattr(collector, "_sample_source_positions", source)
    monkeypatch.setattr(collector, "_compare_probes", lambda search, batch, *args, **kwargs:
        ({"belief_zero": [{"action_id": 10} for _ in batch], "belief_soft_z": [{"action_id": 11} for _ in batch]},
         {}, np.zeros((len(batch), 2, 1, 1, 24))))
    monkeypatch.setattr(collector, "RoutedPoolSearch", lambda *args, **kwargs: SimpleNamespace())

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("stop before the first rollout")

    monkeypatch.setattr(collector, "_branches", interrupted)
    options = dict(output=tmp_path, settings=settings, excluded_map_seeds={2000}, pool=pool)
    with pytest.raises(KeyboardInterrupt):
        collector.collect_decisive_value_pairs(model, model, TrainingConfig(device="cpu"), MCTSConfig(), **options)
    paths = [tmp_path / "pairs.jsonl", *tmp_path.glob("*-particles.npz")]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

    def unexpected(*args, **kwargs):
        raise AssertionError("resume must not create source games or new probes")

    played = []

    def branches(search, batch, prefixes, repeat, temperature, output, **kwargs):
        for pair in batch:
            played.append((pair.pair_id, kwargs["repeat_by_pair"][pair.pair_id]))
            for arm in ("zero", "soft_z"):
                np.savez_compressed(output / f"pair-{pair.pair_id:03d}-{arm}-successor.npz", to_play=pair.position.player)
        return [{"score_a": 1.0, "truncated": False} for _ in range(2 * len(batch))]

    monkeypatch.setattr(collector, "_sample_source_positions", unexpected)
    monkeypatch.setattr(collector, "_compare_probes", unexpected)
    monkeypatch.setattr(collector, "_branches", branches)
    report = collector.collect_decisive_value_pairs(model, model, TrainingConfig(device="cpu"), MCTSConfig(),
                                                    **options, resume_rollouts_only=True)
    assert report["resume_rollouts_only"] and report["usable_pairs"] == report["imported"]["candidates"] == 2
    assert report["reason"] == "saved_candidates_complete"
    assert report["completed_paired_repeats"] == 4 and len(played) == len(set(played)) == 4
    assert report["reserved_map_seeds"] == [1000, 1001] and report["source_games"] == 4
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def test_adaptive_resume_reuses_old_roots_and_rollouts_and_adds_diverse_pairs(tmp_path, monkeypatch):
    import hashlib
    from polybot.training.value_pair_sampling import AdaptivePairSampling

    model = torch.nn.Linear(1, 1)
    settings = collector.DecisivePairCollection(target=2, seed=1000, source_games=4, source_batch_games=4,
        mode="pooled", candidate_pool=8, min_rollouts=4, max_rollouts=8, screening_rollouts=4)
    pool = dict(num_envs=4, num_threads=1, max_pending_leaves=1, visible_action_history=0)
    stage, played = ["legacy"], []

    def source(*args, **kwargs):
        return [_ReplayPosition(game, 1000 + game // 2, game % 2, 1, f"root-{game}") for game in (0, 2)], {
            0: [1, 2], 2: [1, 2]}, {}

    def compare(search, batch, *args, **kwargs):
        result = {"belief_zero": [{"action_id": 10} for _ in batch],
                  "belief_soft_z": [{"action_id": 11 if stage[0] == "legacy" else 13} for _ in batch]}
        if kwargs.get("action_types"):
            result["public_action_candidates"] = [[{"action_id": action, "action_type": kind,
                "raw_probability": 0.5} for action, kind in ((10, 0), (11, 0), (12, 1))] for _ in batch]
        return result, {}, np.zeros((len(batch), 2, 1, 1, 24))

    def branches(search, batch, prefixes, repeat, temperature, output, **kwargs):
        outcomes = []
        for pair in batch:
            index = kwargs["repeat_by_pair"][pair.pair_id]
            played.append((pair.pair_id, index))
            for arm in ("zero", "soft_z"):
                path = output / f"pair-{pair.pair_id:03d}-{arm}-successor.npz"
                if not path.exists():
                    np.savez_compressed(path, to_play=pair.position.player)
            left = 1 if stage[0] == "legacy" or pair.pair_id == 2 else 0
            outcomes.extend([{"score_a": left, "truncated": False}, {"score_a": 1, "truncated": False}])
        return outcomes

    monkeypatch.setattr(collector, "_sample_source_positions", source)
    monkeypatch.setattr(collector, "_compare_probes", compare)
    monkeypatch.setattr(collector, "_branches", branches)
    monkeypatch.setattr(collector, "RoutedPoolSearch", lambda *args, **kwargs: SimpleNamespace())
    options = dict(output=tmp_path, settings=settings, excluded_map_seeds={2000}, pool=pool)
    first = collector.collect_decisive_value_pairs(model, model, TrainingConfig(device="cpu"), MCTSConfig(), **options)
    assert first["usable_pairs"] == 2 and first["usable_informative_pairs"] == 0
    old_metadata, old_rollouts = (tmp_path / "pairs.jsonl").read_bytes(), (tmp_path / "rollouts.jsonl").read_bytes()
    paths = [*tmp_path.glob("*-particles.npz"), *tmp_path.glob("*-successor.npz")]
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    stage[0] = "adaptive"
    monkeypatch.setattr(collector, "_sample_source_positions", lambda *args, **kwargs: pytest.fail("source budget already spent"))
    report = collector.collect_decisive_value_pairs(model, model, TrainingConfig(device="cpu"), MCTSConfig(),
        **options, adaptive_sampling=AdaptivePairSampling(zero_budget=4, zero_fraction=1, source_interval=1, saved_root_probes=2))
    assert report["candidates"] == 4 and report["usable_pairs"] == 4
    assert report["zero_recheck_attempts"] == report["zero_recheck_reserved"] == 4
    assert report["usable_informative_pairs"] == 3
    assert (tmp_path / "pairs.jsonl").read_bytes().startswith(old_metadata)
    assert (tmp_path / "rollouts.jsonl").read_bytes().startswith(old_rollouts)
    assert hashes == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert len(played) == len(set(played))
    added = [json.loads(line) for line in (tmp_path / "pairs.jsonl").read_text().splitlines()][2:]
    assert all(row["proposal_source"] == "public_action_diversity" and row["choices"]["soft_z"]["action_id"] == 12 for row in added)
