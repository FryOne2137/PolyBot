import numpy as np
import pytest
import torch

from polybot.config import MCTSConfig, ModelConfig, TrainingConfig
from polybot.data.full_actions import (
    compact_actions, exact_vector_env, exact_mcts_pool, exact_self_play_pool,
    require_complete_actions, merge_action_batches,
)
from polybot.data.polyenv_features import native_batch_to_tensors
from polybot.data.token_schema import MAP_FEATURE_MAX
from polybot.models.policy_value_belief import PolicyValueBeliefNet


def tiny_model():
    return PolicyValueBeliefNet(ModelConfig(d_model=16, transformer_layers=1, attention_heads=4,
        belief_decoder_layers=1, dropout=0, policy_architecture="relational",
        action_encoder_dim=16, action_encoder_heads=4, action_latent_count=4)).eval()


@pytest.mark.parametrize("temperature", [0.0, 0.15])
def test_belief_samplers_choose_an_enemy_owner_for_hidden_units(temperature):
    model = tiny_model()
    observed = torch.full((2, 9, 24), -1, dtype=torch.long)
    observed[..., 0] = 0
    observed[:, 0, 0] = 1
    observed[:, 0, 4] = 1
    observed[:, 0, 3] = torch.arange(2)
    logits = [torch.full((2, 9, maximum + 2), -100.0) for maximum in MAP_FEATURE_MAX]
    for values in logits:
        values[..., 0] = 0.0
    logits[4][..., 2] = 10.0
    logits[19][..., 3] = 10.0
    logits[3][0, :, 1] = 10.0
    logits[3][1, :, 2] = 10.0
    batched = model.sample_completed_map_batch_from_belief_logits(
        observed, logits, temperature=temperature, perspective_players=torch.arange(2)
    )
    for player in range(2):
        scalar = model.sample_completed_maps_from_belief_logits(
            observed[player:player + 1],
            [values[player:player + 1] for values in logits],
            particles=1, temperature=temperature, perspective_player=player,
        )[0]
        for particle in (scalar, batched[player]):
            assert torch.equal(particle[0], observed[player, 0])
            hidden_units = (particle[:, 0] == 0) & (particle[:, 4] >= 1)
            assert hidden_units.any()
            assert (particle[hidden_units, 3] == 1 - player).all()


@pytest.mark.parametrize(
    "terrain,settlement", [(1, 0), (2, 1)],
    ids=["all-hidden-water", "all-hidden-villages"],
)
def test_belief_samplers_reserve_a_center_for_a_referenced_city(terrain, settlement):
    from PolyEnv import GameEnv

    seed = 730184220
    env = GameEnv(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(70):
        action_ids = env.model_request_numpy()["actions"]["action_id"]
        assert env.step_fast(int(rng.choice(action_ids)))[0]
    observed = torch.as_tensor(env.model_request_numpy()["map_tokens"], dtype=torch.long)
    player = env.current_player()
    visible = observed[:, 0] == 1
    public = observed[visible]
    references = set(public[:, 6].tolist()) | set(public[:, 22].tolist())
    missing = (references - {-1}) - set(public[public[:, 14] == 2, 15].tolist())
    assert len(missing) == 1
    city_id = missing.pop()

    model = tiny_model()
    logits = [torch.full((1, len(observed), maximum + 2), -10.0) for maximum in MAP_FEATURE_MAX]
    for values in logits:
        values[..., 0] = 10.0
    logits[19][..., terrain + 1] = 20.0
    logits[14][..., settlement + 1] = 20.0
    scalar, = model.sample_completed_maps_from_belief_logits(
        observed[None], logits, 1, temperature=0.0, perspective_player=player
    )
    batched, = model.sample_completed_map_batch_from_belief_logits(
        observed[None], logits, temperature=0.0, perspective_players=torch.tensor([player])
    )

    for particle in (scalar, batched):
        assert torch.equal(particle[visible], public)
        center = ~visible & (particle[:, 14] == 2) & (particle[:, 15] == city_id)
        assert center.sum().item() == 1
        assert int(particle[center, 16].item()) == 1 - player
        assert int(particle[center, 19].item()) == 2
        assert (particle[~visible & ~center, 14] == settlement).all()
        env.make_belief_env(np.ascontiguousarray(particle, dtype=np.int32), perspective=player)


def test_full_vector_packet_survives_capacity_one_and_keeps_old_observations():
    from PolyEnv import VectorGameEnv, GameEnv
    vector = exact_vector_env(VectorGameEnv, num_envs=1, map_size=11, seed=2137, num_threads=1, max_actions=1)
    batch = vector.reset(seed=2137)
    scalar = GameEnv(seed=2137).model_request_numpy()
    assert batch["action_mask"].sum() == len(scalar["actions"]["action_id"]) == 14
    assert batch["action_mask"].shape == (1, 14)
    assert not batch["action_truncated"].any()
    original = batch["map_tokens"].copy()
    for _ in range(3):
        vector.step(batch["action_id"][:, 0].copy())
    np.testing.assert_array_equal(batch["map_tokens"], original)
    assert vector.peak_legal_actions >= 14


def test_incomplete_native_packet_is_rejected_before_inference():
    from PolyEnv import VectorGameEnv
    raw = VectorGameEnv(num_envs=1, map_size=11, seed=2137, max_actions=1, num_threads=1).reset(seed=2137)
    assert raw["total_legal_action_count"][0] == 14
    with pytest.raises(RuntimeError, match="incomplete legal action"):
        native_batch_to_tensors(raw, torch.device("cpu"), 11, "lakes")


def test_compaction_keeps_holes_and_all_legal_ids_and_merges_variable_widths():
    packet = {"action_mask": np.array([[1, 0, 0, 0, 1, 0]], dtype=np.uint8),
              "action_id": np.arange(6)[None], "total_legal_action_count": np.array([2])}
    compact = compact_actions(packet)
    assert compact["action_id"].shape == (1, 5)
    assert compact["action_id"][compact["action_mask"].astype(bool)].tolist() == [0, 4]
    small = {"action_mask": np.array([[1]], dtype=np.uint8), "action_id": np.array([[7]]),
             "total_legal_action_count": np.array([1])}
    merged = merge_action_batches([compact, small])
    assert merged["action_mask"].shape == (2, 5)
    assert merged["action_mask"].sum(axis=1).tolist() == [2, 1]
    require_complete_actions(merged)


def test_full_mcts_leaf_and_backup_work_with_compact_policy_output():
    from PolyEnv import MctsPool, GameEnv
    pool = exact_mcts_pool(MctsPool, GameEnv(seed=2137), max_actions=1, num_threads=1)
    leaves = pool.select_leaves()
    assert leaves["action_mask"].shape == (1, 14)
    t = native_batch_to_tensors(leaves, torch.device("cpu"), 11, "lakes")
    with torch.no_grad():
        output = tiny_model()(t.map_tokens, t.global_features, t.action_categorical,
                              t.action_continuous, t.action_mask, include_belief=False)
    pool.expand_and_backup(leaves["leaf_id"], output.policy_logits.numpy(), output.value.numpy())
    root = pool.root_policy()
    assert root["action_mask"].sum() == 14 and root["policy"].shape == (1, 14)
    pool.abort_search()
    for capacity in range(1, 8):
        pool._raw("leaf_batch_spec", capacity)
    assert len([key for key in pool._buffers if key[0] == "leaf_batch_spec"]) == 2


def exact_test_worlds(pool, state_ids):
    # Authoritative labels are a test fixture for controlling native API state,
    # never an observation or world generator used by production search.
    truth = pool.belief_targets_numpy(state_ids)
    request = pool.all_player_belief_requests()
    worlds = np.repeat(truth[:, None, None], pool.num_players, axis=1)
    for player in range(pool.num_players):
        obs = request["map_tokens"][:, player]
        visible = obs[..., 0] == 1
        worlds[:, player, 0, ..., 0] = obs[..., 0]
        worlds[:, player, 0][visible] = obs[visible]
    return np.ascontiguousarray(worlds)


@pytest.mark.parametrize("root_player", (0, 1))
@pytest.mark.parametrize("turn_changes", [(False,), (True,), (True, False), (True, True), (False, True)])
def test_native_mcts_backup_respects_leaf_player(root_player, turn_changes):
    from PolyEnv import SelfPlayPool

    pool = exact_self_play_pool(SelfPlayPool, num_envs=1, seed=2137, num_threads=1)
    request = pool.reset(seed=2137)
    if root_player:
        ids = request["state_id"].copy()
        pool.submit_beliefs(ids, exact_test_worlds(pool, ids))
        pool.step_into_checked(np.array([0], dtype=np.int32), ids, request)
    assert int(request["to_play"][0]) == root_player
    ids = request["state_id"].copy()
    pool.submit_beliefs(ids, exact_test_worlds(pool, ids))

    acting_player = root_player
    for switch in (*turn_changes, None):
        leaf = pool.select_leaves_masked(np.ones(1, dtype=np.uint8), max_leaves=1)
        assert len(leaf["leaf_id"]) == 1
        assert int(leaf["to_play"][0]) == acting_player
        legal = leaf["action_mask"][0].astype(bool)
        logits = np.full((1, len(legal)), -np.inf, dtype=np.float32)
        if switch is None:
            logits[0, legal] = 0.0
            value = 0.75
        else:
            target = leaf["action_id"][0] == 0 if switch else leaf["action_features"][0, :, 0] == 0
            logits[0, np.flatnonzero(legal & target)[0]] = 0.0
            value = 0.0
            acting_player ^= switch
        pool.expand_and_backup(leaf["leaf_id"], logits, np.array([value], dtype=np.float32))

    root = pool.root_policy()
    visits = int(root["root_visit_count"][0])
    assert visits == len(turn_changes) + 1
    expected_value = -0.75 if sum(turn_changes) % 2 else 0.75
    assert float(root["root_value"][0]) == pytest.approx(expected_value / visits)


@pytest.mark.parametrize("candidate_scale", [0.0, 1.0])
def test_mcts_scales_opponent_leaves_independently(candidate_scale):
    from polybot.training.routed_self_play_pool import RoutedPoolSearch

    model = tiny_model()
    with torch.no_grad():
        model.value_head[-2].weight.zero_()
        model.value_head[-2].bias.fill_(0.5)
    search = RoutedPoolSearch(
        TrainingConfig(device="cpu"), MCTSConfig(simulations=1, particles=1), [model, model],
        num_envs=2, num_threads=1, max_pending_leaves=1, visible_action_history=0,
        belief_retries=0, seed=2137, leaf_value_scale=candidate_scale, opponent_leaf_value_scale=0.25,
    )
    request = search.reset(np.full(2, 2137, dtype=np.uint32), np.array([[0, 1], [1, 0]], dtype=np.int32))
    ids = request["state_id"].copy()
    search.pool.submit_beliefs(ids, exact_test_worlds(search.pool, ids))
    leaves = {}
    count = search.pool.select_leaves_masked_into(leaves, np.ones(2, dtype=np.uint8), max_leaves=2)
    search._expand_leaves(leaves, count, prior_scale=0.0)
    np.testing.assert_allclose(search.pool.root_policy()["root_value"], np.tanh(0.5) * np.array([candidate_scale, 0.25]))


def test_mcts_value_matrix_preserves_original_schedules_and_separates_truncation(monkeypatch):
    from polybot.evaluation import mcts_diagnostics

    map_groups = (tuple(range(1287345901, 1287345907)), tuple(range(1387345901, 1387345904)))
    calls = []

    def evaluate(model_a, model_b, training, mcts, games, **kwargs):
        group = map_groups[len(calls) % 2]
        scales = (kwargs["leaf_value_scale"], kwargs["opponent_leaf_value_scale"])
        calls.append((scales, training.seed, games))
        assert kwargs["map_seeds"] == [seed for seed in group for _ in range(2)]
        assert kwargs["model_a_uses_mcts"] is True and kwargs["model_b_uses_mcts"] is False
        assert kwargs["belief_action_mode"] == "root_shield" and kwargs["replace_failed_pairs"] is False
        for index, seed in enumerate(kwargs["map_seeds"]):
            truncated = scales == (0, 1) and training.seed == map_groups[1][0] and index == 0
            kwargs["game_callback"]({
                "map_seed": seed, "model_a_player": index % 2, "moves": 100,
                "score_a": 0.5 if truncated else float(index % 2), "truncated": truncated,
            })
        return {"valid": True, "evaluated_map_seeds": kwargs["map_seeds"], "completed_games": games}

    monkeypatch.setattr(mcts_diagnostics, "evaluate_matches_pool", evaluate)
    report = mcts_diagnostics.evaluate_mcts_value_matrix(
        None, None, TrainingConfig(device="cpu"), MCTSConfig(simulations=32), map_groups,
    )
    assert calls == [
        (scales, group[0], 2 * len(group))
        for scales in ((0, 0), (0, 1), (1, 1)) for group in map_groups
    ]
    complete = report["variants"]["1,1"]
    assert (complete["wins"], complete["losses"], complete["draws"], complete["score"]) == (9, 9, 0, 0.5)
    limited = report["variants"]["0,1"]
    assert limited["score"] is None and limited["draws"] == 0 and limited["truncated_games"] == 1
    assert limited["groups"][1]["games"][0]["score_a"] is None


def test_branch_value_labels_have_the_successor_players_perspective():
    from polybot.evaluation.mcts_diagnostics import _ReplayPosition
    from polybot.training.mcts_value import _branch_examples

    position = _ReplayPosition(game=1, seed=2137, player=1, prefix_length=3, fingerprint="")
    successors = [(np.zeros(16), 1, 1), (np.zeros(16), 2, 0)]
    wins = [{"score_a": 1.0}, {"score_a": 1.0}]
    examples = _branch_examples(position, wins, successors, phase="late")
    assert [example.return_target for example in examples] == [1.0, -1.0]
    assert [example.phase for example in examples] == ["late", "late"]
    assert not _branch_examples(position, [{"score_a": None}, wins[1]], successors)
    assert not _branch_examples(position, wins, successors[:1])


def test_mcts_value_benchmark_compares_only_complete_common_map_pairs(monkeypatch):
    from polybot.training import mcts_value

    model, opponent = tiny_model(), tiny_model()
    with torch.no_grad():
        model.value_head[-2].bias.fill_(0.75)
    seen = []

    def evaluate(candidate, rival, training, mcts, games, **kwargs):
        assert candidate is model and rival is opponent and games == 4
        assert kwargs["opponent_leaf_value_scale"] == 0.25
        arm = len(seen)
        seen.append((kwargs["leaf_value_scale"], candidate.value_head[-2].bias.item()))
        seeds = kwargs["map_seeds"]
        for index, seed in enumerate(seeds):
            kwargs["game_callback"]({
                "map_seed": seed, "model_a_player": index % 2,
                "score_a": float(arm == 0), "truncated": arm == 1 and index == 3,
            })
        return {"valid": True, "evaluated_map_seeds": seeds}

    monkeypatch.setattr(mcts_value, "evaluate_matches_pool", evaluate)
    report = mcts_value.evaluate_mcts_value_heads(
        model, opponent, TrainingConfig(device="cpu", seed=2137), MCTSConfig(), 4,
        num_envs=4, num_threads=1, visible_action_history=32, max_pending_leaves=4, opponent_value_scale=0.25,
    )
    assert seen == [(1.0, 0.75), (0.0, 0.75)]
    assert model.value_head[-2].bias.item() == 0.75
    assert report["compared_map_seeds"] == [2137]
    assert report["paired_scores"] == {"fitted": 1.0, "zero": 0.0}
    assert report["beats_zero"]


def test_branch_collection_selects_disagreements_and_averages_paired_repeats(monkeypatch):
    from polybot.training import mcts_value
    from polybot.training.routed_self_play_pool import RoutedPoolSearch

    seed = 2137
    model = tiny_model()
    training = TrainingConfig(device="cpu", seed=seed, max_moves=24)
    mcts = MCTSConfig(simulations=32, particles=4)
    replay = RoutedPoolSearch(
        training, mcts, [model, model], num_envs=4, num_threads=1,
        max_pending_leaves=4, visible_action_history=32, belief_retries=0, seed=seed,
    )

    def play_games(*args, **kwargs):
        assert kwargs["leaf_value_scale"] == kwargs["opponent_leaf_value_scale"] == 0.0
        assert kwargs["model_b_uses_mcts"] is False
        seeds = np.asarray(kwargs["map_seeds"], dtype=np.uint32)
        assignments = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int32)
        request = replay.reset(seeds, assignments)
        actions = np.zeros(4, dtype=np.int32)
        for _ in range(12):
            kwargs["action_callback"](np.arange(4), request, actions)
            replay.step_public_policy(actions, request["state_id"], request)
        for index in range(4):
            kwargs["game_callback"]({"game_index": index, "truncated": index == 3})
        return {"valid": True, "evaluated_map_seeds": seeds.tolist()}

    def compare(search, request, *, include_true_map):
        assert not include_true_map and search.opponent_leaf_value_scale == 1.0
        alternatives = []
        for row, turn in enumerate(request["state"][:, 0]):
            legal = request["action_id"][row, request["action_mask"][row].astype(bool)]
            alternate = 0 if turn == 0 else int(legal[legal != 0][0])
            alternatives.append({"action_id": alternate})
        return {"belief_value_0": [{"action_id": 0}] * search.num_envs, "belief_value_1": alternatives}

    streams = {}

    def complete_branches(search, request, positions, first_actions, max_moves, *,
                          after_first_action, raw_rngs, raw_temperature):
        assert not any(position.game == 3 for position in positions)
        assert raw_temperature == 0.5
        outcomes = []
        for row in range(0, len(positions), 2):
            assert first_actions[row] == 0 and first_actions[row + 1] != 0
            first_draw, second_draw = raw_rngs[row].random(), raw_rngs[row + 1].random()
            assert first_draw == second_draw
            history = streams.setdefault((positions[row].game, positions[row].prefix_length), [])
            assert first_draw not in history
            outcomes.extend({"score_a": score} for score in ((0.0, 1.0), (1.0, 1.0), (None, 0.0))[len(history)])
            history.append(first_draw)
        search.step_public_policy(np.asarray(first_actions, dtype=np.int32), request["state_id"], request)
        assert request["action_valid"].all()
        if after_first_action is not None:
            after_first_action(request)
        return outcomes

    monkeypatch.setattr(mcts_value, "evaluate_matches_pool", play_games)
    monkeypatch.setattr(mcts_value, "_compare_leaf_scales", compare)
    monkeypatch.setattr(mcts_value, "_play_replayed_continuations", complete_branches)
    examples, report = mcts_value.collect_mcts_value_branches(
        model, model, training, mcts, games=4, positions_per_game=6,
        num_envs=4, num_threads=1, visible_action_history=32, max_pending_leaves=4,
        opponent_value_scale=1.0, disagreement_probes_per_position=2,
        branch_rollouts=3, rollout_temperature=0.5, label="test",
    )
    assert report["terminal_source_games"] == 3 and report["truncated_source_games"] == 1
    assert report["sampled_roots"] == report["labelled_pairs"] == len(streams) == 15
    assert report["completed_rollout_pairs"] == report["truncated_branches"] * 2 == 30
    assert report["terminal_branches"] == 75
    assert all(len(history) == 3 for history in streams.values())
    assert sum(example.return_target == 0.0 for example in examples) == 15
    assert sum(abs(example.return_target) == 1.0 for example in examples) == 15
    for phase, counts in report["phases"].items():
        expected = 3 if phase == "opening" else 6
        assert counts["probed_roots"] == 6
        assert counts["sampled_roots"] == counts["labelled_pairs"] == counts["pairs_with_different_mean_scores"] == expected
        indices = {"opening": range(4), "mid": range(4, 8), "late": range(8, 12)}[phase]
        assert all(example.game_key[2] in indices for example in examples if example.phase == phase)
        assert sum(example.phase == phase for example in examples) == 2 * expected


def test_branch_probe_reuses_particles_without_searching_the_true_map(monkeypatch):
    from polybot.data.full_actions import ExactActionNative
    from polybot.evaluation.mcts_diagnostics import _compare_leaf_scales
    from polybot.training.routed_self_play_pool import RoutedPoolSearch

    worlds = []

    def submit(pool, state_ids, particles):
        pool.native.submit_beliefs(state_ids, particles)
        worlds.append(particles.copy())

    monkeypatch.setattr(ExactActionNative, "submit_beliefs", submit, raising=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(21)
        model = tiny_model()
    search = RoutedPoolSearch(
        TrainingConfig(device="cpu"), MCTSConfig(simulations=32, particles=4), [model, model],
        num_envs=1, num_threads=1, max_pending_leaves=4, visible_action_history=32,
        belief_retries=4, seed=2137, belief_action_mode="root_shield", opponent_leaf_value_scale=1.0,
    )
    request = search.reset(np.array([2137], dtype=np.uint32), np.array([[0, 1]], dtype=np.int32))
    result = _compare_leaf_scales(search, request, include_true_map=False)
    assert set(result) == {"belief_value_0", "belief_value_1"}
    assert all(0 < choices[0]["root_visits"] <= 32 for choices in result.values())
    assert len(worlds) == 2
    np.testing.assert_array_equal(worlds[0], worlds[1])


def test_branch_fit_can_benchmark_an_update_with_worse_offline_rmse():
    from polybot.training.flat_experiments import FlatImprovements
    from polybot.training.flat_value import EncodedValuePosition, fit_value_head

    model = tiny_model()
    model.requires_grad_(False)
    model.value_head.requires_grad_(True)
    with torch.no_grad():
        model.value_head[-2].weight.zero_()
        model.value_head[-2].bias.zero_()
    training = TrainingConfig(device="cpu")
    settings = FlatImprovements(value_updates=2, value_batch_size=8, value_fit_backoffs=0)
    features = np.ones(model.config.d_model, dtype=np.float32)
    replay = [EncodedValuePosition(features, 0, 1.0, (seed,)) for seed in range(8)]
    validation = [EncodedValuePosition(features, 0, -1.0, (seed,)) for seed in range(8, 16)]
    baseline = fit_value_head(model, training, replay, validation, settings, seed=2137)
    assert not baseline["kept"]
    branch = fit_value_head(model, training, replay, validation, settings, seed=2137, keep_best_update=True)
    assert branch["kept"] and not branch["selection_improved"]
    assert branch["after"]["rmse"] > branch["before"]["rmse"]
    assert branch["selected_updates"] > 0


def test_mcts_diagnostics_replays_only_disagreements_on_the_same_positions():
    from polybot.evaluation.mcts_diagnostics import evaluate_mcts_diagnostics

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(21)
        model = tiny_model()
    report = evaluate_mcts_diagnostics(
        model,
        model,
        TrainingConfig(device="cpu", seed=916438271, max_moves=30),
        MCTSConfig(simulations=8, particles=4),
        games=4,
        positions_per_game=2,
        num_threads=1,
        max_pending_leaves=4,
        visible_action_history=32,
        belief_retries=4,
        max_envs=4,
        model_a_label="apprentice",
        model_b_label="champion",
    )
    assert report["diagnostics_valid"]
    assert 0 < report["compared_positions"] <= 8
    assert report["belief_mcts_evaluation"]["evaluated_map_seeds"] == [
        916438271, 916438271, 916438272, 916438272,
    ]
    for position in report["positions"]:
        assert position["raw_action"] != position["belief_action"]
        assert position["belief_visits"] > 0 and position["true_map_visits"] > 0
    assert report["terminal_positions"] == 0
    assert report["value_vs_observed_belief_game"]["raw_value"]["count"] == 0


def test_mcts_value_diagnostics_reuses_the_same_roots_and_particles(monkeypatch):
    from polybot.data.full_actions import ExactActionNative
    from polybot.evaluation.mcts_diagnostics import evaluate_mcts_value_diagnostics

    submitted_worlds = []

    def record_beliefs(pool, state_ids, worlds):
        pool.native.submit_beliefs(state_ids, worlds)
        if worlds.ndim == 5:
            submitted_worlds.append(worlds.copy())

    monkeypatch.setattr(ExactActionNative, "submit_beliefs", record_beliefs, raising=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(21)
        model = tiny_model()
    seed = 1048957319
    report = evaluate_mcts_value_diagnostics(
        model, model, TrainingConfig(device="cpu", seed=seed, max_moves=16),
        MCTSConfig(simulations=32, particles=4), games=12,
        positions_per_game=2, num_threads=2, max_pending_leaves=4,
        visible_action_history=32, belief_retries=4, max_envs=4,
        model_a_label="apprentice", model_b_label="champion",
        play_disagreement_games=True,
    )
    assert report["raw"]["valid"] and report["raw"]["self_play_pool_leaf_rows"] == 0
    assert [(game["map_seed"], game["model_a_player"]) for game in report["raw_games"]] == [
        (seed + index // 2, index % 2) for index in range(12)
    ]
    assert report["diagnostics_valid"]
    assert report["sampled_positions"] == report["compared_positions"] == 24
    assert sum(report["changes"].values()) == 24
    assert len(report["positions"]) == 24 - report["changes"]["neither"]
    assert len(submitted_worlds) == 24
    for index in range(0, len(submitted_worlds), 4):
        np.testing.assert_array_equal(submitted_worlds[index], submitted_worlds[index + 1])
        np.testing.assert_array_equal(submitted_worlds[index + 2], submitted_worlds[index + 3])
    for position in report["positions"]:
        assert position["observed_raw_game_outcome"] is None
        for mode in ("belief", "true_map"):
            for scale in (0, 1):
                choice = position[f"{mode}_value_{scale}"]
                assert 0 < choice["root_visits"] <= 32
                assert choice["action_visits"][choice["action_id"]] == choice["chosen_action_visits"]
        assert (
            position["belief_value_1"]["action_id"] != position["belief_value_0"]["action_id"]
            or position["true_map_value_1"]["action_id"] != position["true_map_value_0"]["action_id"]
        )
        expected_actions = set()
        for mode in ("belief", "true_map"):
            action_0 = position[f"{mode}_value_0"]["action_id"]
            action_1 = position[f"{mode}_value_1"]["action_id"]
            if action_0 != action_1:
                expected_actions.update((action_0, action_1))
        assert position["continuations"].keys() == expected_actions
        for outcome in position["continuations"].values():
            assert outcome["truncated"] and outcome["score_a"] is None
            assert outcome["moves_from_position"] == 17 - position["move"]
    continuation_summary = report["continuations"]
    assert continuation_summary["played_branches"] == sum(
        len(position["continuations"]) for position in report["positions"]
    )
    assert continuation_summary["terminal_branches"] == 0
    assert continuation_summary["truncated_branches"] == continuation_summary["played_branches"]
    assert continuation_summary["belief"]["pairs"] == report["changes"]["belief_only"] + report["changes"]["both"]
    assert continuation_summary["true_map"]["pairs"] == report["changes"]["true_map_only"] + report["changes"]["both"]
    for mode in ("belief", "true_map"):
        assert continuation_summary[mode]["incomplete"] == continuation_summary[mode]["pairs"]
        assert continuation_summary[mode]["mean_score_delta_value_1_minus_value_0"] is None


@pytest.mark.parametrize("use_vector", [False, True])
def test_raw_continuations_distinguish_a_terminal_win_from_the_move_limit(use_vector):
    from PolyEnv import GameEnv
    from polybot.evaluation.mcts_diagnostics import (
        _ReplayPosition, _play_continuations, _position_fingerprint,
    )
    from polybot.evaluation.public_rollouts import PublicRolloutPool
    from polybot.planning.heuristic_policy import choose_teacher_action
    from polybot.training.routed_self_play_pool import RoutedPoolSearch

    seed = 2137
    env = GameEnv(seed=seed)
    rng = np.random.default_rng(seed)
    model = tiny_model()
    search = RoutedPoolSearch(
        TrainingConfig(device="cpu"), MCTSConfig(), [model, model],
        num_envs=2, num_threads=1, max_pending_leaves=4, visible_action_history=32,
        belief_retries=0, seed=seed,
    )
    seeds = np.full(2, seed, dtype=np.uint32)
    assignments = np.array([[0, 1], [0, 1]], dtype=np.int32)
    request = search.reset(seeds, assignments)
    prefix = []
    for _ in range(1000):
        packet = env.model_request_numpy()
        action = int(packet["actions"]["action_id"][choose_teacher_action(packet, rng)])
        ok, done, _, winner, _ = env.step_fast(action)
        assert ok
        if done:
            break
        prefix.append(action)
        search.step_public_policy(np.full(2, action, dtype=np.int32), request["state_id"], request)
        assert request["action_valid"].all()
    else:
        pytest.fail("the fixture did not reach a terminal state")
    assert winner == int(request["to_play"][0]) == 0
    assert action != 0 and 0 in request["action_id"][0, request["action_mask"][0].astype(bool)]
    position = _ReplayPosition(0, seed, 0, len(prefix), _position_fingerprint(request, 0))
    if use_vector:
        search = PublicRolloutPool(
            search.training, [model, model], num_envs=2, visible_action_history=32,
        )

    win, truncated = _play_continuations(
        search, [position, position], [action, 0], {0: prefix}, max_moves=len(prefix) + 3,
    )
    assert win == {"score_a": 1.0, "winner_player": 0, "truncated": False, "moves_from_position": 1}
    assert truncated == {"score_a": None, "winner_player": None, "truncated": True, "moves_from_position": 3}


def test_sampled_value_branches_pass_a_position_rejected_by_snapshot_reconstruction():
    from polybot.evaluation.mcts_diagnostics import (
        _ReplayPosition, _play_replayed_continuations, _position_fingerprint, _replay_positions,
    )
    from polybot.evaluation.public_rollouts import PublicRolloutPool
    from polybot.training.routed_self_play_pool import RoutedPoolSearch

    seed = 1834517902
    training = TrainingConfig(device="cpu", seed=seed, max_moves=267)
    model = tiny_model()
    search = RoutedPoolSearch(
        training, MCTSConfig(), [model, model], num_envs=1, num_threads=1,
        max_pending_leaves=4, visible_action_history=32, belief_retries=0, seed=seed,
    )
    request = search.reset(np.array([seed], dtype=np.uint32), np.array([[1, 0]], dtype=np.int32))
    rng = np.random.default_rng(seed ^ 1111)
    prefix = []
    for _ in range(257):
        legal = request["action_mask"][0].astype(bool)
        ids = request["action_id"][0, legal]
        features = request["action_features"][0, legal]
        weights = np.exp(
            1.5 * (features[:, 0] == 1) + 0.5 * (features[:, 0] == 0)
            + 4 * ((features[:, 0] == 8) & np.isin(features[:, 8], (11, 14)))
        )
        action = int(rng.choice(ids, p=weights / weights.sum()))
        actions = np.array([action], dtype=np.int32)
        search.step_public_policy(actions, request["state_id"], request)
        prefix.append(action)
    action = 4527
    assert action in request["action_id"][0, request["action_mask"][0].astype(bool)]
    assert int(request["to_play"][0]) == 1
    position = _ReplayPosition(0, seed, 1, len(prefix), _position_fingerprint(request, 0))
    with pytest.raises(ValueError, match="changes the current player's legal action set"):
        search.step_public_policy(np.array([action], dtype=np.int32), request["state_id"], request)

    raw = PublicRolloutPool(training, [model, model], num_envs=2, visible_action_history=32)
    for repeat in range(3):
        request = _replay_positions(raw, [position, position], {0: prefix})
        outcomes = _play_replayed_continuations(
            raw, request, [position, position], [action, 0], training.max_moves,
            raw_rngs=[np.random.default_rng((seed, repeat)) for _ in range(2)], raw_temperature=0.5,
        )
        assert all(outcome["moves_from_position"] == 10 for outcome in outcomes)


def test_first_divergence_replays_only_opposite_winners_with_shared_mcts(monkeypatch):
    from polybot.evaluation import mcts_diagnostics as diagnostics
    from polybot.training.routed_self_play_pool import RoutedPoolSearch

    seed = 2137
    model = tiny_model()
    training = TrainingConfig(device="cpu", seed=seed, max_moves=4)
    mcts = MCTSConfig(simulations=32, particles=1)
    fixture = RoutedPoolSearch(
        training, mcts, [model, model], num_envs=1, num_threads=1,
        max_pending_leaves=4, visible_action_history=32, belief_retries=4, seed=seed,
    )
    seeds = np.array([seed], dtype=np.uint32)
    assignments = np.array([[1, 0]], dtype=np.int32)
    game_index = np.array([1], dtype=np.int64)
    evaluated_scales = []

    def play_baseline(*args, **kwargs):
        scale = kwargs["leaf_value_scale"]
        evaluated_scales.append(scale)
        assert kwargs["map_seeds"] == [seed, seed, seed + 1, seed + 1]
        assert kwargs["model_b_uses_mcts"] is False and kwargs["replace_failed_pairs"] is False
        record = kwargs["action_callback"]
        request = fixture.reset(seeds, assignments)
        end_turn = np.array([0], dtype=np.int32)
        record(game_index, request, end_turn)
        fixture.step_public_policy(end_turn, request["state_id"], request)
        assert request["agent_id"].tolist() == [0]
        moves = request["action_id"][0, request["action_features"][0, :, 0] == 0]
        record(game_index, request, np.array([moves[int(scale)]], dtype=np.int32))
        for index in (3, 1, 0, 2):
            winner = (0, 1, 1, -1)[index] if scale else 0
            kwargs["game_callback"]({
                "game_index": index, "map_seed": seed + index // 2, "model_a_player": index % 2,
                "winner_player": winner, "truncated": bool(scale and index == 2),
            })
        return {"valid": True, "evaluated_map_seeds": kwargs["map_seeds"]}

    searched = []
    evaluate = diagnostics._evaluation_search_with_recovery

    def record_search(search, request, active, **kwargs):
        assert search.mcts.simulations == 32
        np.testing.assert_array_equal(kwargs["mcts_mask"], active & (request["agent_id"] == 0))
        searched.append((search.num_envs, search.leaf_value_scale))
        return evaluate(search, request, active, **kwargs)

    monkeypatch.setattr(diagnostics, "evaluate_matches_pool", play_baseline)
    monkeypatch.setattr(diagnostics, "_evaluation_search_with_recovery", record_search)
    report = diagnostics.evaluate_mcts_first_divergence(
        model, model, training, mcts, games=4, num_threads=1,
        max_pending_leaves=4, visible_action_history=32, belief_retries=4, max_envs=4,
        model_a_label="apprentice", model_b_label="champion", continuation_value_scale=0.0,
    )
    assert evaluated_scales == [1.0, 0.0]
    assert searched and set(searched) == {(2, 0.0)}
    assert report["diagnostics_valid"]
    assert report["changed_winner_pairs"] == 1
    assert report["excluded_truncated_pairs"] == report["excluded_draw_pairs"] == 1
    position, = report["positions"]
    assert (position["seed"], position["model_a_player"], position["move"]) == (seed, 1, 2)
    assert position["actor"] == "apprentice"
    assert position["action_value_0"] != position["action_value_1"]
    assert report["continuations"]["mcts"]["leaf_value_scale"] == 0.0
    for policy in ("raw", "mcts"):
        assert report["continuations"][policy]["incomplete"] == 1
        assert all(
            branch["truncated"] and branch["moves_from_position"] == 3
            for branch in position["continuations"][policy].values()
        )


def test_full_self_play_masked_into_expands_and_checked_step_preserves_actions():
    from PolyEnv import SelfPlayPool
    pool = exact_self_play_pool(SelfPlayPool, num_envs=1, seed=2137, num_threads=1, max_actions=1)
    request = pool.reset(seed=2137)
    ids = request["state_id"].copy()
    pool.submit_beliefs(ids, exact_test_worlds(pool, ids))
    leaves = {}
    count = pool.select_leaves_masked_into(leaves, np.ones(1, dtype=np.uint8), max_leaves=1)
    assert count == 1 and leaves["action_mask"].shape == (1, 14)
    pool.expand_and_backup(leaves["leaf_id"][:count], np.zeros((count, 14), dtype=np.float32),
                           np.zeros(count, dtype=np.float32))
    root = pool.root_policy()
    pool.step_into_checked(root["action_id"][:, 0].copy(), root["state_id"].copy(), request)
    require_complete_actions(request)
    assert request["action_mask"].sum() == request["total_legal_action_count"][0]


@pytest.mark.parametrize(
    "leaf_value_scale,zero_leaf_values",
    [(1.0, False), (0.25, False), (0.0, False), (1.0, True)],
)
def test_routed_search_runs_multiple_masked_waves_with_all_actions(
    monkeypatch, leaf_value_scale, zero_leaf_values
):
    from polybot.training.routed_self_play_pool import RoutedPoolSearch

    model = tiny_model()
    with torch.no_grad():
        model.value_head[-2].weight.zero_()
        model.value_head[-2].bias.fill_(0.5)
    search = RoutedPoolSearch(
        TrainingConfig(device="cpu"), MCTSConfig(simulations=3, particles=1, native_max_actions=1),
        [model], num_envs=1, num_threads=1, max_pending_leaves=2,
        visible_action_history=2, belief_retries=0, seed=2137,
        zero_leaf_values=zero_leaf_values,
        leaf_value_scale=leaf_value_scale,
    )
    expand_and_backup = search.pool.expand_and_backup
    backed_up_values = []

    def record_backup(leaf_ids, logits, values):
        backed_up_values.append(values.copy())
        expand_and_backup(leaf_ids, logits, values)

    monkeypatch.setattr(search.pool, "expand_and_backup", record_backup)
    request = search.reset(np.array([2137], dtype=np.uint32), np.zeros((1, 2), dtype=np.int32))
    worlds = exact_test_worlds(search.pool, request["state_id"].copy())
    # Exact proposals are a controlled fixture, not a production generator.
    search._all_player_particles = lambda *args: worlds.copy()
    search._actor_particles = lambda request, rows, *args: worlds[
        rows, np.asarray(request["to_play"])[rows]].copy()
    root = search.search(request, np.ones(1, dtype=bool), root_temperature=1,
                         belief_temperature=.15, prior_scale=0)
    assert root["action_mask"].sum() == 14
    assert root["visit_count"].sum() == 2
    expected_value = 0.0 if zero_leaf_values else leaf_value_scale * np.tanh(0.5)
    np.testing.assert_allclose(np.concatenate(backed_up_values), expected_value)
    search.pool.step_into_checked(root["action_id"][:, 0].copy(), root["state_id"].copy(), request)
    require_complete_actions(request)


@pytest.mark.parametrize("active, searched", [
    ([True, True], [False, False]),  # belief calibration
    ([True, True], [False, True]),   # selective MCTS
    ([False, True], [False, True]),  # a slot has reached its move limit
])
def test_routed_unsearched_roots_keep_the_public_action_packet(active, searched):
    from polybot.training.routed_self_play_pool import RoutedPoolSearch

    search = RoutedPoolSearch(
        TrainingConfig(device="cpu"), MCTSConfig(simulations=3, particles=1, native_max_actions=1),
        [tiny_model()], num_envs=2, num_threads=1, max_pending_leaves=2,
        visible_action_history=2, belief_retries=0, seed=1732409,
    )
    request = search.reset(np.array([1732409, 1732410], dtype=np.uint32),
                           np.zeros((2, 2), dtype=np.int32))
    # The unsearched slot has the widest public action row in the batch.
    assert request["total_legal_action_count"].tolist() == [15, 14]
    worlds = exact_test_worlds(search.pool, request["state_id"].copy())
    search._all_player_particles = lambda *args: worlds.copy()
    search._actor_particles = lambda request, rows, *args: worlds[
        rows, request["to_play"][rows]].copy()
    active, searched = np.array(active), np.array(searched)
    direct = active & ~searched
    policy, value = search.direct_policy_value(request, np.flatnonzero(direct))
    root = search.search(request, active, root_temperature=1, belief_temperature=.15,
                         prior_scale=0, mcts_mask=searched, direct_policy=policy,
                         direct_value=value)

    require_complete_actions(root)
    np.testing.assert_array_equal(root["action_id"], request["action_id"])
    np.testing.assert_array_equal(root["action_mask"], request["action_mask"])
    np.testing.assert_allclose(root["policy"][direct], policy[direct])
    np.testing.assert_allclose(root["root_value"][direct], value[direct])
    np.testing.assert_array_equal(root["visit_count"].sum(axis=1), 2 * searched)
    actions = np.where(active, root["action_id"][:, 0], -1).astype(np.int32)
    search.pool.step_into_checked(actions, root["state_id"], request)
    require_complete_actions(request)
    assert request["action_valid"][active].all()


def test_belief_calibration_trains_on_public_rollouts_with_rejected_particles(
    monkeypatch, tmp_path
):
    from polybot.training.routed_self_play_pool import RoutedPoolSearch, train_population_pool
    from polybot.training.run_artifacts import TrainingRun

    monkeypatch.setattr(TrainingRun, "_write_plots", lambda self: None)

    def invalid_particles(maps, count, axis):
        particles = np.repeat(np.expand_dims(maps, axis), count, axis=axis)
        particles[..., 0] = 2  # A proposal must preserve the observed visibility mask.
        return particles

    monkeypatch.setattr(
        RoutedPoolSearch, "_all_player_particles",
        lambda self, request, count, *args: invalid_particles(request["map_tokens"], count, 2),
    )
    monkeypatch.setattr(
        RoutedPoolSearch, "_actor_particles",
        lambda self, request, rows, count, *args: invalid_particles(
            request["map_tokens"][rows], count, 1
        ),
    )
    candidate = tiny_model()
    before = {name: tensor.clone() for name, tensor in candidate.state_dict().items()}
    replay = []
    result = train_population_pool(
        TrainingConfig(device="cpu", seed=1732409, max_moves=96, batch_size=8),
        candidate.config, MCTSConfig(simulations=3, particles=2),
        games=4, updates_per_game=1, candidate=candidate,
        opponent_models=[tiny_model()], opponent_weights=[1.0],
        belief_only=True, raw_policy_only=True, offline_collection=True,
        max_envs=2, num_threads=1, visible_action_history=32,
        replay_snapshot_callback=replay.extend,
        run=TrainingRun.create(tmp_path, "belief-regression"),
    )

    assert result["population_pool_completion_rate"] == 1
    assert result["population_pool_updates"] == 4
    assert result["belief_action_equivalence_checked_particles"] > 0
    assert result["belief_action_equivalence_rejection_rate"] == 1
    assert result["belief_sampled_rejected_belief_build_error"] == result[
        "belief_action_equivalence_rejected_particles"
    ]
    assert result["belief_build_errors"] == [{
        "message": "completed_map_tokens must preserve the source visibility mask",
        "count": result["belief_action_equivalence_rejected_particles"],
        "map_seed": 1732409,
        "move": 1,
        "player": 0,
        "particle": 0,
    }]
    assert result["belief_sampled_unit_predicted"] == 0
    assert result["belief_slot_resets"] == result["population_pool_leaf_rows"] == 0
    assert replay and all(example.policy_loss_weight == 0 for example in replay)
    changed = [name for name, tensor in candidate.state_dict().items()
               if not torch.equal(before[name], tensor)]
    assert changed and all(name.startswith("belief_") for name in changed)


@pytest.mark.parametrize("phase", ["teacher", "actor_critic", "objectives"])
def test_native_vector_training_phases_learn_with_capacity_hint_one(tmp_path, monkeypatch, phase):
    from polybot.training.run_artifacts import TrainingRun
    from polybot.training.vector_teacher import train_vector_teacher
    from polybot.training.vector_actor_critic import train_vector_actor_critic
    from polybot.training.vector_objectives import train_vector_objective_curriculum
    monkeypatch.setattr(TrainingRun, "_write_plots", lambda *args: None)
    model = tiny_model()
    before = model.action_encoder.type_embedding.weight.detach().clone()
    common = dict(num_envs=2, steps=3, num_threads=1, max_actions=1, model=model,
                  run=TrainingRun.create(tmp_path, phase))
    training = TrainingConfig(device="cpu", seed=2137)
    if phase == "teacher":
        summary = train_vector_teacher(training, model.config, visible_event_history=2, **common)
    elif phase == "actor_critic":
        summary = train_vector_actor_critic(training, model.config, **common)
    else:
        summary = train_vector_objective_curriculum(training, model.config, rollout_horizon=2,
                                                     updates_per_rollout=1, **common)
    loss_key = {"teacher": "loss", "actor_critic": "flat_loss", "objectives": "objective_loss"}[phase]
    assert np.isfinite(summary[loss_key])
    assert not torch.equal(before, model.action_encoder.type_embedding.weight)
