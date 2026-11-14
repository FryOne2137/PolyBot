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
