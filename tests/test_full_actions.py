import numpy as np
import pytest
import torch

from polybot.config import MCTSConfig, ModelConfig, TrainingConfig
from polybot.data.full_actions import (
    compact_actions, exact_vector_env, exact_mcts_pool, exact_self_play_pool,
    require_complete_actions, merge_action_batches,
)
from polybot.data.polyenv_features import native_batch_to_tensors
from polybot.models.policy_value_belief import PolicyValueBeliefNet


def tiny_model():
    return PolicyValueBeliefNet(ModelConfig(d_model=16, transformer_layers=1, attention_heads=4,
        belief_decoder_layers=1, dropout=0, policy_architecture="relational",
        action_encoder_dim=16, action_encoder_heads=4, action_latent_count=4)).eval()


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


def test_routed_search_runs_multiple_masked_waves_with_all_actions():
    from polybot.training.routed_self_play_pool import RoutedPoolSearch
    search = RoutedPoolSearch(
        TrainingConfig(device="cpu"), MCTSConfig(simulations=3, particles=1, native_max_actions=1),
        [tiny_model()], num_envs=1, num_threads=1, max_pending_leaves=2,
        visible_action_history=2, belief_retries=0, seed=2137,
    )
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
    search.pool.step_into_checked(root["action_id"][:, 0].copy(), root["state_id"].copy(), request)
    require_complete_actions(request)


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
