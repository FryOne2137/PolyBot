from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from polybot.config import ModelConfig, TrainingConfig
from polybot.data.player_context import (
    CONTEXT_SIZE, GLOBAL_SIZE, UNKNOWN_POPULATION, context_from_obs,
    flatten_player_context, polyenv_map_type,
)
from polybot.data.polyenv_features import (
    global_features_from_obs, native_batch_to_tensors, packet_to_tensors, stack_examples,
)
from polybot.infrastructure.model_io import load_model
from polybot.models.policy_value_belief import PolicyValueBeliefNet
from polybot.training.vector_teacher import _vector_tensors
from polybot.training.vector_population import _ppo_snapshot, _collate_public_observations


def config():
    return ModelConfig(d_model=16, attention_heads=4, transformer_layers=1,
                       belief_decoder_layers=1, dropout=0, policy_architecture="relational")


def forward(model, tensors, **kwargs):
    return model(tensors.map_tokens, tensors.global_features, tensors.action_categorical,
                 tensors.action_continuous, tensors.action_mask, tensors.event_features,
                 action_history=tensors.action_history, action_history_mask=tensors.action_history_mask,
                 **kwargs)


def engine_batch():
    from PolyEnv import Bardur, Imperius, VectorGameEnv
    env = VectorGameEnv(num_envs=2, map_size=11, seed=2137, players=(Bardur, Imperius),
                        num_threads=1, max_actions=64, visible_event_history=4)
    return env, env.reset(seed=2137)


def test_unknown_is_distinct_from_empty_tech_tree_and_zero_score():
    unknown = context_from_obs({"current_player": 0})
    known = context_from_obs({"current_player": 0, "own_techs": [0]*25,
                              "player_scores": [0]*16, "player_scores_mask": [1,1]+[0]*14})
    assert unknown[25] == 0 and known[25] == 1
    assert unknown[106:122].sum() == 0 and known[106:122].sum() == 2
    assert known[74:90].tolist() == [1]+[0]*15  # participation is not alive status
    assert known[58:74].tolist() == [1]+[0]*15
    assert not np.array_equal(known, unknown)


def test_map_types_are_distinct_and_legacy_fields_keep_their_meaning():
    values = [global_features_from_obs({"turn": 7, "map_size": 16, "map_type": kind})
              for kind in ("lakes", "drylands", "water_world")]
    assert all(x.shape == (GLOBAL_SIZE,) and x[7] == .5 for x in values)
    for index, row in enumerate(values):
        assert row[9+122:9+125].tolist() == [int(i == index) for i in range(3)]
    assert global_features_from_obs({"map_type": "waterworld"})[-4:].tolist() == [0,0,1,1]


def test_scalar_vector_and_mcts_metadata_match_real_engine():
    from PolyEnv import Bardur, Imperius, GameEnv, MctsPool
    env, batch = engine_batch()
    scalar = GameEnv(seed=2137, map_size=11, players=(Bardur, Imperius))
    packet = scalar.model_request_numpy()
    scalar_t = packet_to_tensors(packet, torch.device("cpu"))
    vector_t = native_batch_to_tensors(batch, torch.device("cpu"), 11, "lakes")
    torch.testing.assert_close(scalar_t.global_features[0], vector_t.global_features[0])
    pool = MctsPool(scalar, num_trees=1, num_threads=1, max_actions=64)
    leaves = pool.select_leaves()
    leaf_t = native_batch_to_tensors(leaves, torch.device("cpu"), 11, "lakes")
    torch.testing.assert_close(leaf_t.global_features[0], scalar_t.global_features[0])
    pool.abort_search()
    assert vector_t.global_features[0, 9+2] == 1
    assert vector_t.global_features[0, 9+28] == 1  # own Bardur
    assert vector_t.global_features[0, 9+39] == 2/16


def test_real_metadata_survives_ppo_snapshot_and_mixed_old_replay():
    _, batch = engine_batch()
    batch["player_scores"][0, 0] = 100000  # no int16 overflow
    snap, _ = _ppo_snapshot(batch, 0, 0)
    old = deepcopy(snap)
    old.pop("player_context")
    old["map_tokens"] = old["map_tokens"][:, :23]
    combined = _collate_public_observations([SimpleNamespace(observation=snap), SimpleNamespace(observation=old)])
    before = _vector_tensors({k: np.asarray(v)[:1] for k,v in batch.items()}, torch.device("cpu"), TrainingConfig())
    after = _vector_tensors(combined, torch.device("cpu"), TrainingConfig())
    torch.testing.assert_close(before["global_features"][0], after["global_features"][0])
    assert after["global_features"][1, 9+25] == 0  # unknown, not empty
    assert (combined["map_tokens"][1, :, 23] == UNKNOWN_POPULATION).all()


def test_joint_encoder_checkpoint_roundtrip_and_previous_version_rejected(tmp_path):
    from PolyEnv import GameEnv
    torch.manual_seed(17)
    model = PolicyValueBeliefNet(config()).eval()
    tensors = packet_to_tensors(GameEnv(seed=2137, map_size=11).model_request_numpy(), torch.device("cpu"))
    with torch.no_grad():
        expected = forward(model, tensors)
    state = model.state_dict()
    cfg = asdict(config())
    path = tmp_path / "joint.pt"
    torch.save({"model_config":cfg,"state_dict":state},path)
    loaded, cfg = load_model(path, torch.device("cpu"))
    loaded.eval()
    with torch.no_grad():
        loaded_out = forward(loaded,tensors)
    torch.testing.assert_close(loaded_out.value, expected.value,rtol=0,atol=0)
    torch.testing.assert_close(loaded_out.policy_logits, expected.policy_logits,rtol=0,atol=0)
    for a,b in zip(loaded_out.belief_logits, expected.belief_logits):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert cfg.observation_context_version == 1
    assert cfg.tile_encoder_version == 2
    torch.manual_seed(902)
    loaded_again, _ = load_model(path, torch.device("cpu"))
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, loaded_again.state_dict()[key], rtol=0, atol=0)
    old_cfg = asdict(cfg); old_cfg.pop("tile_encoder_version")
    torch.save({"model_config":old_cfg,"state_dict":state},path)
    with pytest.raises(ValueError, match="previous tile encoder"):
        load_model(path, torch.device("cpu"))
    # Other missing parameters must still be rejected.
    state.pop("value_head.0.weight")
    torch.save({"model_config":asdict(cfg),"state_dict":state},path)
    with pytest.raises(RuntimeError,match="value_head"):
        load_model(path,torch.device("cpu"))


def test_new_branches_receive_gradients_and_reject_enemy_population():
    _, batch = engine_batch()
    model = PolicyValueBeliefNet(config()).eval()
    t = native_batch_to_tensors(batch,torch.device("cpu"),11,"lakes")
    out = forward(model,t)
    objective = out.value.square().sum() + out.policy_logits[out.policy_logits.isfinite()].square().mean()
    objective.backward()
    assert model.observation_context[-1].weight.grad.abs().sum() > 0
    assert model.tile_encoder.fusion[0].weight.grad.abs().sum() > 0
    assert model.tile_encoder.embeddings[13].weight.grad.abs().sum() > 0
    assert not hasattr(model, "city_population_encoder")
    torch.nn.init.normal_(model.observation_context[-1].weight, std=.03)
    out = forward(model,t)
    changed = deepcopy(t)
    # Known foreign city population must never affect our outputs, even if a
    # malformed caller supplies its actual number in the auxiliary column.
    foreign = changed.map_tokens[0, 0]
    foreign[0], foreign[14], foreign[16], foreign[23] = 1, 2, 1, 0
    a = forward(model,changed)
    changed.map_tokens[0,0,23] = 999
    b = forward(model,changed)
    torch.testing.assert_close(a.value,b.value,rtol=0,atol=0)
    changed = deepcopy(t)
    changed.global_features[:, 9+2] = 0
    assert not torch.allclose(out.value,forward(model,changed).value)


def test_negative_owned_population_is_real_not_unknown():
    from PolyEnv import GameEnv
    model=PolicyValueBeliefNet(config()).eval()
    t=packet_to_tensors(GameEnv(seed=2137,map_size=11).model_request_numpy(),torch.device("cpu"))
    city=torch.nonzero(t.map_tokens[0,:,14]==2).flatten()[0]
    t.map_tokens[0,city,23] = -1
    a=forward(model,t).value
    t.map_tokens[0,city,23] = UNKNOWN_POPULATION
    assert not torch.allclose(a,forward(model,t).value)


def test_belief_completion_keeps_population_and_new_width():
    from PolyEnv import GameEnv
    model=PolicyValueBeliefNet(config()).eval()
    e=GameEnv(seed=2137,map_size=11)
    t=packet_to_tensors(e.model_request_numpy(),torch.device("cpu"))
    out=forward(model,t)
    result=model.sample_completed_map_batch_from_belief_logits(
        t.map_tokens,out.belief_logits,temperature=0,
        perspective_players=torch.tensor([0]),known_city_counts=torch.tensor([1]))
    assert result.shape == t.map_tokens.shape
    known=t.map_tokens[0,:,0]==1
    torch.testing.assert_close(result[0,known],t.map_tokens[0,known])


def test_per_player_context_does_not_share_own_technologies():
    from PolyEnv import SelfPlayPool
    pool=SelfPlayPool(num_envs=1,num_threads=1,max_actions=64)
    pool.reset()
    request=pool.all_player_belief_requests()
    flattened=flatten_player_context(request)
    ctx=flattened["player_context"]
    assert ctx.shape==(2,CONTEXT_SIZE)
    assert ctx[0,2] == 1 and ctx[0,3] == 0
    assert ctx[1,2] == 0 and ctx[1,3] == 1
    assert ctx[0,28] == 1 and ctx[1,27] == 1


def test_supervised_pool_replay_preserves_context_and_mixes_legacy_rows():
    from PolyEnv import SelfPlayPool
    from polybot.training.self_play_pool_train import _root_examples

    pool = SelfPlayPool(num_envs=1, num_threads=1, max_actions=64)
    request = pool.reset()
    # Full-map targets are training labels only. Use a copy of the public map
    # here: the test concerns observation persistence, not decoder supervision.
    policy = np.asarray(request["action_mask"], dtype=np.float32)
    policy /= policy.sum(axis=-1, keepdims=True)
    example = _root_examples(request, {"policy": policy}, TrainingConfig(),
                             np.asarray(request["map_tokens"]).copy())[0]
    legacy = deepcopy(example)
    legacy.global_features = legacy.global_features[:9]
    legacy.map_tokens = legacy.map_tokens[:, :23]
    legacy.full_map_target = legacy.full_map_target[:, :23]
    batch = stack_examples([example, legacy], torch.device("cpu"))
    expected = native_batch_to_tensors(request, torch.device("cpu"), 11, "lakes")
    torch.testing.assert_close(batch["global_features"][0], expected.global_features[0])
    assert batch["global_features"].shape == (2, GLOBAL_SIZE)
    assert not batch["global_features"][1, 9:].any()
    assert (batch["map_tokens"][1, :, 23] == UNKNOWN_POPULATION).all()
    assert batch["full_map_target"].shape == (2, 121, 24)


def test_native_leaf_after_turn_handoff_has_opponents_own_context():
    from PolyEnv import Bardur, Imperius, GameEnv, MctsPool

    pool = MctsPool(GameEnv(seed=2137, map_size=11, players=(Bardur, Imperius)),
                    num_trees=1, num_threads=1, max_actions=64)
    root = pool.select_leaves()
    logits = np.full(root["action_mask"].shape, -1000, dtype=np.float32)
    end_turn = (root["action_features"][..., 0] == 3) & root["action_mask"].astype(bool)
    assert end_turn.sum() == 1
    logits[end_turn] = 0
    pool.expand_and_backup(root["leaf_id"], logits, np.zeros(1, dtype=np.float32))
    leaf = pool.select_leaves()
    context = native_batch_to_tensors(leaf, torch.device("cpu"), 11, "lakes").global_features[0, 9:]
    assert leaf["state"][0, 1] == 1
    assert context[2] == 0 and context[3] == 1  # Imperius tech, not root Bardur tech
    assert context[27] == 1 and context[28] == 0  # Imperius tribe
    assert context[42] == 1  # own identity now player 1
    pool.abort_search()


@pytest.mark.parametrize("kind", ["lakes", "drylands", "water_world"])
def test_native_pool_map_type_matches_scalar_generation(kind):
    from PolyEnv import Bardur, Imperius, GameEnv, MapType, SelfPlayPool

    pool = SelfPlayPool(num_envs=1, seed=2137, map_size=11, tribes=(Bardur, Imperius),
                        map_type=polyenv_map_type(kind, MapType), num_threads=1, max_actions=64)
    request = pool.reset()
    packet = GameEnv(seed=2137, map_size=11, players=(Bardur, Imperius),
                     map_type=kind).model_request_numpy()
    np.testing.assert_array_equal(request["map_tokens"][0], packet["map_tokens"])
    native = native_batch_to_tensors(request, torch.device("cpu"), 11, kind)
    scalar = packet_to_tensors(packet, torch.device("cpu"))
    torch.testing.assert_close(native.global_features, scalar.global_features)
