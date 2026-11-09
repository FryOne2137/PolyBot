from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from polybot.config import ModelConfig, TrainingConfig
from polybot.data.polyenv_features import packet_to_tensors
from polybot.models.policy_value_belief import PolicyValueBeliefNet


def network(architecture="relational"):
    torch.manual_seed(123)
    return PolicyValueBeliefNet(ModelConfig(
        d_model=16, transformer_layers=1, attention_heads=4,
        belief_decoder_layers=1, dropout=0, policy_architecture=architecture,
        action_encoder_dim=16, action_encoder_heads=4, action_latent_count=4,
    )).eval()


def inputs(actions=5):
    from PolyEnv import GameEnv
    data = packet_to_tensors(GameEnv(seed=2137).model_request_numpy(), torch.device("cpu"))
    # Repeating public candidates is a tensor workload, not an engine action list.
    indexes = torch.arange(actions) % data.action_mask.shape[1]
    return [data.map_tokens, data.global_features, data.action_categorical[:, indexes],
            data.action_continuous[:, indexes], data.action_mask[:, indexes]]


@pytest.mark.parametrize("architecture", ["factorized", "attention", "relational"])
def test_shared_action_set_is_permutation_equivariant_and_padding_invariant(architecture):
    model = network(architecture)
    original = inputs()
    perm = torch.tensor([4, 2, 0, 1, 3])
    reordered = [original[0], original[1], original[2][:, perm], original[3][:, perm], original[4][:, perm]]
    a, b = model(*original), model(*reordered)
    torch.testing.assert_close(a.policy_logits[:, perm], b.policy_logits)
    torch.testing.assert_close(a.value, b.value)
    for left, right in zip(a.belief_logits, b.belief_logits):
        torch.testing.assert_close(left, right)
    padded = [*original[:2], F.pad(original[2], (0, 0, 0, 7), value=-1),
              F.pad(original[3], (0, 0, 0, 7)), F.pad(original[4], (0, 7), value=False)]
    c = model(*padded)
    torch.testing.assert_close(a.policy_logits, c.policy_logits[:, :5])
    torch.testing.assert_close(a.value, c.value)
    for left, right in zip(a.belief_logits, c.belief_logits):
        torch.testing.assert_close(left, right)
    assert torch.isneginf(c.policy_logits[:, 5:]).all()


@pytest.mark.parametrize("head", ["policy", "value", "belief"])
def test_each_head_backpropagates_into_the_same_action_encoder(head):
    model = network()
    output = model(*inputs())
    loss = {"policy": lambda: -output.policy_logits.log_softmax(-1)[0, 0],
            "value": lambda: (output.value - 1).square().mean(),
            "belief": lambda: -output.belief_logits[19].log_softmax(-1)[..., 2].mean()}[head]()
    loss.backward()
    encoder = model.action_encoder
    assert encoder.type_embedding.weight.grad.abs().sum() > 0
    assert encoder.read_actions.in_proj_weight.grad.abs().sum() > 0
    assert encoder.write_actions.in_proj_weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_all_4097_actions_are_read_without_quadratic_self_attention():
    model = network()
    seen = []
    handle = model.action_encoder.read_actions.register_forward_pre_hook(
        lambda module, args: seen.append((args[0].shape[1], args[1].shape[1])))
    data = inputs(4097)
    result = model(*data, include_belief=False)
    handle.remove()
    assert seen == [(4, 4097)]
    assert result.policy_logits.shape == (1, 4097)
    assert torch.isfinite(result.policy_logits).all()


def test_empty_action_set_has_finite_backward_and_neutral_value_context():
    model = network()
    data = inputs()
    data[4].fill_(False)
    output = model(*data)
    loss = output.value.square().mean() + sum(x.square().mean() for x in output.belief_logits)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    no_actions = [*data[:2], data[2][:, :0], data[3][:, :0], data[4][:, :0]]
    empty = model(*no_actions)
    torch.testing.assert_close(output.value, empty.value)


def test_value_fitting_uses_the_same_action_conditioned_state_as_inference():
    from polybot.training.flat_value import _states
    from polybot.training.vector_population import _ppo_snapshot
    from polybot.training.vector_teacher import _vector_tensors
    from polybot.data.full_actions import exact_vector_env
    from PolyEnv import VectorGameEnv

    env = exact_vector_env(VectorGameEnv, num_envs=1, seed=2137, num_threads=1, max_actions=1)
    batch = env.reset(seed=2137)
    observation, _ = _ppo_snapshot(batch, 0, 0)
    model = network()
    t = _vector_tensors(batch, torch.device("cpu"), TrainingConfig())
    with torch.no_grad():
        expected = model(**t, include_belief=False)
        actual = _states(model, [SimpleNamespace(observation=observation)], TrainingConfig())
    torch.testing.assert_close(actual, expected.state)
    torch.testing.assert_close(model.value_head(actual).squeeze(-1), expected.value)


@pytest.mark.parametrize("architecture", ["factorized", "attention", "relational"])
def test_policy_cannot_bypass_shared_action_tokens(architecture):
    model = network(architecture)
    # A neutral shared representation must remove all action preferences,
    # including those formerly supplied by a separate raw-action scorer.
    handle = model.action_encoder.register_forward_hook(
        lambda module, args, result: (torch.zeros_like(result[0]), result[1]))
    try:
        data = inputs()
        a = model(*data, include_belief=False)
        data[2][..., 0] = torch.arange(data[4].shape[1])
        data[2][..., 1:3] = 0
        data[3].fill_(5)
        b = model(*data, include_belief=False)
    finally:
        handle.remove()
    torch.testing.assert_close(a.policy_logits, a.policy_logits[:, :1].expand_as(a.policy_logits))
    torch.testing.assert_close(a.policy_logits, b.policy_logits)
    removed = {"relational_policy", "action_embeddings", "action_continuous",
               "action_to_map", "type_policy_head", "option_policy_head"}
    assert not removed.intersection(dict(model.named_children()))


@pytest.mark.parametrize("old_version", [None, 1])
def test_parallel_scorer_checkpoints_are_rejected_before_loading_weights(tmp_path, old_version):
    from dataclasses import asdict
    from polybot.infrastructure.model_io import load_model
    model = network()
    config = asdict(model.config)
    if old_version is None:
        config.pop("action_encoder_version")
    else:
        config["action_encoder_version"] = old_version
    checkpoint = tmp_path / "parallel.pt"
    torch.save({"model_config": config, "state_dict": model.state_dict()}, checkpoint)
    with pytest.raises(ValueError, match="single shared-action policy"):
        load_model(checkpoint, torch.device("cpu"))


def test_direct_evaluation_joins_unequal_legal_lists_without_changing_predictions():
    from polybot.evaluation.matches import _join_tensor_rows
    model = network()
    rows = [inputs(7), inputs(16)]
    names = ("map_tokens", "global_features", "action_categorical", "action_continuous", "action_mask")
    groups = [dict(zip(names, data)) for data in rows]
    expected = [model(*data, include_belief=False) for data in rows]
    joined = [_join_tensor_rows(groups, name) for name in names]
    actual = model(*joined, include_belief=False)
    assert joined[-1].sum(dim=1).tolist() == [7, 16]
    assert torch.isneginf(actual.policy_logits[0, 7:]).all()
    for row, (data, output) in enumerate(zip(rows, expected)):
        torch.testing.assert_close(actual.policy_logits[row, :data[-1].shape[1]], output.policy_logits[0])
        torch.testing.assert_close(actual.value[row], output.value[0])
