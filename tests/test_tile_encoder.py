from dataclasses import replace

import pytest
import torch

from polybot.config import ModelConfig
from polybot.data.player_context import UNKNOWN_POPULATION
from polybot.models.policy_value_belief import PolicyValueBeliefNet


def network(architecture="relational"):
    torch.manual_seed(31)
    return PolicyValueBeliefNet(ModelConfig(
        policy_architecture=architecture, d_model=16, attention_heads=4,
        transformer_layers=1, belief_decoder_layers=1, dropout=0,
    )).eval()


def own_city_tokens():
    tokens = torch.full((1, 1, 24), -1, dtype=torch.long)
    tokens[..., 0] = 1
    tokens[..., 13] = 1
    tokens[..., 14] = 2
    tokens[..., 16] = 0
    tokens[..., 23] = 2
    return tokens


@pytest.mark.parametrize("architecture", ["factorized", "attention", "relational"])
def test_population_and_level_are_fused_jointly_and_receive_gradients(architecture):
    model = network(architecture)
    encoder = model.tile_encoder
    tokens = own_city_tokens()
    observer = torch.tensor([0])
    low_level = encoder(tokens, observer)
    more_population = tokens.clone(); more_population[..., 23] = 6
    low_level_pop_delta = encoder(more_population, observer) - low_level
    higher_level = tokens.clone(); higher_level[..., 13] = 3
    higher_level_pop = higher_level.clone(); higher_level_pop[..., 23] = 6
    high_level_pop_delta = encoder(higher_level_pop, observer) - encoder(higher_level, observer)
    assert not torch.allclose(low_level_pop_delta, torch.zeros_like(low_level_pop_delta))
    # The population effect can depend on city level before the map Transformer.
    assert not torch.allclose(low_level_pop_delta, high_level_pop_delta)
    low_level[..., 0].sum().backward()
    assert encoder.fusion[0].weight.grad[:, -2].abs().sum() > 0
    assert encoder.embeddings[13].weight.grad.abs().sum() > 0


def test_missing_zero_negative_population_and_foreign_population():
    encoder = network().tile_encoder
    observer = torch.tensor([0])
    tokens = own_city_tokens()
    values = []
    for population in (UNKNOWN_POPULATION, -1, 0, 1):
        tokens[..., 23] = population
        values.append(encoder(tokens, observer))
    for a, b in zip(values, values[1:]):
        assert not torch.allclose(a, b)
    for visibility, owner, identity in ((0, 0, 0), (1, 1, 0), (1, -1, -1)):
        tokens[..., 0], tokens[..., 16], tokens[..., 23] = visibility, owner, 3
        a = encoder(tokens, torch.tensor([identity]))
        tokens[..., 23] = 999
        torch.testing.assert_close(a, encoder(tokens, torch.tensor([identity])), rtol=0, atol=0)


def test_legacy_map_is_identical_to_explicitly_unknown_population():
    encoder = network().tile_encoder
    tokens = own_city_tokens()
    tokens[..., 23] = UNKNOWN_POPULATION
    torch.testing.assert_close(encoder(tokens, torch.tensor([0])),
                               encoder(tokens[..., :23], torch.tensor([0])), rtol=0, atol=0)


@pytest.mark.parametrize("side", [11, 13, 16])
def test_joint_encoder_forward_backward_on_native_scalar_map_sizes(side):
    from PolyEnv import GameEnv
    from polybot.data.polyenv_features import packet_to_tensors

    model = network()
    packet = GameEnv(seed=2137, map_size=side).model_request_numpy()
    data = packet_to_tensors(packet, torch.device("cpu"))
    output = model(data.map_tokens, data.global_features, data.action_categorical,
                   data.action_continuous, data.action_mask)
    loss = output.value.square().mean() - output.policy_logits.log_softmax(-1)[0, 0]
    loss += sum(x.square().mean() for x in output.belief_logits)
    loss.backward()
    assert output.tile_states.shape == (1, side*side, 16)
    assert model.tile_encoder.fusion[0].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_tile_schema_version_is_explicit():
    config = replace(network().config, tile_encoder_version=1)
    with pytest.raises(ValueError, match="tile_encoder_version"):
        PolicyValueBeliefNet(config)
