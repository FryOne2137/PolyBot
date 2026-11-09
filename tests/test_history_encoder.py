"""History semantics must survive padding, ID renaming and model persistence."""

from dataclasses import asdict

import pytest
import torch
from torch.nn import functional as F

from polybot.config import ModelConfig
from polybot.infrastructure.model_io import load_model, save_model
from polybot.models.history_encoder import HistoryEncoder
from polybot.models.policy_value_belief import PolicyValueBeliefNet


def history(length=2):
    result = torch.full((1, length, 12), -1, dtype=torch.long)
    result[..., 0] = 1
    result[..., 1] = 0
    result[..., 2] = 3
    result[..., 3] = 3
    result[..., 4] = 25
    result[..., 5] = 26
    result[..., 6] = 40
    result[..., 7] = 70
    result[..., 8:10] = 1
    result[..., 10] = 0
    return result


def model():
    torch.manual_seed(42)
    result = PolicyValueBeliefNet(ModelConfig(
        d_model=16, attention_heads=4, transformer_layers=1,
        action_encoder_dim=16, action_encoder_heads=4, action_latent_count=4,
        belief_decoder_layers=1, dropout=0,
    )).eval()
    # Emulate the open gate after warm-up; initialization deliberately starts
    # neutral, so semantic tests must not merely measure a zero output gate.
    with torch.no_grad():
        result.history_gate[-1].weight.copy_(torch.eye(16))
    return result


def test_high_unit_types_are_distinct_and_receive_their_own_gradients():
    torch.manual_seed(7)
    encoder = HistoryEncoder(16, 32, 16)
    first = history()
    second = first.clone()
    second[..., 6] = 70
    inputs = torch.cat([first, second])
    output = encoder(inputs, 11)
    assert not torch.allclose(output[0], output[1])
    target = torch.randn_like(output)
    F.mse_loss(output, target).backward()
    gradient = encoder.categories["source_unit_type"].weight.grad
    assert gradient[41].abs().sum() > 0
    assert gradient[71].abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in encoder.parameters() if p.grad is not None)


def test_history_coordinates_use_the_actual_board_width_and_keep_unknown_separate():
    torch.manual_seed(3)
    encoder = HistoryEncoder(16, 32, 16).eval()
    small = history()
    large = small.clone()
    # Identical coordinates (3,2) and (4,2) have different row-major indexes.
    large[..., 4] = 35
    large[..., 5] = 36
    torch.testing.assert_close(encoder(small, 11), encoder(large, 16), atol=0, rtol=0)
    unknown = small.clone()
    unknown[..., 4] = -1
    origin = small.clone()
    origin[..., 4] = 0
    assert not torch.allclose(encoder(unknown, 11), encoder(origin, 11))


def test_detail_categories_do_not_alias_between_spawn_and_tile_actions():
    torch.manual_seed(2)
    encoder = HistoryEncoder(16, 32, 16)
    data = history()
    data[..., 10] = 1
    data[..., 11] = 7
    other = data.clone()
    other[..., 10] = 3
    assert not torch.allclose(encoder(data, 11), encoder(other, 11))
    # Unknown category and valid category zero also have distinct embeddings.
    other = data.clone()
    other[..., 6] = -1
    data[..., 6] = 0
    assert not torch.allclose(encoder(data, 11), encoder(other, 11))


def test_visible_unit_id_renaming_cannot_change_history_features():
    encoder = HistoryEncoder(16, 32, 16).eval()
    data = history()
    renamed = data.clone()
    renamed[..., 8] = 9184
    renamed[..., 9] = 42
    torch.testing.assert_close(encoder(data, 11), encoder(renamed, 11), rtol=0, atol=0)


def test_full_model_uses_semantic_history_and_is_invariant_to_padding():
    from PolyEnv import GameEnv
    from polybot.data.polyenv_features import packet_to_tensors

    network = model()
    tensors = packet_to_tensors(GameEnv(seed=2137).model_request_numpy(), torch.device("cpu"))
    args = (tensors.map_tokens, tensors.global_features, tensors.action_categorical,
            tensors.action_continuous, tensors.action_mask)
    data = history()
    changed = data.clone()
    changed[..., 6] = 70
    a = network(*args, action_history=data, action_history_mask=torch.ones(1, 2, dtype=torch.bool))
    b = network(*args, action_history=changed, action_history_mask=torch.ones(1, 2, dtype=torch.bool))
    assert not torch.allclose(a.policy_logits, b.policy_logits)
    assert not torch.allclose(a.value, b.value)
    for left, right in zip(a.belief_logits, b.belief_logits):
        assert not torch.allclose(left, right)
    padded = F.pad(data, (0, 0, 5, 0), value=999)
    mask = F.pad(torch.ones(1, 2, dtype=torch.bool), (5, 0), value=False)
    c = network(*args, action_history=padded, action_history_mask=mask)
    torch.testing.assert_close(a.policy_logits, c.policy_logits)
    torch.testing.assert_close(a.value, c.value)
    for left, right in zip(a.belief_logits, c.belief_logits):
        torch.testing.assert_close(left, right)


def test_semantic_history_checkpoint_roundtrip_and_previous_schema_rejection(tmp_path):
    network = model()
    path = tmp_path / "semantic.pt"
    save_model(path, network, network.config)
    restored, config = load_model(path, torch.device("cpu"))
    restored.eval()
    assert config.history_encoder_version == 2
    data, mask = history(), torch.ones(1, 2, dtype=torch.bool)
    expected = network._history_context(data, mask, 1, torch.device("cpu"), board_side=11)
    actual = restored._history_context(data, mask, 1, torch.device("cpu"), board_side=11)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for previous in (None, 1):
        saved_config = asdict(network.config)
        if previous is None:
            saved_config.pop("history_encoder_version")
        else:
            saved_config["history_encoder_version"] = previous
        torch.save({"model_config": saved_config, "state_dict": network.state_dict()}, path)
        with pytest.raises(ValueError, match="previous history encoder"):
            load_model(path, torch.device("cpu"))
