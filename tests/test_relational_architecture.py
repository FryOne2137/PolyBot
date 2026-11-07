from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from polybot.config import ModelConfig, TrainingConfig
from polybot.infrastructure.model_io import load_model, save_model
from polybot.models.policy_value_belief import PolicyValueBeliefNet
from polybot.models.relational_policy import city_tile_indices, unit_tile_indices
from polybot.training.flat_experiments import model_fingerprint
from polybot.training.vector_teacher import _vector_tensors


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model(architecture="relational"):
    return PolicyValueBeliefNet(ModelConfig(
        d_model=16, transformer_layers=1, attention_heads=2,
        action_interaction_dim=8, dropout=0.0, policy_architecture=architecture,
    )).eval()


def tensors(side=3, actions=4):
    board = torch.zeros((1, side * side, 23), dtype=torch.long)
    board[..., 0] = 1
    board[..., 14:16] = -1
    board[:, :2, 14] = 2
    board[0, :2, 15] = torch.tensor([5, 17])
    categorical = torch.full((1, actions, 12), -1, dtype=torch.long)
    categorical[..., 0] = 3
    categorical[..., 1] = torch.arange(actions) % (side * side)
    categorical[..., 2] = (torch.arange(actions) + 1) % (side * side)
    categorical[0, :2, 4] = torch.tensor([5, 17])
    return [board, torch.zeros(1, 9), categorical, torch.zeros(1, actions, 4),
            torch.ones(1, actions, dtype=torch.bool)]


@pytest.mark.parametrize("side,actions", [(1, 4), (3, 4), (11, 256), (13, 1024)])
def test_relational_scores_variable_maps_and_masks_padding(side, actions):
    network = model()
    if side == 1:
        data = tensors(3, actions)
        data[0] = data[0][:, :1].clone()
        data[2][..., 1:3] = 0
    else:
        data = tensors(side, actions)
    data[4][:, -1] = False
    output = network(*data, include_belief=False)
    assert output.policy_logits.shape == (1, actions)
    assert torch.isfinite(output.policy_logits[:, :-1]).all()
    assert torch.isneginf(output.policy_logits[:, -1]).all()
    assert torch.isfinite(output.value).all() and output.value.abs().max() <= 1
    assert not output.belief_logits


def test_city_lookup_uses_actual_public_centre_and_handles_missing_or_large_ids():
    board = tensors()[0]
    board[0, 0, 15] = 900_000
    ids = torch.tensor([[17, -1, 900_000, 999]])
    indices, present = city_tile_indices(board, ids)
    assert present.tolist() == [[True, False, True, False]]
    assert indices[0, [0, 2]].tolist() == [1, 0]
    board[0, 0, 0] = 0
    assert not city_tile_indices(board, ids)[1][0, 2]


def test_relational_encoder_rejects_ambiguous_board_coordinates():
    network = model()
    data = tensors()
    data[0] = data[0][:, :6]
    with pytest.raises(ValueError, match="square and fit"):
        network(*data, include_belief=False)
    network.config.max_board_side = 2
    with pytest.raises(ValueError, match="square and fit"):
        network(*tensors(), include_belief=False)


def test_city_lookup_matches_real_native_public_centres():
    from PolyEnv import Bardur, Imperius, VectorGameEnv
    env = VectorGameEnv(
        num_envs=2, seed=15271, map_size=11, players=(Bardur, Imperius),
        max_actions=1024, num_threads=1, visible_event_history=0,
    )
    batch = env.reset(seed=15271)
    public = _vector_tensors(batch, torch.device("cpu"), TrainingConfig())
    tokens = public["map_tokens"]
    centres = [torch.nonzero((row[:, 0] == 1) & (row[:, 14] == 2)).flatten() for row in tokens]
    assert all(len(row) for row in centres)
    cities = torch.full((2, max(map(len, centres)) + 1), -1, dtype=torch.long)
    for row, positions in enumerate(centres):
        cities[row, :len(positions)] = tokens[row, positions, 15]
    indices, present = city_tile_indices(public["map_tokens"], cities)
    requested = cities >= 0
    assert present[requested].all()
    assert torch.equal(tokens[..., 15].gather(1, indices)[requested], cities[requested])
    units = public["action_categorical"][..., 3]
    unit_index, unit_present = unit_tile_indices(tokens, units)
    requested_units = public["action_mask"] & (units >= 0)
    assert requested_units.any() and unit_present[requested_units].all()
    assert torch.equal(tokens[..., 6].gather(1, unit_index)[requested_units], units[requested_units])


def test_runtime_id_renaming_does_not_change_relational_policy_or_value():
    torch.manual_seed(19)
    network = model()
    with torch.no_grad():
        network.history_gate[-1].weight.normal_(std=0.1)
    original = tensors()
    history = torch.zeros((1, 4, 12))
    history[..., 8:10] = 6
    original += [None, history, torch.ones(1, 4, dtype=torch.bool)]
    changed = [item.clone() if isinstance(item, torch.Tensor) else item for item in original]
    changed[0][..., 6] = 30
    changed[0][..., 22] = 91
    changed[0][0, :2, 15] = torch.tensor([901, 42])
    changed[1][..., 6] = 901 / 1024
    changed[2][..., 3] = 33
    changed[2][0, :2, 4] = torch.tensor([901, 42])
    changed[6][..., 8:10] = 80
    a, b = network(*original, include_belief=False), network(*changed, include_belief=False)
    torch.testing.assert_close(a.policy_logits, b.policy_logits, atol=0, rtol=0)
    torch.testing.assert_close(a.value, b.value, atol=0, rtol=0)


def test_candidate_order_or_padding_does_not_change_scores():
    network = model()
    data = tensors(actions=6)
    data[4][0, -2:] = False
    permuted = [t.clone() for t in data]
    order = torch.tensor([5, 2, 4, 0, 3, 1])
    for column in (2, 3, 4):
        permuted[column] = permuted[column][:, order]
    first = network(*data, include_belief=False)
    second = network(*permuted, include_belief=False)
    torch.testing.assert_close(first.policy_logits[:, order], second.policy_logits)
    torch.testing.assert_close(first.value, second.value)


def test_unit_only_pointer_is_equivalent_to_explicit_source_tile():
    network = model()
    data = tensors()
    data[0][0, 3, 4] = 1
    data[0][0, 3, 6] = 71
    data[2][0, :, 1] = -1
    data[2][0, :, 3] = 71
    inferred = network(*data, include_belief=False)
    data[2][0, :, 1] = 3
    explicit = network(*data, include_belief=False)
    torch.testing.assert_close(inferred.policy_logits, explicit.policy_logits, atol=0, rtol=0)


def test_shared_relational_scorer_learns_nonadditive_source_target_preferences():
    torch.manual_seed(731)
    legacy, relational = model("factorized"), model()
    state = torch.randn(1, 16)
    tiles = torch.randn(1, 9, 16)
    data = tensors()
    actions = data[2]
    actions[..., 4] = -1
    actions[0, :, 1] = torch.tensor([0, 0, 1, 1])
    actions[0, :, 2] = torch.tensor([2, 3, 2, 3])
    additive = legacy._factorized_policy(state, tiles, data[0], actions, data[3])[0]
    assert abs(float((additive[0] + additive[3] - additive[1] - additive[2]).detach())) < 1e-5
    # Prefer opposite destinations for the two sources. No sum of independent
    # source and destination scores can fit these preferences simultaneously.
    target = torch.tensor([[0.45, 0.05, 0.05, 0.45]])
    optimizer = torch.optim.Adam(relational.relational_policy.parameters(), lr=0.01)
    for _ in range(100):
        logits = relational.relational_policy(state, tiles, data[0], actions, data[3])
        loss = -(target * logits.log_softmax(dim=-1)).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    probabilities = logits.detach().softmax(dim=-1)[0]
    assert (probabilities[[0, 3]] > 0.35).all()
    assert (probabilities[[1, 2]] < 0.15).all()


def test_empty_visibility_and_history_have_finite_backward_gradients():
    network = model()
    data = tensors()
    data[0].fill_(-1)
    data[0][..., 0] = 0
    output = network(*data, include_belief=True)
    loss = F.cross_entropy(output.policy_logits, torch.tensor([0])) + output.value.square().mean()
    loss += sum(logits.square().mean() for logits in output.belief_logits)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)
    assert network.spatial_readout.combine[0].weight.grad.abs().sum() > 0
    assert network.relational_policy.source_projection.weight.grad is not None


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device is unavailable",
))])
def test_relational_autocast_has_finite_policy_value_and_gradients(device):
    network = model().to(device).train()
    data = [tensor.to(device) for tensor in tensors()]
    data[0][..., 0] = 0  # Exercise all-hidden spatial pooling under autocast.
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    with torch.autocast(device, dtype=dtype):
        output = network(*data, include_belief=False)
        loss = F.cross_entropy(output.policy_logits.float(), torch.tensor([0], device=device))
        loss += output.value.float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in network.parameters() if p.grad is not None)


@pytest.mark.parametrize("architecture", ["attention", "factorized", "relational"])
def test_checkpoint_roundtrip_and_legacy_config_default(tmp_path, architecture):
    torch.manual_seed(32)
    network = model(architecture)
    file = tmp_path / "model.pt"
    save_model(file, network, network.config)
    if architecture != "relational":
        payload = torch.load(file, weights_only=False)
        payload["model_config"].pop("action_interaction_dim")
        torch.save(payload, file)
    loaded, config = load_model(file, torch.device("cpu"))
    loaded.eval()
    assert config.policy_architecture == architecture
    data = tensors()
    original, restored = network(*data, include_belief=False), loaded(*data, include_belief=False)
    torch.testing.assert_close(original.policy_logits, restored.policy_logits, atol=0, rtol=0)
    torch.testing.assert_close(original.value, restored.value, atol=0, rtol=0)
    if architecture == "relational":
        payload = torch.load(file, weights_only=False)
        payload["state_dict"].pop("relational_policy.score.3.weight")
        torch.save(payload, file)
        with pytest.raises(RuntimeError, match="architecture is incompatible"):
            load_model(file, torch.device("cpu"))


def test_initialization_cli_records_reproducible_random_weights_and_blocks_overwrite(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/init_model.py"
    fingerprints = []
    for name in ("first", "second"):
        arguments = [str(script), "--model-out", str(tmp_path / (name + ".pt")),
                     "--project", str(tmp_path / "runs"), "--name", name,
                     "--d-model", "16", "--heads", "2", "--layers", "1",
                     "--action-interaction-dim", "8", "--seed", "42"]
        with patch.object(sys, "argv", arguments), patch.object(sys, "path", [str(script.parent), *sys.path]):
            runpy.run_path(str(script), run_name="__main__")
        initialized, _ = load_model(tmp_path / (name + ".pt"), torch.device("cpu"))
        fingerprints.append(model_fingerprint(initialized))
    assert fingerprints[0] == fingerprints[1]
    manifests = list((tmp_path / "runs").rglob("experiment.json"))
    assert len(manifests) == 2
    assert json.loads(manifests[0].read_text())["initialization_seed"] == 42
    with patch.object(sys, "argv", arguments), patch.object(sys, "path", [str(script.parent), *sys.path]):
        with pytest.raises(SystemExit):
            runpy.run_path(str(script), run_name="__main__")
