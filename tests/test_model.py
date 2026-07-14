import numpy as np
import torch

from polybot.config import ModelConfig
from polybot.data.token_schema import MAP_FEATURE_MAX
from polybot.models.policy_value_belief import PolicyValueBeliefNet


def test_model_scores_only_unmasked_actions_and_keeps_known_tiles():
    model = PolicyValueBeliefNet(ModelConfig(d_model=64, transformer_layers=2, attention_heads=4))
    board = torch.full((1, 9, 23), -1, dtype=torch.long)
    board[:, :, 0] = 0
    board[:, 0, 0] = 1
    board[:, 0, 19] = 2
    global_features = torch.zeros((1, 9))
    categorical = torch.full((1, 3, 12), -1, dtype=torch.long)
    categorical[:, :, 0] = torch.tensor([0, 1, 3])
    continuous = torch.zeros((1, 3, 4))
    mask = torch.tensor([[True, True, False]])

    output = model(board, global_features, categorical, continuous, mask)
    assert output.policy_logits.shape == (1, 3)
    assert torch.isneginf(output.policy_logits[0, 2])
    particles = model.sample_completed_maps(board, global_features, particles=2)
    assert len(particles) == 2
    assert torch.equal(particles[0][0], board[0, 0])
