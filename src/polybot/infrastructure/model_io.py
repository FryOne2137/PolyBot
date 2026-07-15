"""Model-only persistence.

Training data, games, replay buffers and optimiser state deliberately stay in
memory.  A saved file contains only PyTorch model weights and model settings.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from ..config import ModelConfig
from ..models.policy_value_belief import PolicyValueBeliefNet


def save_model(path: Path, model: PolicyValueBeliefNet, config: ModelConfig) -> None:
    """Write the one allowed training artifact: model weights."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_config": asdict(config), "state_dict": model.state_dict()},
        path,
    )


def load_model(path: Path, device: torch.device) -> tuple[PolicyValueBeliefNet, ModelConfig]:
    """Load a model-only checkpoint created by :func:`save_model`."""
    payload = torch.load(path, map_location=device)
    saved_config = dict(payload["model_config"])
    # Pre-factorized checkpoints have the old cross-attention policy and do
    # not contain this field.  Keep them runnable instead of silently loading
    # them into a different architecture with random policy weights.
    saved_config.setdefault("policy_architecture", "attention")
    config = ModelConfig(**saved_config)
    model = PolicyValueBeliefNet(config).to(device)
    model.load_state_dict(payload["state_dict"])
    return model, config
