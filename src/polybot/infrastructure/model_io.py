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
    config = ModelConfig(**payload["model_config"])
    model = PolicyValueBeliefNet(config).to(device)
    model.load_state_dict(payload["state_dict"])
    return model, config
