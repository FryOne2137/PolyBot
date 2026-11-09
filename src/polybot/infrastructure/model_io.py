"""Model-only persistence.

A model checkpoint contains only PyTorch weights and architecture settings.
Trainer continuation state is stored separately from this portable format.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from ..config import ModelConfig
from ..models.policy_value_belief import PolicyValueBeliefNet


def save_model(path: Path, model: PolicyValueBeliefNet, config: ModelConfig) -> None:
    """Atomically write model weights and their architecture settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(
            {"model_config": asdict(config), "state_dict": model.state_dict()},
            temporary_path,
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_model(path: Path, device: torch.device) -> tuple[PolicyValueBeliefNet, ModelConfig]:
    """Load a model-only checkpoint created by :func:`save_model`."""
    # Portable checkpoints contain tensors and primitive configuration only.
    # Set this explicitly: older supported PyTorch releases default to an
    # unrestricted pickle loader, unlike recent releases.
    payload = torch.load(path, map_location=device, weights_only=True)
    saved_config = dict(payload["model_config"])
    if saved_config.get("tile_encoder_version") != 2:
        raise ValueError(
            "checkpoint uses the previous tile encoder; the joint 24-field "
            "encoder requires a new model initialized with scripts/init_model.py "
            "and a separate experiment/checkpoint directory"
        )
    if saved_config.get("action_encoder_version") != 2:
        raise ValueError("checkpoint predates the single shared-action policy (action encoder v2); initialize a new model with scripts/init_model.py")
    if saved_config.get("history_encoder_version") != 2:
        raise ValueError("checkpoint uses the previous history encoder; semantic history v2 requires a new model initialized with scripts/init_model.py")
    # The saved map-encoding profile must not be inferred from the file name.
    saved_config.setdefault("policy_architecture", "attention")
    config = ModelConfig(**saved_config)
    model = PolicyValueBeliefNet(config).to(device)
    incompatible = model.load_state_dict(payload["state_dict"], strict=False)
    # Version 2 is a new architecture with a complete, explicit schema.
    # Missing layers must never turn into silently random checkpoint weights.
    unexpected = set(incompatible.unexpected_keys)
    missing = set(incompatible.missing_keys)
    if unexpected or missing:
        raise RuntimeError(
            "checkpoint architecture is incompatible; unexpected="
            f"{sorted(unexpected)}, missing={sorted(missing)}"
        )
    return model, config
