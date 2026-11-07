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
    # Pre-factorized checkpoints have the old cross-attention policy and do
    # not contain this field.  Keep them runnable instead of silently loading
    # them into a different architecture with random policy weights.
    saved_config.setdefault("policy_architecture", "attention")
    config = ModelConfig(**saved_config)
    model = PolicyValueBeliefNet(config).to(device)
    incompatible = model.load_state_dict(payload["state_dict"], strict=False)
    # New action-conditioned belief layers are zero-gated initially, so old
    # checkpoints remain behaviourally identical until subsequent training.
    unexpected = set(incompatible.unexpected_keys)
    missing = set(incompatible.missing_keys)
    allowed_missing = {
        name for name in missing
        if name.startswith(("belief_", "history_"))
    }
    if unexpected or missing != allowed_missing:
        raise RuntimeError(
            "checkpoint architecture is incompatible; unexpected="
            f"{sorted(unexpected)}, missing={sorted(missing - allowed_missing)}"
        )
    return model, config
