"""Opt-in controls and auditable decisions for flat-policy experiments."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..config import ModelConfig


@dataclass(frozen=True)
class FlatImprovements:
    value_updates: int = 8
    value_batch_size: int = 128
    value_learning_rate: float = 1e-4
    value_capacity: int = 16384
    value_validation_games: int = 8
    value_positions_per_game: int = 32
    paired_rollback: bool = True
    cache_evaluations: bool = True
    scientific_artifacts: bool = True
    exploiter_objective: bool = True
    rollout_steps_final: int = 256

    def __post_init__(self):
        if self.value_updates < 0 or self.value_batch_size < 1:
            raise ValueError("invalid flat value update budget")
        if self.value_capacity < self.value_batch_size or self.value_learning_rate <= 0:
            raise ValueError("invalid flat value buffer or learning rate")
        if self.value_validation_games < 2 or self.value_validation_games % 2:
            raise ValueError("value validation needs a positive even game count")
        if self.value_positions_per_game < 1 or self.rollout_steps_final < 0:
            raise ValueError("invalid value position or rollout budget")


def model_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    config = getattr(model, "config", None)
    config_identity = repr(config)
    if isinstance(config, ModelConfig) and config.policy_architecture != "relational":
        # This new setting has no effect on legacy networks. Keep their
        # pre-relational identity so existing continuation files remain valid.
        config_identity = config_identity.replace(
            f", action_interaction_dim={config.action_interaction_dim}", ""
        )
    digest.update(config_identity.encode())
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        value = tensor.detach().cpu().contiguous()
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


@contextmanager
def isolated_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def paired_regression(
    candidate: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    *, margin: float = 0.02,
) -> dict[str, Any]:
    """Test a paired map-level difference, never treat the two sides as IID."""
    def rows_by_key(rows):
        result = {}
        for row in rows:
            key = (int(row["map_seed"]), int(row["model_a_player"]))
            if key in result:
                raise ValueError("duplicate map/player in regression evidence")
            score = float(row["score_a"])
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("invalid regression score")
            result[key] = score
        return result

    a, b = rows_by_key(candidate), rows_by_key(reference)
    if not a or a.keys() != b.keys():
        return {"valid": False, "confirmed": False, "reason": "unpaired_evidence"}
    maps = sorted({key[0] for key in a})
    if any((seed, side) not in a for seed in maps for side in (0, 1)):
        return {"valid": False, "confirmed": False, "reason": "incomplete_pairs"}
    differences = np.asarray([
        np.mean([a[seed, side] - b[seed, side] for side in (0, 1)])
        for seed in maps
    ])
    if len(differences) < 8:
        return {"valid": False, "confirmed": False, "reason": "too_few_pairs"}
    mean = float(differences.mean())
    se = float(differences.std(ddof=1) / np.sqrt(len(differences)))
    # A conservative small-sample normal approximation. Evidence is clustered
    # by map; no certainty is inferred from one paired map.
    radius = (1.96 + 2.0 / len(differences)) * se
    return {
        "valid": True, "pairs": len(maps), "difference": mean,
        "lower95": max(-1.0, mean - radius),
        "upper95": min(1.0, mean + radius),
        "confirmed": mean + radius < -margin,
        "reason": "paired_map_difference",
    }


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".flat-state-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_manifest(directory: Path, settings: FlatImprovements | None = None, **details) -> None:
    """Capture code/environment identity without copying user checkpoints."""
    root = Path(__file__).resolve().parents[3]
    def git(*arguments):
        try:
            return subprocess.run(
                ["git", *arguments], cwd=root, capture_output=True, text=True,
                timeout=10, check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    packages = {}
    for name in ("torch", "numpy", "PolyEnv"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    sources = {}
    for folder in (root / "src", root / "scripts"):
        for path in sorted(folder.rglob("*.py")):
            sources[str(path.relative_to(root))] = path.read_text(encoding="utf-8")
    encoded = json.dumps(sources, sort_keys=True).encode()
    payload = {
        "schema_version": 1, "commit": git("rev-parse", "HEAD"),
        "working_tree": git("status", "--porcelain"),
        "source_sha256": hashlib.sha256(encoded).hexdigest(),
        "python": sys.version, "platform": platform.platform(),
        "packages": packages, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "improvements": asdict(settings) if settings is not None else None, **details,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "experiment.json").write_text(json.dumps(payload, indent=2) + "\n")
    (directory / "source_snapshot.json").write_bytes(encoded)
