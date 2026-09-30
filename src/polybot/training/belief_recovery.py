"""Shared, fog-safe recovery helpers for rejected SelfPlayPool beliefs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


RETRY_TEMPERATURES = (0.0, 0.15, 0.30, 0.60, 1.0)
_MAX_SEED = 2**31 - 1
# Must mirror PolyEnv's UnitType::GiantSuper. Keeping the local diagnostic
# stricter than the model vocabulary ensures it explains every native
# ``invalid unit state`` rejection instead of accepting unused token ids.
_UNIT_TYPE_MAX = 90


def _invalid_unit_token_details(
    completed_map: np.ndarray,
    player_count: int,
) -> dict[str, int] | None:
    """Describe the first locally invalid unit row without source-map access."""
    tokens = np.asarray(completed_map, dtype=np.int32)
    if tokens.ndim != 2 or tokens.shape[1] < 23:
        return None
    for tile, token in enumerate(tokens):
        unit_type = int(token[4])
        health = int(token[2])
        owner = int(token[3])
        max_health = int(token[21])
        origin_city = int(token[22])
        if unit_type == -1:
            if any(value != -1 for value in (health, owner, max_health, origin_city)):
                return {
                    "invalid_unit_tile": tile,
                    "unit_health": health,
                    "unit_owner": owner,
                    "unit_type": unit_type,
                    "unit_max_health": max_health,
                    "unit_origin_city": origin_city,
                    "unit_validation_code": 2,
                }
            continue
        if (
            not 1 <= unit_type <= _UNIT_TYPE_MAX
            or not 0 <= owner < player_count
            or health < 0
            or max_health < health
            or origin_city < -1
        ):
            return {
                "invalid_unit_tile": tile,
                "unit_health": health,
                "unit_owner": owner,
                "unit_type": unit_type,
                "unit_max_health": max_health,
                "unit_origin_city": origin_city,
                "unit_validation_code": 1,
            }
    return None


def _enrich_rejection_diagnostics(
    diagnostics: Sequence[Mapping[str, Any]],
    completed: np.ndarray,
    rejection_mask: np.ndarray,
) -> tuple[Mapping[str, Any], ...]:
    """Attach actionable unit fields and synthesize rows missing from PolyEnv."""
    enriched: list[dict[str, Any]] = []
    represented: set[tuple[int, int, int]] = set()
    for raw in diagnostics:
        row = dict(raw)
        env = int(row.get("env_id", -1))
        player = int(row.get("player", -1))
        particle = int(row.get("particle", -1))
        if (
            0 <= env < completed.shape[0]
            and 0 <= player < completed.shape[1]
            and 0 <= particle < completed.shape[2]
        ):
            represented.add((env, player, particle))
            details = _invalid_unit_token_details(
                completed[env, player, particle], completed.shape[1]
            )
            if details is not None:
                row.update(details)
        enriched.append(row)

    for env, player, particle in np.argwhere(rejection_mask):
        key = (int(env), int(player), int(particle))
        if key in represented:
            continue
        details = _invalid_unit_token_details(
            completed[key[0], key[1], key[2]], completed.shape[1]
        )
        if details is None:
            continue
        enriched.append({
            "accepted": False,
            "env_id": key[0],
            "player": key[1],
            "particle": key[2],
            "reason": "belief_build_error",
            "error": "completed_map_tokens contains an invalid unit state",
            **details,
        })
    return tuple(enriched)


def _refine_env_rejection_mask(
    rejection_mask: np.ndarray,
    diagnostics: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Narrow an env-level verdict to exact rejected particles when possible."""
    refined = np.asarray(rejection_mask, dtype=bool).copy()
    detailed = np.zeros_like(refined)
    covered_envs: set[int] = set()
    for row in diagnostics:
        if bool(row.get("accepted", False)):
            continue
        env = int(row.get("env_id", -1))
        if not 0 <= env < refined.shape[0] or not refined[env].any():
            continue
        covered_envs.add(env)
        player = int(row.get("player", -1))
        particle = int(row.get("particle", -1))
        if 0 <= player < refined.shape[1] and 0 <= particle < refined.shape[2]:
            detailed[env, player, particle] = True
        else:
            # An env-only diagnostic cannot prove which particle is safe.
            detailed[env] = True
    for env in covered_envs:
        refined[env] = detailed[env]
    return refined


class BeliefRecoveryRequired(RuntimeError):
    """The caller must truncate and replace only the listed training slots."""

    def __init__(
        self,
        env_ids: Sequence[int] | np.ndarray,
        message: str,
        diagnostics: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        super().__init__(message)
        self.env_ids = np.unique(np.asarray(env_ids, dtype=np.int32))
        self.diagnostics = tuple(dict(row) for row in diagnostics)


class BeliefQualityError(RuntimeError):
    """Belief failure rate is too high for trustworthy large-scale training."""

    def __init__(self, message: str, metrics: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.metrics = dict(metrics or {})


def particle_temperature(base: float, attempt: int, particle: int) -> float:
    """Use one greedy particle, then independent stochastic completions."""
    schedule = RETRY_TEMPERATURES[min(max(0, int(attempt)), len(RETRY_TEMPERATURES) - 1)]
    if attempt == 0 and particle == 0:
        return max(0.0, float(base))
    return max(float(base), schedule, 0.15)


def ismcts_rejected_envs(
    pool: object,
    state_ids: np.ndarray,
    particles: np.ndarray,
    *,
    submit_failed: bool = True,
) -> tuple[np.ndarray, np.ndarray, int, tuple[Mapping[str, Any], ...]]:
    """Return rejected slots without reading authoritative hidden map tokens.

    New PolyEnv builds expose a cheap full-ISMCTS mask. Older compatible
    builds expose detailed diagnostics containing only failures. The latter is
    used only after an atomic submit failed, so it is outside the hot path.
    """
    ids = np.ascontiguousarray(state_ids, dtype=np.uint64)
    completed = np.ascontiguousarray(particles, dtype=np.int32)
    if completed.ndim != 5 or completed.shape[0] != ids.size:
        raise ValueError("ISMCTS particles must have shape [B,P,K,T,23]")
    rejection_mask = np.zeros(completed.shape[:3], dtype=bool)
    mask_method = getattr(pool, "ismcts_belief_acceptance_mask", None)
    if mask_method is not None:
        try:
            accepted = np.asarray(mask_method(ids, completed), dtype=bool)
            env_level_verdict = False
            if accepted.shape == completed.shape[:3]:
                rejection_mask = ~accepted
            elif accepted.shape == ids.shape:
                rejection_mask[~accepted] = True
                env_level_verdict = True
            elif accepted.ndim >= 1 and accepted.shape[0] == ids.size:
                # Be forward-compatible with a mask that omits one of the
                # player/particle axes by reducing it to an env-level verdict.
                rejected_env_mask = ~accepted.reshape(ids.size, -1).all(axis=1)
                rejection_mask[rejected_env_mask] = True
                env_level_verdict = True
            else:
                raise ValueError("unexpected ISMCTS acceptance-mask shape")
            rejected = np.flatnonzero(rejection_mask.any(axis=(1, 2))).astype(np.int32)
            if rejected.size == 0 and submit_failed:
                # This helper is called only after atomic submit_beliefs
                # rejected the packet. An inconsistent empty mask must fail
                # closed rather than turn the following retry into a no-op.
                rejection_mask[:] = True
                rejected = np.arange(ids.size, dtype=np.int32)
            diagnostics: tuple[Mapping[str, Any], ...] = ()
            diagnostic_method = getattr(pool, "ismcts_belief_diagnostics", None)
            if rejected.size and callable(diagnostic_method):
                try:
                    diagnostics = tuple(diagnostic_method(ids, completed))
                except Exception:
                    diagnostics = ()
            if env_level_verdict and diagnostics:
                # Current PolyEnv returns one acceptance bit per environment,
                # while diagnostics identify the precise player/particle.
                # Use that detail so a single bad particle does not cause all
                # otherwise valid hypotheses in the slot to be regenerated.
                rejection_mask = _refine_env_rejection_mask(
                    rejection_mask, diagnostics
                )
                rejected = np.flatnonzero(
                    rejection_mask.any(axis=(1, 2))
                ).astype(np.int32)
            diagnostics = _enrich_rejection_diagnostics(
                diagnostics, completed, rejection_mask
            )
            return rejected, rejection_mask, int(rejection_mask.sum()), diagnostics
        except (TypeError, ValueError, RuntimeError):
            # A partially upgraded binding may expose a different prototype.
            # Fall back to the existing safe diagnostics API.
            pass

    if not submit_failed:
        return np.empty(0, dtype=np.int32), rejection_mask, 0, ()

    diagnostic_method = getattr(pool, "ismcts_belief_diagnostics", None)
    if diagnostic_method is None:
        rejected = np.arange(ids.size, dtype=np.int32)
        rejection_mask[:] = True
        return rejected, rejection_mask, int(rejection_mask.sum()), ()
    try:
        diagnostics = tuple(diagnostic_method(ids, completed))
    except Exception:
        rejected = np.arange(ids.size, dtype=np.int32)
        rejection_mask[:] = True
        return rejected, rejection_mask, int(rejection_mask.sum()), ()
    for row in diagnostics:
        if bool(row.get("accepted", False)):
            continue
        env = int(row.get("env_id", -1))
        if not 0 <= env < ids.size:
            continue
        player = int(row.get("player", -1))
        particle = int(row.get("particle", -1))
        if 0 <= player < completed.shape[1] and 0 <= particle < completed.shape[2]:
            rejection_mask[env, player, particle] = True
        else:
            rejection_mask[env] = True
    rejected = np.flatnonzero(rejection_mask.any(axis=(1, 2))).astype(np.int32)
    if rejected.size == 0:
        # Submit failed, therefore treating the batch as accepted would be
        # unsafe even if an older diagnostic binding returned no rows.
        rejected = np.arange(ids.size, dtype=np.int32)
        rejection_mask[:] = True
    return (
        rejected,
        rejection_mask,
        int(rejection_mask.sum()),
        _enrich_rejection_diagnostics(diagnostics, completed, rejection_mask),
    )


def ismcts_action_difference_rejections(
    pool: object,
    state_ids: np.ndarray,
    particles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, tuple[Mapping[str, Any], ...]]:
    """Reject particles whose public legal-action set differs from the root.

    The PolyEnv method is a rules-oracle query only: it returns public action
    signatures, never a hidden-world target.  Keeping this check before
    ``submit_beliefs`` prevents MCTS from assigning visits to routes or builds
    that only exist in a hallucinated determinisation.
    """
    ids = np.ascontiguousarray(state_ids, dtype=np.uint64)
    completed = np.ascontiguousarray(particles, dtype=np.int32)
    if completed.ndim != 5 or completed.shape[0] != ids.size:
        raise ValueError("ISMCTS particles must have shape [B,P,K,T,23]")
    method = getattr(pool, "ismcts_belief_action_differences", None)
    if not callable(method):
        return (
            np.empty(0, dtype=np.int32),
            np.zeros(completed.shape[:3], dtype=bool),
            0,
            (),
        )
    records = method(ids, completed)
    if isinstance(records, Mapping):
        records = records.get("differences", records.get("records", ()))
    if records is None:
        records = ()
    rejection_mask = np.zeros(completed.shape[:3], dtype=bool)
    diagnostics: list[Mapping[str, Any]] = []
    for raw in records:
        if not isinstance(raw, Mapping):
            raise RuntimeError(
                "PolyEnv ismcts_belief_action_differences must return mapping rows"
            )
        env = int(raw.get("env_id", -1))
        player = int(raw.get("player", -1))
        particle = int(raw.get("particle", -1))
        if not (
            0 <= env < completed.shape[0]
            and 0 <= player < completed.shape[1]
            and 0 <= particle < completed.shape[2]
        ):
            raise RuntimeError(
                "PolyEnv action-difference record has invalid env/player/particle indices"
            )
        missing = tuple(raw.get("missing_actions", ()) or ())
        extra = tuple(raw.get("extra_actions", ()) or ())
        invalid = bool(raw.get("invalid_state", False))
        if not (missing or extra or invalid):
            continue
        rejection_mask[env, player, particle] = True
        diagnostics.append({
            "accepted": False,
            "env_id": env,
            "player": player,
            "particle": particle,
            "reason": str(raw.get("reason", "legal_action_mismatch")),
            "missing_actions": missing,
            "extra_actions": extra,
            "invalid_state": invalid,
        })
    rejected = np.flatnonzero(rejection_mask.any(axis=(1, 2))).astype(np.int32)
    return rejected, rejection_mask, int(rejection_mask.sum()), tuple(diagnostics)


def capped_update_target(
    requested_updates: int,
    example_count: int,
    batch_size: int,
    max_replay_epochs: float,
) -> int:
    """Cap optimiser work by the number of examples actually collected."""
    if requested_updates <= 0 or example_count <= 0 or batch_size <= 0:
        return 0
    exposure_budget = float(example_count) * max(0.0, float(max_replay_epochs))
    return min(int(requested_updates), int(exposure_budget // int(batch_size)))


def replace_rejected_particles(
    particles: np.ndarray,
    rejected_envs: np.ndarray,
    rejection_mask: np.ndarray,
    replacements: np.ndarray,
) -> None:
    """Replace only rejected [env, player, particle] hypotheses in place."""
    rows = np.asarray(rejected_envs, dtype=np.int32)
    if replacements.shape != particles[rows].shape:
        raise ValueError("replacement particles do not match selected env rows")
    local_mask = np.asarray(rejection_mask, dtype=bool)[rows, :, :, None, None]
    selected = particles[rows].copy()
    np.copyto(selected, replacements, where=local_mask)
    particles[rows] = selected


def partition_recovery_envs(
    env_ids: Sequence[int] | np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep valid rejected slots, including already completed native rows."""
    active_mask = np.asarray(active, dtype=bool).reshape(-1)
    rows = np.unique(np.asarray([
        int(env) for env in env_ids if 0 <= int(env) < active_mask.size
    ], dtype=np.int32))
    return rows, rows[active_mask[rows]], rows[~active_mask[rows]]


def effective_replay_epochs(updates: int, batch_size: int, example_count: int) -> float:
    return float(updates * batch_size / max(1, example_count))


def replacement_seeds(
    base_seed: int,
    first_index: int,
    count: int,
    excluded: Sequence[int] = (),
) -> np.ndarray:
    """Deterministic training-only seeds far from the ordinary rollout range."""
    blocked = {int(value) for value in excluded}
    values: list[int] = []
    cursor = int(first_index)
    while len(values) < int(count):
        value = (int(base_seed) - 1 + 1_000_000_007 + cursor) % _MAX_SEED + 1
        cursor += 1
        if value not in blocked:
            values.append(value)
    return np.asarray(values, dtype=np.uint32)


def recovery_rate_exceeded(resets: int, decisions: int, threshold: float = 0.01) -> bool:
    # A single isolated recovery must remain recoverable even in a small
    # population chunk. Two failures are enough to make the rate meaningful.
    return resets > 1 and decisions > 0 and float(resets) / float(decisions) > float(threshold)
