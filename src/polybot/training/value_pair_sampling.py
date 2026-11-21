"""Budgeted pair sampling; changes proposals, never the reference rollout policy."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AdaptivePairSampling:
    zero_fraction: float = 0.25
    zero_budget: int = 24
    zero_max_rollouts: int = 8
    source_interval: int = 4
    saved_root_probes: int = 16
    pairs_per_root: int = 2
    action_types: int = 12

    def __post_init__(self):
        if (not 0 < self.zero_fraction <= 1 or self.zero_budget < 0 or self.zero_max_rollouts < 2
                or self.source_interval < 1 or self.saved_root_probes < 0 or self.pairs_per_root < 1
                or self.action_types < 1):
            raise ValueError("invalid adaptive pair sampling budget")


def adaptive_rollout_batch(positions, results, decisions, settings, capacity, policy, state, regular):
    """Reserve a bounded block of repeats for zero-mean pairs, with deterministic fairness."""
    phase, usual = regular
    screening = [pair for key, pair in positions.items() if (
        decisions[key]["status"] != "missing_nonterminal_successor"
        and decisions[key]["completed_paired_repeats"] < settings.screening_rollouts
        and len(results[key]) < 2 * settings.max_rollouts)]
    if screening:
        screening.sort(key=lambda pair: (-decisions[pair.pair_id]["completed_paired_repeats"], pair.pair_id))
        phase, usual = "screening", screening[:capacity]
    stride = max(1, math.ceil(1 / policy.zero_fraction))
    if usual and (state.get("adaptive_rollout_batches", 0) + 1) % stride:
        return phase, usual
    targets = state.setdefault("zero_recheck_targets", {})
    allocations = state.setdefault("zero_recheck_allocations", {})
    attempts = state.setdefault("zero_recheck_attempts", {})
    selected = []
    # Finish previously reserved blocks, even if their mean has since changed.
    for key, target in sorted(targets.items(), key=lambda item: int(item[0])):
        pair = positions[int(key)]
        row = decisions[pair.pair_id]
        if (row["status"] != "missing_nonterminal_successor"
                and row["completed_paired_repeats"] < target
                and len(results[pair.pair_id]) < 2 * settings.max_rollouts
                and attempts.get(key, 0) < allocations[key]):
            selected.append(pair)
            if len(selected) == capacity:
                break
    cap = min(policy.zero_max_rollouts, settings.max_rollouts)
    eligible = [pair for key, pair in positions.items() if (
        str(key) not in targets and decisions[key]["usable"] and decisions[key]["status"] != "confirmed"
        and abs(decisions[key]["soft_z_minus_zero"]) <= 1e-8
        and decisions[key]["completed_paired_repeats"] < cap
        and len(results[key]) < 2 * settings.max_rollouts)]
    eligible.sort(key=lambda pair: (-decisions[pair.pair_id].get("gap_standard_error", 0), pair.pair_id))
    for pair in eligible:
        if len(selected) >= capacity:
            break
        row = decisions[pair.pair_id]
        # Reserve attempts, not just successes: failing beliefs cannot consume
        # an unbounded number of games from a nominally small zero-pair budget.
        allocation = min(cap - row["completed_paired_repeats"],
                         policy.zero_budget - state.get("zero_recheck_reserved", 0))
        if allocation <= 0:
            break
        targets[str(pair.pair_id)] = row["completed_paired_repeats"] + allocation
        allocations[str(pair.pair_id)] = allocation
        state["zero_recheck_reserved"] = state.get("zero_recheck_reserved", 0) + allocation
        selected.append(pair)
    return ("zero_recheck", selected[:capacity]) if selected else (phase, usual)


def proposal_pairs(choices, alternatives, *, adaptive):
    """Keep the original MCTS pair and add a different public action class."""
    zero, other = choices["zero"], choices["soft_z"]
    result = []
    if zero["action_id"] != other["action_id"]:
        result.append((choices, "mcts_value_disagreement"))
    if adaptive:
        types = {row["action_id"]: (row.get("action_type"), row.get("action_subtype")) for row in alternatives}
        baseline_type = types.get(zero["action_id"])
        ordered = sorted(alternatives, key=lambda row: (
            (row.get("action_type"), row.get("action_subtype")) == baseline_type,
            -row.get("raw_probability", 0), row["action_id"]))
        seen = {tuple(sorted((pair["zero"]["action_id"], pair["soft_z"]["action_id"]))) for pair, _ in result}
        for alternate in ordered:
            key = tuple(sorted((zero["action_id"], alternate["action_id"])))
            if alternate["action_id"] == zero["action_id"] or key in seen:
                continue
            result.append(({"zero": zero, "soft_z": alternate}, "public_action_diversity"))
            seen.add(key)
    return result
