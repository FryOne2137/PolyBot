"""Resume bounded collection of paired, repeatable value-action preferences."""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, replace

import numpy as np

from ..evaluation.mcts_diagnostics import _ReplayPosition
from ..evaluation.value_disagreements import (
    ARMS, PHASES, Disagreement, _branches, _compare_probes, _sample_source_positions, _save_supervised_pairs,
)
from ..infrastructure.randomness import seed_everything
from .belief_recovery import BeliefRecoveryRequired
from .flat_experiments import model_fingerprint
from .flat_value import paired_outcome_weights
from .routed_self_play_pool import RoutedPoolSearch, _evaluation_belief_retries


@dataclass(frozen=True)
class DecisivePairCollection:
    target: int = 50
    seed: int = 1838127001
    source_games: int = 256
    source_batch_games: int = 8
    probes_per_game: int = 12
    min_rollouts: int = 4
    max_rollouts: int = 16
    rollout_step: int = 4
    confidence: float = 0.95
    temperature: float = 0.5
    max_pairs_per_map: int = 2
    max_candidates_per_map: int = 8
    hours: float = 8.0
    mode: str = "confirmed"
    candidate_pool: int = 100
    screening_rollouts: int = 2
    scout_fraction: float = 0.2

    def __post_init__(self):
        if (self.target < 1 or self.seed < 1 or self.source_games < 2 or self.source_games % 2
                or self.source_batch_games < 2 or self.source_batch_games % 2 or self.probes_per_game < 1
                or self.min_rollouts < 2 or self.max_rollouts < self.min_rollouts or self.max_rollouts > 128
                or self.rollout_step < 1 or (self.max_rollouts - self.min_rollouts) % self.rollout_step
                or not 0 < self.confidence < 1 or not np.isfinite(self.temperature) or self.temperature <= 0
                or self.max_pairs_per_map < 1 or self.max_candidates_per_map < self.max_pairs_per_map
                or not np.isfinite(self.hours) or self.hours <= 0
                or self.mode not in ("confirmed", "pooled") or self.candidate_pool < 2
                or self.screening_rollouts < 2 or self.screening_rollouts > self.max_rollouts
                or not 0 < self.scout_fraction < 1):
            raise ValueError("invalid decisive-pair collection budget or uncertainty settings")


def completed_score_pairs(entries):
    complete = []
    for entry in entries:
        outcomes = entry.get("outcomes", {})
        if any(arm not in outcomes or outcomes[arm].get("score_a") is None
               or outcomes[arm].get("truncated", False) for arm in ARMS):
            continue
        values = [float(outcomes[arm]["score_a"]) for arm in ARMS]
        if any(value not in (0.0, 0.5, 1.0) for value in values):
            raise ValueError("paired rollouts must contain finite terminal scores")
        complete.append(values)
    return complete


def paired_evidence(entries, settings):
    """Exact paired sign test, with a Bonferroni budget for scheduled looks."""
    complete = completed_score_pairs(entries)
    differences = [right - left for left, right in complete]
    positive, negative = sum(delta > 0 for delta in differences), sum(delta < 0 for delta in differences)
    signs = positive + negative
    pvalue = min(1.0, 2 * sum(math.comb(signs, k) for k in range(min(positive, negative) + 1)) / 2**signs)
    looks = tuple(range(settings.min_rollouts, settings.max_rollouts + 1, settings.rollout_step))
    alpha = (1 - settings.confidence) / len(looks)
    means = {arm: float(np.mean([row[i] for row in complete])) for i, arm in enumerate(ARMS)} if complete else {}
    delta = means["soft_z"] - means["zero"] if means else None
    count = len(complete)
    status = "pending"
    if count in looks and pvalue <= alpha and delta * (positive - negative) > 0:
        status = "confirmed"
    elif count in looks and count >= max(8, settings.min_rollouts) and not signs:
        status = "no_observed_difference"
    elif count >= settings.max_rollouts:
        status = "uncertain_at_limit"
    elif len(entries) >= 2 * settings.max_rollouts:
        status = "too_many_incomplete_rollouts"
    return {
        "status": status, "completed_paired_repeats": count, "attempted_paired_repeats": len(entries),
        "mean_scores": means, "soft_z_minus_zero": delta, "positive_differences": positive,
        "negative_differences": negative, "tied_differences": count - signs,
        "sign_test_pvalue": pvalue, "per_look_alpha": alpha, "scheduled_looks": looks,
    }


def _resume_policy(saved, requested):
    """Allow the explicit legacy-to-pooled scheduling upgrade with unchanged rollouts."""
    if {key: value for key, value in saved.items() if key != "settings"} != {
        key: value for key, value in requested.items() if key != "settings"
    }:
        raise ValueError("decisive-pair directory belongs to other checkpoints or settings; use another directory")
    defaults = asdict(DecisivePairCollection())
    old = {**{key: defaults[key] for key in requested["settings"] if key not in saved["settings"]}, **saved["settings"]}
    new = requested["settings"]
    if old == new:
        return False
    scheduling = {"mode", "candidate_pool", "screening_rollouts", "scout_fraction"}
    if old["mode"] == "confirmed" and new["mode"] == "pooled" and all(
        old[key] == new[key] for key in old if key not in scheduling
    ):
        return True
    raise ValueError("decisive-pair directory belongs to other checkpoints or settings; use another directory")


def _pooled_rollout_batch(positions, results, decisions, settings, capacity):
    """Finish short screens first; refine useful nonzero gaps without starving new pairs."""
    available = [pair for key, pair in positions.items() if (
        decisions[key]["status"] != "missing_nonterminal_successor"
        and len(results[key]) < 2 * settings.max_rollouts
    )]
    screening = [pair for pair in available
                 if decisions[pair.pair_id]["completed_paired_repeats"] < settings.screening_rollouts]
    if screening and sum(row["usable"] for row in decisions.values()) < settings.target:
        screening.sort(key=lambda pair: (-decisions[pair.pair_id]["completed_paired_repeats"], pair.pair_id))
        return "screening", screening[:capacity]
    refinement = [pair for pair in available if (
        decisions[pair.pair_id]["usable"] and decisions[pair.pair_id]["status"] != "confirmed"
        and decisions[pair.pair_id]["completed_paired_repeats"] < settings.max_rollouts
        and abs(decisions[pair.pair_id]["soft_z_minus_zero"]) > 1e-8
    )]
    refinement.sort(key=lambda pair: -abs(decisions[pair.pair_id]["soft_z_minus_zero"]) / (
        (decisions[pair.pair_id]["gap_standard_error"] + 0.25)
        * (decisions[pair.pair_id]["completed_paired_repeats"] + 1)
    ))
    return "refinement", refinement[:capacity]


def _atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _journal(path):
    """Recover a partially written last record; reject corruption in earlier rows."""
    if not path.exists():
        path.touch()
    rows, offset = [], 0
    content = path.read_bytes()
    lines = content.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError(f"corrupt collection journal: {path}")
            with path.open("r+b") as handle:
                handle.truncate(offset)
            break
        offset += len(line)
    else:
        if content and not content.endswith(b"\n"):
            with path.open("ab") as handle:
                handle.write(b"\n")
    return rows


def _append(path, records):
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def collect_decisive_value_pairs(model, opponent, training, mcts, *, output, settings,
                                 excluded_map_seeds, pool):
    """Collect up to target confirmed pairs, saving every probe and paired repeat."""
    started = time.monotonic()
    deadline = started + settings.hours * 3600
    scout_deadline = started + settings.hours * 3600 * settings.scout_fraction
    training = replace(training, seed=settings.seed)
    settings_identity = {key: value for key, value in asdict(settings).items() if key != "hours"}
    identity = json.loads(json.dumps({
        "settings": settings_identity, "training": asdict(training), "mcts": asdict(mcts), "pool": pool,
        "model": model_fingerprint(model), "opponent": model_fingerprint(opponent),
        "device": str(next(model.parameters()).device),
    }))
    output.mkdir(parents=True, exist_ok=True)
    config_path, state_path = output / "config.json", output / "collection-state.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        migrated = _resume_policy(config.get("collector_identity", {}), identity)
        if migrated:
            for name in ("config.json", "summary.json", "decisions.json"):
                original = output / name
                backup = output / f"legacy-{name}"
                if original.exists() and not backup.exists():
                    backup.write_bytes(original.read_bytes())
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        if any(output.iterdir()):
            raise ValueError("new decisive-pair directory must be empty")
        state = {"reserved_map_seeds": [], "source_games": 0, "next_seed": settings.seed,
                 "pending_probes": [], "compared_probes": 0, "failed_probes": 0}
        config = {
            "schema_version": 1, "collector_identity": identity, "training": asdict(training),
            "mcts": asdict(mcts), "visible_action_history": pool["visible_action_history"],
            "source_games": 0, "reserved_map_seeds": [], "label_filter": "scheduled_paired_sign_test",
            "continuation": "fixed_mcts_0_1_vs_sampled_raw", "opponent_temperature": settings.temperature,
            "phase_definition": "thirds_of_completed_raw_source_game_by_action_index",
        }
        _atomic_json(state_path, state)
        _atomic_json(config_path, config)
    excluded = set(excluded_map_seeds)
    if set(state["reserved_map_seeds"]) & excluded:
        raise ValueError("saved decisive-pair source maps overlap currently reserved maps")
    config["collector_identity"] = identity
    if settings.mode == "pooled":
        config.update(label_mode="weighted_soft", labels_file="weighted_pairs.jsonl", pair_weighting_version=1)
    metadata = _journal(output / "pairs.jsonl")
    rollout_rows = _journal(output / "rollouts.jsonl")
    if len({row["pair_id"] for row in metadata}) != len(metadata):
        raise ValueError("duplicate decisive pair ID")
    results = {row["pair_id"]: [] for row in metadata}
    for row in rollout_rows:
        if row["pair_id"] not in results or row["repeat"] != len(results[row["pair_id"]]) + 1:
            raise ValueError("duplicate, unknown or out-of-order paired rollout")
        results[row["pair_id"]].append(row)
    positions = {
        row["pair_id"]: Disagreement(row["pair_id"], _ReplayPosition(
            row["pair_id"], row["map_seed"], row["player"], row["move"] - 1, row["root_fingerprint"],
        ), tuple(row["choices"][arm]["action_id"] for arm in ARMS)) for row in metadata
    }
    prefixes = {row["pair_id"]: row["action_prefix"] for row in metadata}
    known = {(row["map_seed"], row["player"], row["move"] - 1) for row in metadata}
    candidates_by_map = Counter(row["map_seed"] for row in metadata)
    imported = {"candidates": len(positions), "paired_repeats": len(rollout_rows),
                "completed_paired_repeats": sum(len(completed_score_pairs(rows)) for rows in results.values())}
    if settings.mode == "pooled":
        print(f"pooled value resume | candidates={len(positions)} | saved_paired_rollouts={len(rollout_rows)} | "
              f"completed={imported['completed_paired_repeats']}", flush=True)

    def decision_rows():
        confirmed_by_map = Counter()
        decisions = {}
        for key, pair in positions.items():
            evidence = paired_evidence(results[key], settings)
            if evidence["completed_paired_repeats"] and not all(
                (output / f"pair-{key:03d}-{arm}-successor.npz").is_file() for arm in ARMS
            ):
                evidence["status"] = "missing_nonterminal_successor"
            evidence["usable"] = (evidence["status"] != "missing_nonterminal_successor"
                                  and evidence["completed_paired_repeats"] >= settings.screening_rollouts)
            if evidence["completed_paired_repeats"]:
                evidence.update(paired_outcome_weights(completed_score_pairs(results[key])))
            if evidence["status"] == "confirmed":
                if settings.mode == "confirmed" and confirmed_by_map[pair.position.seed] >= settings.max_pairs_per_map:
                    evidence["status"] = "map_quota_reached"
                else:
                    confirmed_by_map[pair.position.seed] += 1
            decisions[key] = {"pair_id": key, "map_seed": pair.position.seed, **evidence}
        for key, evidence in decisions.items():
            if settings.mode == "confirmed" and evidence["status"] == "pending" and confirmed_by_map[evidence["map_seed"]] >= settings.max_pairs_per_map:
                evidence["status"] = "map_quota_reached"
        return decisions

    def checkpoint(reason):
        decisions = decision_rows()
        confirmed = [positions[key] for key, row in decisions.items() if row["status"] == "confirmed"][:settings.target]
        selected = ([positions[key] for key, row in decisions.items() if row["usable"]]
                    if settings.mode == "pooled" else confirmed)
        labels_file = "weighted_pairs.jsonl" if settings.mode == "pooled" else "labelled_pairs.jsonl"
        _save_supervised_pairs(output, selected, [decisions[pair.pair_id] for pair in selected], filename=labels_file)
        config["reserved_map_seeds"] = state["reserved_map_seeds"]
        _atomic_json(state_path, state)
        _atomic_json(config_path, config)
        _atomic_json(output / "decisions.json", decisions)
        summary = {
            "directory": str(output), "target": settings.target, "mode": settings.mode,
            "confirmed_pairs": len(confirmed), "usable_pairs": len(selected),
            "usable_informative_pairs": sum(abs(decisions[pair.pair_id]["soft_z_minus_zero"]) > 1e-8 for pair in selected),
            "confirmed_maps": len({pair.position.seed for pair in confirmed}),
            "usable_maps": len({pair.position.seed for pair in selected}), "reached_target": len(selected) >= settings.target,
            "labels_file": labels_file, "imported": imported,
            "reason": reason, "elapsed_hours_this_invocation": (time.monotonic() - started) / 3600,
            "source_games": state["source_games"], "candidates": len(positions),
            "compared_probes": state["compared_probes"], "failed_probes": state["failed_probes"],
            "attempted_paired_repeats": sum(len(rows) for rows in results.values()),
            "completed_paired_repeats": sum(row["completed_paired_repeats"] for row in decisions.values()),
            "status_counts": dict(Counter(row["status"] for row in decisions.values())),
            "reserved_map_seeds": state["reserved_map_seeds"],
            "rollout_batches": state.get("rollout_batches", 0),
            "mean_pairs_per_rollout_batch": state.get("rollout_pair_slots", 0) / max(1, state.get("rollout_batches", 0)),
        }
        _atomic_json(output / "summary.json", summary)
        return decisions, summary

    def make_search(batch):
        return RoutedPoolSearch(
            training, mcts, (model, opponent), num_envs=len(batch), num_threads=pool["num_threads"],
            max_pending_leaves=pool["max_pending_leaves"], visible_action_history=pool["visible_action_history"],
            belief_retries=_evaluation_belief_retries(4), seed=batch[0].seed,
            belief_action_mode="root_shield", leaf_value_scale=0.0, opponent_leaf_value_scale=1.0,
        )

    probe_batch_size = pool["num_envs"]
    while True:
        decisions, summary = checkpoint("collecting")
        if summary["reached_target"] and settings.mode == "confirmed":
            reason = "target_reached"
            break
        if time.monotonic() >= deadline:
            reason = "collection_time_limit"
            break
        scouting = False
        if settings.mode == "pooled":
            available_source = bool(state["pending_probes"]) or state["source_games"] < settings.source_games
            scouting = (not state.get("pooled_scout_complete", False)
                        and len(positions) < settings.candidate_pool and time.monotonic() < scout_deadline
                        and available_source)
            if not scouting:
                state["pooled_scout_complete"] = True
            phase, batch = _pooled_rollout_batch(positions, results, decisions, settings, pool["num_envs"] // 2)
            # Fill the batch while data are still being screened. Never drain
            # one candidate through sixteen repeats before looking for others.
            refill = (not summary["reached_target"] and available_source
                      and (phase != "screening" or len(batch) < pool["num_envs"] // 2))
            scouting = scouting or refill
            active = [] if scouting else batch
            state["pooled_phase"] = "scouting" if scouting else phase
            if not scouting and not active:
                reason = "refinement_complete" if summary["reached_target"] else "source_game_budget_exhausted"
                break
        else:
            active = [positions[key] for key, row in decisions.items() if row["status"] == "pending"]
        if active:
            repeat = min(len(results[pair.pair_id]) for pair in active)
            batch = (active if settings.mode == "pooled" else
                     [pair for pair in active if len(results[pair.pair_id]) == repeat][:pool["num_envs"] // 2])
            jobs = deque([batch])
            while jobs and time.monotonic() < deadline:
                batch = jobs.popleft()
                repeats = {pair.pair_id: len(results[pair.pair_id]) for pair in batch}
                state["rollout_batches"] = state.get("rollout_batches", 0) + 1
                state["rollout_pair_slots"] = state.get("rollout_pair_slots", 0) + len(batch)
                repeated = [pair.position for pair in batch for _ in ARMS]
                seed_everything(int(np.random.SeedSequence((settings.seed, batch[0].pair_id, repeat)).generate_state(1)[0]))
                try:
                    extra = {"repeat_by_pair": repeats} if settings.mode == "pooled" else {}
                    played = _branches(make_search(repeated), batch, prefixes, repeat, settings.temperature,
                                       output, save_missing_successors=True, **extra)
                    entries = [{"outcomes": dict(zip(ARMS, played[i:i + 2], strict=True))}
                               for i in range(0, len(played), 2)]
                except BeliefRecoveryRequired as error:
                    if len(batch) > 1:
                        jobs.extendleft([[pair] for pair in reversed(batch)])
                        continue
                    entries = [{"belief_error": str(error)}]
                rows = [{
                    "pair_id": pair.pair_id, "repeat": repeats[pair.pair_id] + 1,
                    "opponent_rng_seed": [settings.seed, pair.position.seed, pair.position.player,
                                          pair.position.prefix_length, repeats[pair.pair_id], 1], **entry,
                } for pair, entry in zip(batch, entries, strict=True)]
                _append(output / "rollouts.jsonl", rows)
                for row in rows:
                    results[row["pair_id"]].append(row)
            _, progress = checkpoint("collecting")
            prefix = (f"pooled value pairs | usable={progress['usable_pairs']}/{settings.target} | confirmed={progress['confirmed_pairs']}"
                      if settings.mode == "pooled" else f"decisive value pairs | confirmed={progress['confirmed_pairs']}/{settings.target}")
            print(f"{prefix} | phase={state.get('pooled_phase', 'confirmation')} | "
                  f"candidates={len(positions)} | paired_rollouts={progress['attempted_paired_repeats']} | "
                  f"mean_pair_batch={progress['mean_pairs_per_rollout_batch']:.2f} | "
                  f"elapsed={progress['elapsed_hours_this_invocation']:.2f}h", flush=True)
            continue

        if state["pending_probes"]:
            pending = state["pending_probes"][:probe_batch_size]
            batch = [replace(_ReplayPosition(**row["position"]), game=index) for index, row in enumerate(pending)]
            batch_prefixes = {index: row["prefix"] for index, row in enumerate(pending)}
            keep = [index for index, position in enumerate(batch) if (
                (position.seed, position.player, position.prefix_length) not in known
                and candidates_by_map[position.seed] < settings.max_candidates_per_map
                and (settings.mode == "pooled" or sum(row["map_seed"] == position.seed and row["status"] == "confirmed"
                        for row in decisions.values()) < settings.max_pairs_per_map
                     )
            )]
            try:
                if keep:
                    batch = [batch[index] for index in keep]
                    before = len(positions)
                    roots, _, particles = _compare_probes(make_search(batch), batch, batch_prefixes, model, capture_leaves=False)
                    state["compared_probes"] += len(batch)
                    for index, position in enumerate(batch):
                        choices = {arm: roots[f"belief_{arm}"][index] for arm in ARMS}
                        actions = tuple(choices[arm]["action_id"] for arm in ARMS)
                        if actions[0] == actions[1] or candidates_by_map[position.seed] >= settings.max_candidates_per_map:
                            continue
                        key = max(positions, default=0) + 1
                        phase = pending[position.game]["phase"]
                        particle_file = f"pair-{key:03d}-particles.npz"
                        np.savez_compressed(output / particle_file, particles=particles[index])
                        row = {"pair_id": key, "map_seed": position.seed, "player": position.player,
                               "move": position.prefix_length + 1, "root_fingerprint": position.fingerprint,
                               "action_prefix": batch_prefixes[position.game][:position.prefix_length],
                               "choices": choices, "phase": phase, "particle_file": particle_file}
                        _append(output / "pairs.jsonl", [row])
                        positions[key] = Disagreement(key, replace(position, game=key), actions)
                        prefixes[key] = row["action_prefix"]
                        results[key] = []
                        candidates_by_map[position.seed] += 1
                        known.add((position.seed, position.player, position.prefix_length))
                    if settings.mode == "pooled" and len(positions) > before:
                        print(f"pooled value scouting | candidates={len(positions)}/{settings.candidate_pool} | "
                              f"probes={state['compared_probes']} | elapsed={(time.monotonic() - started) / 3600:.2f}h", flush=True)
            except BeliefRecoveryRequired:
                if len(pending) > 1:
                    probe_batch_size = 1
                    continue
                state["failed_probes"] += 1
            del state["pending_probes"][:len(pending)]
            probe_batch_size = pool["num_envs"]
            continue

        if state["source_games"] >= settings.source_games:
            reason = "source_game_budget_exhausted"
            break
        games = min(settings.source_batch_games, settings.source_games - state["source_games"])
        cursor = state["next_seed"]
        forbidden = excluded | set(state["reserved_map_seeds"])
        while set(range(cursor, cursor + games // 2)) & forbidden:
            cursor += games // 2
        if cursor + games // 2 > 2**31:
            raise ValueError("decisive-pair source map seeds exceed the native 31-bit range")
        state["reserved_map_seeds"].extend(range(cursor, cursor + games // 2))
        state["next_seed"] = cursor + games // 2
        state["source_games"] += games
        checkpoint("source_games")
        probes, source_prefixes, _ = _sample_source_positions(
            model, opponent, replace(training, seed=cursor), mcts, games,
            num_envs=pool["num_envs"], num_threads=pool["num_threads"],
            history=pool["visible_action_history"], samples_per_game=settings.probes_per_game,
        )
        state["pending_probes"] = [{
            "position": asdict(position), "prefix": source_prefixes[position.game],
            "phase": PHASES[3 * position.prefix_length // len(source_prefixes[position.game])],
        } for position in probes]
    _, summary = checkpoint(reason)
    print(f"value collection complete | mode={settings.mode} | usable={summary['usable_pairs']}/{settings.target} | "
          f"confirmed={summary['confirmed_pairs']} | maps={summary['usable_maps']} | reason={reason}", flush=True)
    return summary
