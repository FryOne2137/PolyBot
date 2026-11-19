"""Collect same-particle value disagreements and test their forced actions."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from ..infrastructure.randomness import seed_everything
from ..training.belief_recovery import BeliefRecoveryRequired
from ..training.mcts_value import _encode_value_rows
from ..training.routed_self_play_pool import RoutedPoolSearch, _evaluation_belief_retries, evaluate_matches_pool
from .mcts_diagnostics import (
    _ReplayPosition, _compare_leaf_scales, _paired_rollout_rngs, _play_replayed_continuations,
    _position_fingerprint, _replay_positions,
)


ARMS = ("zero", "soft_z")
PHASES = ("opening", "mid", "late")
PUBLIC_FIELDS = (
    "map_tokens", "state", "own_techs", "own_tribe", "initial_player_count", "player_scores",
    "player_scores_mask", "visible_action_history", "visible_action_history_mask", "to_play", "agent_id",
)


@dataclass(frozen=True)
class Disagreement:
    pair_id: int
    position: _ReplayPosition
    actions: tuple[int, int]


def _sample_source_positions(model, opponent, training, mcts, games, *, num_envs, num_threads,
                             history, samples_per_game):
    prefixes = defaultdict(list)
    candidates = defaultdict(list)
    games_played = set()
    rng = np.random.default_rng(training.seed ^ 0x44495341)
    schedule = [training.seed + index // 2 for index in range(games)]

    def action_callback(indices, packet, actions):
        for row in np.flatnonzero(actions >= 0):
            game = int(indices[row])
            prefix = prefixes[game]
            if packet["agent_id"][row] == 0 and np.count_nonzero(packet["action_mask"][row]) >= 2:
                candidates[game].append(_ReplayPosition(
                    game, int(packet["map_seed"][row]), int(packet["to_play"][row]), len(prefix),
                    _position_fingerprint(packet, row),
                ))
            prefix.append(int(actions[row]))

    def game_callback(game):
        if not game["truncated"]:
            games_played.add(int(game["game_index"]))

    report = evaluate_matches_pool(
        model, opponent, training, mcts, games, max_envs=num_envs, num_threads=num_threads,
        visible_action_history=history, model_a_uses_mcts=False, model_b_uses_mcts=False,
        map_seeds=schedule, replace_failed_pairs=False, model_a_label="disagreement source raw",
        model_b_label="raw opponent", action_callback=action_callback, game_callback=game_callback,
    )
    if not report["valid"] or report["evaluated_map_seeds"] != schedule:
        raise RuntimeError("raw disagreement source did not play the fixed schedule")

    sampled = []
    # Visit one probe per source game before taking another from the same game.
    for game in sorted(games_played):
        buckets = [[] for _ in PHASES]
        for position in candidates[game]:
            buckets[3 * position.prefix_length // len(prefixes[game])].append(position)
        for bucket in buckets:
            rng.shuffle(bucket)
        offset = game % len(PHASES)
        buckets = buckets[offset:] + buckets[:offset]
        selected = []
        for _ in range(samples_per_game):
            for bucket in buckets:
                if bucket and len(selected) < samples_per_game:
                    selected.append(bucket.pop())
        sampled.append(selected)
    probes = [rows[index] for index in range(samples_per_game) for rows in sampled if index < len(rows)]
    return probes, prefixes, {"terminal_source_games": len(games_played), "map_seeds": schedule[::2]}


def _public_row(packet, row):
    data = {key: np.array(packet[key][row], copy=True) for key in PUBLIC_FIELDS}
    mask = packet["action_mask"][row].astype(bool)
    data["action_id"] = packet["action_id"][row, mask]
    data["action_features"] = packet["action_features"][row, mask]
    return data


def _pack_rows(rows):
    counts = [len(row["action_id"]) for row in rows]
    return {
        "action_offsets": np.cumsum([0, *counts], dtype=np.int32),
        **{key: np.concatenate([row[key] for row in rows]) if key in {"action_id", "action_features"}
           else np.stack([row[key] for row in rows]) for key in rows[0]},
    }


def _compare_probes(search, batch, prefixes, model, *, capture_leaves=True):
    request = _replay_positions(search, batch, prefixes, recover_beliefs=True)
    if np.any(request["agent_id"] != 0):
        raise RuntimeError("value disagreement probes must start on the apprentice's decision")
    leaves = defaultdict(list)
    particles = []

    def capture(arm, packet, rows, features, prediction, backed_up, is_root):
        states = features.float().cpu().numpy()
        predictions = prediction.reshape(-1).float().cpu().numpy()
        for local, row in enumerate(rows):
            data = _public_row(packet, row)
            data.update(
                features=states[local], prediction=predictions[local], backed_up_value=backed_up[local],
                is_root=is_root, leaf_id=packet["leaf_id"][row], tree_id=packet["tree_id"][row],
            )
            leaves[int(packet["env_id"][row]), arm].append(data)

    roots = _compare_leaf_scales(
        search, request, include_true_map=False, variants=(("zero", 0.0, model), ("soft_z", 1.0, model)),
        observe_leaves=capture if capture_leaves else None, observe_particles=lambda worlds: particles.append(worlds),
    )
    return roots, leaves, particles[0]


def _branches(search, pairs, prefixes, repeat, temperature, output, *, save_missing_successors=False):
    positions = [pair.position for pair in pairs for _ in ARMS]
    actions = [action for pair in pairs for action in pair.actions]
    request = _replay_positions(search, positions, prefixes, recover_beliefs=True)

    def save_successors(packet):
        if repeat and not save_missing_successors:
            return
        live = np.flatnonzero(~packet["terminated"].astype(bool))
        if save_missing_successors:
            live = np.array([row for row in live if not (
                output / f"pair-{pairs[row // 2].pair_id:03d}-{ARMS[row % 2]}-successor.npz"
            ).is_file()], dtype=np.int32)
        features = _encode_value_rows(search.models[0], search.training, search.device, packet, live) if live.size else []
        for row, encoded in zip(live, features, strict=True):
            pair = pairs[row // 2]
            arm = ARMS[row % 2]
            # These are real continuation states. Simulated leaves have their
            # own observations and do not inherit the forced action's outcome.
            np.savez_compressed(output / f"pair-{pair.pair_id:03d}-{arm}-successor.npz",
                                **_public_row(packet, row), features=encoded)

    return _play_replayed_continuations(
        search, request, positions, actions, search.training.max_moves, mcts_value_scale=0.0,
        after_first_action=save_successors,
        opponent_rngs=_paired_rollout_rngs(search.training.seed, positions, repeat, 1),
        raw_temperature=temperature,
    )


def _pair_results(pairs, results, repeats):
    rows = []
    by_map = defaultdict(list)
    for pair in pairs:
        complete = [result["outcomes"] for result in results[pair.pair_id] if result.get("outcomes")
                    and all(branch["score_a"] is not None for branch in result["outcomes"].values())]
        means = {arm: float(np.mean([repeat[arm]["score_a"] for repeat in complete])) for arm in ARMS} if complete else {}
        delta = means["soft_z"] - means["zero"] if means else None
        if delta is not None:
            by_map[pair.position.seed].append(delta)
        rows.append({
            "pair_id": pair.pair_id, "map_seed": pair.position.seed, "completed_paired_repeats": len(complete),
            "mean_scores": means, "soft_z_minus_zero": delta,
        })
    deltas = [row["soft_z_minus_zero"] for row in rows if row["soft_z_minus_zero"] is not None]
    map_deltas = {seed: float(np.mean(values)) for seed, values in by_map.items()}
    return {
        "measured_pairs": len(deltas), "fully_measured_pairs": sum(row["completed_paired_repeats"] == repeats for row in rows),
        "planned_paired_repeats": len(pairs) * repeats,
        "belief_failed_repeats": sum("belief_error" in row for group in results.values() for row in group),
        "completed_paired_repeats": sum(row["completed_paired_repeats"] for row in rows),
        "soft_z_better": sum(delta > 0.0 for delta in deltas),
        "zero_better": sum(delta < 0.0 for delta in deltas), "reference_ties": deltas.count(0.0),
        "mean_score_delta": float(np.mean(deltas)) if deltas else None,
        "mean_score_delta_by_map": map_deltas,
        "map_balanced_score_delta": float(np.mean(list(map_deltas.values()))) if map_deltas else None,
        "positions": rows,
    }


def _save_supervised_pairs(output, pairs, score_rows):
    """Save labels for the real successors whose outcomes were paired."""
    labelled = 0
    with (output / "labelled_pairs.jsonl").open("w", encoding="utf-8") as stream:
        for pair, scores in zip(pairs, score_rows, strict=True):
            paths = [output / f"pair-{pair.pair_id:03d}-{arm}-successor.npz" for arm in ARMS]
            if not scores["completed_paired_repeats"] or not all(path.is_file() for path in paths):
                continue
            branches = {}
            for arm, action, path in zip(ARMS, pair.actions, paths, strict=True):
                with np.load(path, allow_pickle=False) as state:
                    actor = int(state["to_play"].item())
                sign = 1 if actor == pair.position.player else -1
                branches[arm] = {
                    "action_id": action, "successor_file": path.name, "to_play": actor,
                    "to_root_sign": sign, "return_target": sign * (2.0 * scores["mean_scores"][arm] - 1.0),
                }
            stream.write(json.dumps({**scores, "player": pair.position.player, "branches": branches},
                                    allow_nan=False) + "\n")
            labelled += 1
    return labelled


def evaluate_value_disagreements(model, opponent, training, mcts, games, *, output: Path,
                                 checkpoints, target, samples_per_game, repeats, temperature,
                                 num_envs, num_threads, max_pending_leaves, history, belief_retries):
    """Run the diagnostic without changing either checkpoint."""
    output.mkdir(parents=True, exist_ok=False)
    model.eval()
    opponent.eval()
    report = {
        "schema_version": 1, "checkpoints": checkpoints, "training": asdict(training), "mcts": asdict(mcts),
        "target_disagreements": target, "rollouts_per_action": repeats, "visible_action_history": history,
        "source_games": games, "samples_per_source_game": samples_per_game, "max_disagreements_per_source_game": 3,
        "num_envs": num_envs, "num_threads": num_threads, "max_pending_leaves": max_pending_leaves,
        "belief_retries": belief_retries,
        "continuation": "fixed_mcts_0_1_vs_sampled_raw", "opponent_temperature": temperature,
        "opponent_tree_value_scale": 1.0, "output_dir": str(output), "source_policy": "raw_argmax",
    }
    (output / "config.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    probes, prefixes, report["source"] = _sample_source_positions(
        model, opponent, training, mcts, games, num_envs=num_envs, num_threads=num_threads,
        history=history, samples_per_game=samples_per_game,
    )

    def make_search(batch):
        return RoutedPoolSearch(
            training, mcts, (model, opponent), num_envs=len(batch), num_threads=num_threads,
            max_pending_leaves=max_pending_leaves, visible_action_history=history,
            belief_retries=_evaluation_belief_retries(belief_retries), seed=batch[0].seed,
            belief_action_mode="root_shield", leaf_value_scale=0.0, opponent_leaf_value_scale=1.0,
        )

    seed_everything(training.seed)
    pairs = []
    by_game = defaultdict(int)
    compared = failed_probes = 0
    report["probe_budget"] = len(probes)
    chunks = deque(probes[start:start + num_envs] for start in range(0, len(probes), num_envs))
    with (output / "pairs.jsonl").open("w", encoding="utf-8", buffering=1) as stream, tqdm(
        total=target, desc="collect same-particle disagreements", unit="pair",
    ) as progress:
        while chunks and len(pairs) < target:
            batch = [position for position in chunks.popleft() if by_game[position.game] < 3]
            if not batch:
                continue
            try:
                roots, leaves, particles = _compare_probes(make_search(batch), batch, prefixes, model)
            except BeliefRecoveryRequired:
                if len(batch) > 1:
                    chunks.appendleft(batch[1:])
                    chunks.appendleft(batch[:1])
                else:
                    failed_probes += 1
                continue
            compared += len(batch)
            for index, position in enumerate(batch):
                choices = {arm: roots[f"belief_{arm}"][index] for arm in ARMS}
                zero, soft_z = (choices[arm]["action_id"] for arm in ARMS)
                if zero == soft_z or by_game[position.game] >= 3:
                    continue
                pair = Disagreement(len(pairs) + 1, position, (zero, soft_z))
                prefix = prefixes[position.game]
                leaf_path = f"pair-{pair.pair_id:03d}-leaves.npz"
                archive = {"particles": particles[index]}
                for arm in ARMS:
                    archive.update({f"{arm}_{key}": value for key, value in _pack_rows(leaves[index, f"belief_{arm}"]).items()})
                np.savez_compressed(output / leaf_path, **archive)
                stream.write(json.dumps({
                    "pair_id": pair.pair_id, "map_seed": position.seed, "player": position.player,
                    "move": position.prefix_length + 1, "root_fingerprint": position.fingerprint,
                    "action_prefix": prefix[:position.prefix_length], "choices": choices, "leaf_file": leaf_path,
                    "phase": PHASES[3 * position.prefix_length // len(prefix)],
                    "particles_sha256": sha256(archive["particles"].tobytes()).hexdigest(),
                }, allow_nan=False) + "\n")
                pairs.append(pair)
                by_game[position.game] += 1
                progress.update(1)
                if len(pairs) == target:
                    break
            progress.set_postfix(probes=compared, belief_failures=failed_probes)

    results = defaultdict(list)
    with (output / "rollouts.jsonl").open("w", encoding="utf-8", buffering=1) as stream, tqdm(
        total=len(pairs) * repeats, desc="paired MCTS continuations", unit="pair",
    ) as progress:
        count = num_envs // 2
        for repeat in range(repeats):
            chunks = deque(pairs[start:start + count] for start in range(0, len(pairs), count))
            while chunks:
                batch = chunks.popleft()
                repeated = [pair.position for pair in batch for _ in ARMS]
                seed_everything(int(np.random.SeedSequence((training.seed, batch[0].pair_id, repeat)).generate_state(1)[0]))
                try:
                    played = _branches(make_search(repeated), batch, prefixes, repeat, temperature, output)
                except BeliefRecoveryRequired as error:
                    if len(batch) > 1:
                        chunks.appendleft(batch[1:])
                        chunks.appendleft(batch[:1])
                        continue
                    entries = [{"belief_error": str(error)}]
                else:
                    entries = [{"outcomes": dict(zip(ARMS, played[i:i + 2], strict=True))}
                               for i in range(0, len(played), 2)]
                for pair, entry in zip(batch, entries, strict=True):
                    pos = pair.position
                    row = {"pair_id": pair.pair_id, "repeat": repeat + 1,
                           "opponent_rng_seed": [training.seed, pos.seed, pos.player, pos.prefix_length, repeat, 1],
                           **entry}
                    results[pair.pair_id].append(row)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                progress.update(len(batch))

    report.update(collected_disagreements=len(pairs), reached_target=len(pairs) == target,
                  compared_positions=compared, belief_failed_probes=failed_probes,
                  **_pair_results(pairs, results, repeats))
    report["labelled_successor_pairs"] = _save_supervised_pairs(output, pairs, report["positions"])
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
