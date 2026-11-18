"""Compare terminal labels with turn-aligned search values on matched inputs."""

from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np
import torch
from tqdm.auto import tqdm

from ..data.full_actions import exact_mcts_pool
from ..data.polyenv_features import event_action_history, native_batch_to_tensors, visible_events_or_empty
from ..evaluation.matches import _new_env
from ..evaluation.mcts_diagnostics import _play_replayed_continuations, _replay_positions, _ReplayPosition
from ..evaluation.public_rollouts import PublicRolloutPool
from ..infrastructure.randomness import seed_everything
from ..planning.turn_reply import TurnReplySearch, _step_world
from .flat_value import EncodedValuePosition, VALUE_PHASES, fit_value_head
from .mcts_value import collect_mcts_terminal_value_games


@dataclass(frozen=True)
class ValueRoot:
    example: EncodedValuePosition
    position: _ReplayPosition

    def metadata(self, prefix):
        return {
            "map_seed": self.position.seed, "player": self.position.player,
            "action_prefix": prefix[:self.position.prefix_length],
            "fingerprint": self.position.fingerprint,
            "turn": self.example.turn, "phase": self.example.phase,
            "terminal_target": self.example.return_target,
        }


def collect_value_roots(model, opponent, training, mcts, games, *, positions_per_game,
                        excluded_map_seeds, label, raw=False, **pool):
    roots = []
    prefixes = {}

    def record(example, position, prefix):
        roots.append(ValueRoot(example, position))
        prefixes[position.game] = prefix

    _, info = collect_mcts_terminal_value_games(
        model, opponent, training, mcts, games, positions_per_game=positions_per_game,
        excluded_map_seeds=excluded_map_seeds, label=label, turn_starts_only=True,
        model_a_uses_mcts=not raw, replace_incomplete=not raw, position_callback=record,
        **pool,
    )
    return roots, prefixes, info


@dataclass
class RepliedBranches:
    actions: np.ndarray
    priors: np.ndarray
    scores: np.ndarray
    ready: np.ndarray
    leaves: list
    accepted_worlds: int


def _replied_positions(search, training, roots, prefixes, *, candidates):
    """Replay sampled roots once per source game and use only detached hypotheses."""
    search.model.eval()
    search.opponent.eval()
    seed_everything(training.seed)
    ordered = sorted(roots, key=lambda item: (item.position.game, item.position.prefix_length))
    game = None
    for root in ordered:
        position = root.position
        if game != position.game:
            game, move = position.game, 0
            env = _new_env(position.seed, training, visible_action_history=search.visible_action_history)
        prefix = prefixes[game]
        for action in prefix[move:position.prefix_length]:
            if _step_world(env, action) is not None:
                raise RuntimeError("sampled value root follows a terminal action")
        move = position.prefix_length
        yield root, _replied_branches(search, root, env, candidates)


@torch.inference_mode()
def _replied_branches(search, root, env, candidates):
    position = root.position
    packets, tensors, output = search._forward(search.model, [env], include_belief=True)
    packet = packets[0]
    # Scalar replay and the native collector must describe the same input.
    if (env.current_player() != position.player or packet["obs"]["turn"] != root.example.turn
            or not np.allclose(output.state[0].float().cpu().numpy(), root.example.features, atol=2e-4, rtol=2e-4)):
        raise RuntimeError(f"value replay differs on seed {position.seed}, player {position.player}, "
                           f"move {position.prefix_length + 1}")
    legal = packet["actions"]["action_id"]
    logits = output.policy_logits[0, :len(legal)].float().cpu().numpy()
    order = np.argsort(-logits, kind="stable")[:candidates]
    actions = legal[order]
    weights = np.exp(logits[order] - logits[order].max(), dtype=np.float64)
    priors = weights / weights.sum()
    worlds = search._accepted_worlds(env, packet, tensors, output.belief_logits, position.player, legal)
    scores, ready, branches, _ = search._rollout_branches(worlds, actions, position.player)
    complete = (np.isfinite(scores) | ready).all(axis=1)
    leaves = [branches[i] for i in np.flatnonzero(ready & complete[:, None])]
    return RepliedBranches(actions, priors, scores[complete], ready[complete], leaves, len(worlds))


def _turn_root_visits(values, priors, simulations, c_puct):
    """Allocate root visits by PUCT over already searched full-reply actions."""
    visits = np.ones(len(values), dtype=np.int32)
    for count in range(len(values), simulations):
        row = np.argmax(values + c_puct * priors * np.sqrt(count) / (1 + visits))
        visits[row] += 1
    return visits


@torch.inference_mode()
def _deep_reply_values(search: TurnReplySearch, worlds, training, mcts, *, simulations, num_threads,
                       opponent_value_scale):
    """Search beyond the reply in each detached world and return root values."""
    from PolyEnv import MctsPool

    histories, masks = zip(*[
        event_action_history(visible_events_or_empty(world), search.visible_action_history) for world in worlds
    ], strict=True)
    histories, masks = np.stack(histories), np.stack(masks)
    root_player = worlds[0].current_player()
    pool = exact_mcts_pool(MctsPool, worlds, num_threads=num_threads, c_puct=mcts.c_puct)
    buffers = {}
    for _ in range(simulations + 1):  # The first wave expands the roots.
        count = pool.select_leaves_into(buffers)
        if not count:
            break
        leaves = {name: values[:count] for name, values in buffers.items() if name != "action_arg_mask"}
        candidate_rows = leaves["to_play"] == root_player
        tree_ids = leaves["tree_id"]
        leaf_history = histories[tree_ids]
        leaf_history[~candidate_rows] = 0
        leaves["visible_action_history"] = leaf_history
        leaves["visible_action_history_mask"] = masks[tree_ids] & candidate_rows[:, None]
        tensors = native_batch_to_tensors(leaves, search.device, training.map_size, training.map_type)
        logits = np.empty(leaves["action_mask"].shape, dtype=np.float32)
        values = np.empty(len(tree_ids), dtype=np.float32)
        for rows, model, scale in (
            (candidate_rows, search.model, 1.0), (~candidate_rows, search.opponent, opponent_value_scale),
        ):
            indices = np.flatnonzero(rows)
            if not indices.size:
                continue
            index = torch.as_tensor(indices, device=search.device)
            output = model(
                tensors.map_tokens[index], tensors.global_features[index], tensors.action_categorical[index],
                tensors.action_continuous[index], tensors.action_mask[index], tensors.event_features[index],
                action_history=tensors.action_history[index], action_history_mask=tensors.action_history_mask[index],
                include_belief=False,
            )
            logits[indices] = output.policy_logits.float().cpu().numpy()
            values[indices] = output.value.reshape(-1).float().cpu().numpy() * scale
        if not np.isfinite(values).all():
            raise RuntimeError("nonfinite teacher leaf value")
        pool.expand_and_backup(leaves["leaf_id"], logits, values)
    roots = pool.root_policy()
    if np.any(roots["root_visit_count"] < 2) or not np.isfinite(roots["root_value"]).all():
        raise RuntimeError("deep teacher did not back up a finite value for every post-reply root")
    return roots["root_value"], roots["root_visit_count"]


def label_soft_z(search, roots, prefixes, training, mcts, *, simulations, num_threads,
                 opponent_value_scale, record, label):
    datasets = {"terminal": [], "soft_z": []}
    stats = dict(sampled=len(roots), matched=0, unavailable=0, accepted_worlds=0, comparable_worlds=0)
    positions = _replied_positions(search, training, roots, prefixes, candidates=search.candidates)
    for root, branches in tqdm(positions, desc=f"{label} soft-Z labels", total=len(roots), unit="position"):
        stats["accepted_worlds"] += branches.accepted_worlds
        stats["comparable_worlds"] += len(branches.scores)
        row = root.metadata(prefixes[root.position.game])
        row.update(actions=branches.actions.tolist(), comparable_worlds=len(branches.scores),
                   accepted_worlds=branches.accepted_worlds)
        if not len(branches.scores):
            stats["unavailable"] += 1
            record({**row, "soft_z_target": None})
            continue
        backed_up = np.zeros(branches.ready.shape, dtype=np.int32)
        if branches.leaves:
            branches.scores[branches.ready], backed_up[branches.ready] = _deep_reply_values(
                search, branches.leaves, training, mcts, simulations=simulations, num_threads=num_threads,
                opponent_value_scale=opponent_value_scale,
            )
        q = branches.scores.mean(axis=0)
        visits = _turn_root_visits(q, branches.priors, simulations, mcts.c_puct)
        target = float(np.average(q, weights=visits))
        datasets["terminal"].append(root.example)
        datasets["soft_z"].append(replace(root.example, return_target=target))
        stats["matched"] += 1
        record({**row, "soft_z_target": target, "action_values": q.tolist(),
                "root_action_visits": visits.tolist(), "post_reply_root_visits": backed_up.tolist()})
    stats["maps"] = len({example.game_key[0] for example in datasets["terminal"]})
    stats["phases"] = {phase: sum(e.phase == phase for e in datasets["terminal"]) for phase in VALUE_PHASES}
    return datasets, stats


def fit_matched_value_heads(model, training, datasets, settings, *, seed):
    """Use identical initial weights, examples and minibatch draws for both targets."""
    original = deepcopy(model.value_head.state_dict())
    heads, fit = {}, {}
    try:
        for name in ("terminal", "soft_z"):
            model.value_head.load_state_dict(original)
            fit[name] = fit_value_head(
                model, training, datasets["train"][name], datasets["selection"][name],
                settings, seed=seed, validation_source=f"turn_{name}_selection", keep_best_update=True,
            )
            if fit[name]["kept"]:
                heads[name] = deepcopy(model.value_head.state_dict())
    finally:
        model.value_head.load_state_dict(original)
    return heads, fit


@torch.inference_mode()
def _rank_with_heads(search, branches, heads):
    scores = branches.scores.copy()
    scores[branches.ready] = 0.0
    result = {"zero": scores.mean(axis=0).tolist()}
    head = deepcopy(search.model.value_head)
    if branches.leaves:
        _, _, output = search._forward(search.model, branches.leaves)
    for name, state in heads.items():
        if branches.leaves:
            head.load_state_dict(state)
            scores[branches.ready] = head(output.state).reshape(-1).float().cpu().numpy()
        if not np.isfinite(scores).all():
            raise RuntimeError(f"nonfinite ranking values for {name}")
        result[name] = scores.mean(axis=0).tolist()
    return result


def _preference_metrics(pairs, arm):
    target = np.asarray([row["terminal_gap"] for row in pairs])
    estimated = np.asarray([row["estimated_gaps"][arm] for row in pairs])
    informative = np.abs(target) > 1e-8
    tied = np.abs(estimated) <= 1e-8
    correct = informative & ~tied & (np.sign(target) == np.sign(estimated))
    return {
        "pairs": len(pairs), "informative_pairs": int(informative.sum()),
        "reference_ties": int((~informative).sum()),
        "predicted_ties": int(tied.sum()), "correct_preferences": int(correct.sum()),
        "preference_accuracy": float(correct.sum() / informative.sum()) if informative.any() else None,
        "gap_rmse": float(np.sqrt(np.square(estimated - target).mean())) if pairs else None,
    }


def evaluate_move_preferences(search, heads, training, roots, prefixes, *, num_envs, positions,
                              branch_rollouts, rollout_temperature, record):
    """Compare post-reply ordering with paired terminal continuations on real maps."""
    rng = np.random.default_rng((training.seed, 0x52414E4B))
    per_phase = {
        phase: iter(rng.permutation([i for i, root in enumerate(roots) if root.example.phase == phase]).tolist())
        for phase in VALUE_PHASES
    }
    selected = []
    while len(selected) < positions:
        before = len(selected)
        for bucket in per_phase.values():
            index = next(bucket, None)
            if index is not None and len(selected) < positions:
                selected.append(roots[index])
        if before == len(selected):
            break
    diagnostics = []
    candidates = _replied_positions(search, training, selected, prefixes, candidates=2)
    for root, branches in tqdm(candidates, total=len(selected), desc="score held-out move pairs", unit="position"):
        row = root.metadata(prefixes[root.position.game])
        row.update(actions=branches.actions.tolist(), comparable_worlds=len(branches.scores))
        if len(branches.scores):
            diagnostics.append((root.position, row, _rank_with_heads(search, branches, heads)))
        else:
            record({**row, "reason": "no_comparable_worlds"})

    measured = []
    completed_repeats = 0
    batch_pairs = num_envs // 2
    for start in tqdm(range(0, len(diagnostics), batch_pairs), desc="play paired ranking continuations", unit="batch"):
        chunk = diagnostics[start:start + batch_pairs]
        paired_positions = [position for position, _, _ in chunk for _ in range(2)]
        actions = [action for _, row, _ in chunk for action in row["actions"]]
        pool = PublicRolloutPool(training, (search.model, search.opponent), num_envs=len(actions),
                                 visible_action_history=search.visible_action_history)
        logs = [[] for _ in chunk]
        for repeat in range(branch_rollouts):
            request = _replay_positions(pool, paired_positions, prefixes)
            seeds = [
                (training.seed, position.seed, position.player, position.prefix_length, repeat)
                for position in paired_positions
            ]
            outcomes = _play_replayed_continuations(
                pool, request, paired_positions, actions, training.max_moves,
                raw_rngs=[np.random.default_rng(seed) for seed in seeds], raw_temperature=rollout_temperature,
            )
            for index, repeats in enumerate(logs):
                repeats.append({"rng_seed": seeds[2 * index], "outcomes": outcomes[2 * index:2 * index + 2]})
        for (_, row, estimates), repeats in zip(chunk, logs, strict=True):
            result = {**row, "estimates": estimates, "continuations": repeats}
            terminal = [[outcome["score_a"] for outcome in repeat["outcomes"]] for repeat in repeats
                        if all(not outcome["truncated"] for outcome in repeat["outcomes"])]
            completed_repeats += len(terminal)
            if terminal:
                target = 2.0 * np.mean(terminal, axis=0) - 1.0
                result.update(
                    terminal_values=target.tolist(), terminal_gap=float(target[0] - target[1]),
                    estimated_gaps={name: values[0] - values[1] for name, values in estimates.items()},
                )
                measured.append(result)
            else:
                result["reason"] = "no_terminal_rollout_pair"
            record(result)
    return {
        "valid": any(abs(row["terminal_gap"]) > 1e-8 for row in measured),
        "reference_policy": "frozen_raw_sampled", "candidate_actions": "top_two_frozen_policy",
        "selected_positions": len(selected), "comparable_world_positions": len(diagnostics),
        "positions_with_terminal_pairs": len(measured), "branch_rollouts_per_move": branch_rollouts,
        "completed_rollout_pairs": completed_repeats,
        "rollout_temperature": rollout_temperature, "seed": training.seed,
        "measured_map_seeds": sorted({row["map_seed"] for row in measured}),
        "arms": {name: _preference_metrics(measured, name) for name in (*heads, "zero")},
    }
