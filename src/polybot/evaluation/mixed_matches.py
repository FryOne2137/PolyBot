"""Bounded mixed-domain arenas sharing masked policy inference batches."""
from __future__ import annotations

from collections import deque

import numpy as np
import torch
from tqdm.auto import tqdm

from ..data.full_actions import compact_actions, require_complete_actions
from ..data.polyenv_features import native_batch_to_tensors
from ..infrastructure.randomness import seed_everything
from .matches import (
    _allocate_vector_buffers, _join_tensor_rows, _new_paired_vector_envs,
    _packet_rows, _vector_capture_flags,
)


def collate_policy_packets(packets):
    """Pad storage only; original board widths remain explicit model inputs."""
    tiles = max(p["map_tokens"].shape[1] for p in packets)
    boards = []
    for packet in packets:
        tokens = packet["map_tokens"]
        if tokens.shape[1] != tiles:
            padded = tokens.new_full((len(tokens), tiles, tokens.shape[2]), -1)
            padded[:, :tokens.shape[1]] = tokens
            tokens = padded
        boards.append(tokens)
    result = {name: _join_tensor_rows(packets, name) for name in packets[0] if name != "map_tokens"}
    result["map_tokens"] = torch.cat(boards) if len(boards) > 1 else boards[0]
    return result


def _bounded_requests(bucket, max_action_tokens):
    """Split padded action storage without removing any action from a row."""
    current, count, width = [], 0, 0
    for side, index, rows, packet in bucket:
        offset = 0
        while offset < len(rows):
            next_width = max(width, packet["action_mask"].shape[1])
            capacity = max(1, max_action_tokens // next_width)
            if count >= capacity:
                yield current
                current, count, width = [], 0, 0
                continue
            take = min(len(rows) - offset, capacity - count)
            current.append((side, index, rows[offset:offset+take],
                            {name: value[offset:offset+take] for name, value in packet.items()}))
            offset += take
            count += take
            width = next_width
    if current:
        yield current


class _Arena:
    def __init__(self, group, domain, seed, first_pair, maps, training, device, *, num_threads,
                 max_actions, history, position):
        from ..training.flat_domains import annotate_domain
        self.group, self.domain = group, domain
        self.first_pair, self.maps, self.seed = first_pair, maps, seed + first_pair
        self.training = domain.training(training, self.seed, evaluation=True)
        self.position = position
        self.envs = _new_paired_vector_envs(self.training, maps, num_threads=num_threads,
            max_actions=max_actions, visible_action_history=history)
        self.allocated = [_allocate_vector_buffers(env, device) for env in self.envs]
        self.batches = [item[0] for item in self.allocated]
        for env, batch in zip(self.envs, self.batches):
            env.reset_into(batch, seed=self.seed)
            annotate_domain(batch, domain, self.seed)
            if "player_context" not in env.batch_spec():
                # Native into() calls refresh native fields only. Never cache
                # a synthesized context across observers/technology changes.
                batch.pop("player_context", None)
        self.done = np.zeros((2, maps), dtype=bool)
        self.terminated = np.zeros_like(self.done)
        self.winners = np.full((2, maps), -1, dtype=np.int32)
        self.counts = {key: np.zeros((2, maps), dtype=np.int32) for key in (
            "moves", "model_a_decisions", "model_a_city_captures", "model_a_capital_captures",
            "opponent_city_captures", "opponent_capital_captures")}

    def advance(self, selected):
        for player, (env, batch, choices) in enumerate(zip(self.envs, self.batches, selected)):
            active = ~self.done[player]
            rows = np.flatnonzero(active)
            if not len(rows):
                continue
            require_complete_actions(batch)
            if np.any(~np.asarray(batch["action_mask"], dtype=bool)[rows, choices[rows]]):
                raise RuntimeError("mixed evaluation selected a padded legal action")
            candidate_turn = np.asarray(batch["state"])[:, 1] == player
            city, capital = _vector_capture_flags(batch, choices, active)
            if self.position is not None:
                for row in np.flatnonzero(candidate_turn & active):
                    self.position(self.group, 2 * (self.first_pair + int(row)) + player, batch, int(row))
            for prefix, turn in (("model_a", candidate_turn), ("opponent", ~candidate_turn)):
                self.counts[prefix + "_city_captures"][player] += city & turn & active
                self.counts[prefix + "_capital_captures"][player] += capital & turn & active
            self.counts["model_a_decisions"][player] += candidate_turn & active
            ids = np.full(self.maps, -1, dtype=np.int32)
            ids[rows] = np.asarray(batch["action_id"], dtype=np.int32)[rows, choices[rows]]
            env.step_into(ids, batch)
            if np.any(~np.asarray(batch["action_valid"], dtype=bool)[rows]):
                raise RuntimeError("mixed evaluation selected an action rejected by PolyEnv")
            self.counts["moves"][player, rows] += 1
            terminal = np.asarray(batch["terminated"], dtype=bool) & active
            self.terminated[player] |= terminal
            self.winners[player, terminal] = np.asarray(batch["state"])[terminal, 3]
            self.done[player] |= terminal | (active & (self.counts["moves"][player] >= self.training.max_moves))

    def game_rows(self):
        return [dict(
            game_index=2 * (self.first_pair + i) + player, map_seed=self.seed + i,
            model_a_player=player, winner_player=int(self.winners[player, i]),
            score_a=.5 if self.winners[player, i] < 0 else float(self.winners[player, i] == player),
            truncated=int(not self.terminated[player, i]),
            **{key: int(values[player, i]) for key, values in self.counts.items()},
        ) for i in range(self.maps) for player in (0, 1)]


@torch.inference_mode()
def _policy_actions(model_a, model_b, arenas, device, same_policy, stats):
    batches, owners, tensors, selected = [], [], [], []
    for arena in arenas:
        for player, batch in enumerate(arena.batches):
            live = np.flatnonzero(~arena.done[player])
            batches.append(batch)
            owners.append((arena, player, live))
            selected.append(np.full(arena.maps, -1, dtype=np.int64))
            if not len(live):
                tensors.append(None)
                continue
            require_complete_actions(batch)
            # Transfer only live rows; completed games never fill GPU batches.
            packet = batch if len(live) == arena.maps else {key: value[live] for key, value in batch.items()}
            packet = compact_actions(packet)
            tensors.append(native_batch_to_tensors(packet, device, arena.domain.size, arena.domain.kind))
    routes = ((model_a, None),) if same_policy else ((model_a, True), (model_b, False))
    for model, candidate in routes:
        requests = []
        for index, ((arena, player, live), tensor) in enumerate(zip(owners, tensors)):
            if tensor is None:
                continue
            own_turn = np.asarray(batches[index]["state"])[live, 1] == player
            rows = np.arange(len(live)) if candidate is None else np.flatnonzero(own_turn == candidate)
            if len(rows):
                requests.append((arena.domain.size, index, live[rows], _packet_rows(tensor, rows)))
        # Nearby sizes share a forward. Avoid spending more than 50% extra
        # map tokens on padding, while batching different types/tribes freely.
        requests.sort(key=lambda item: item[0])
        buckets = []
        for item in requests:
            if not buckets or item[0]**2 > 1.5 * buckets[-1][0][0]**2:
                buckets.append([])
            buckets[-1].append(item)
        bounded = (part for bucket in buckets for part in _bounded_requests(bucket, 131_072))
        for bucket in bounded:
            packets = [item[3] for item in bucket]
            sides = [side for side, _, rows, _ in bucket for _ in rows]
            inputs = collate_policy_packets(packets)
            output = model(**inputs, include_belief=False,
                           board_sizes=sides if len(set(sides)) > 1 else None)
            choices = output.policy_logits.argmax(dim=-1).cpu().numpy()
            offset = 0
            for _, index, rows, _ in bucket:
                selected[index][rows] = choices[offset:offset + len(rows)]
                offset += len(rows)
            stats["model_forwards"] += 1
            stats["inference_rows"] += len(sides)
            stats["max_inference_batch"] = max(stats["max_inference_batch"], len(sides))
            stats["max_action_tokens"] = max(stats["max_action_tokens"], len(sides) * inputs["action_mask"].shape[1])
            stats["padded_tile_tokens"] += max(sides)**2 * len(sides)
            stats["real_tile_tokens"] += sum(s*s for s in sides)
    return [selected[2*i:2*i+2] for i in range(len(arenas))]


def _summary(rows):
    result = {key: float(sum(row[key] for row in rows)) for key in (
        "moves", "model_a_decisions", "model_a_city_captures", "model_a_capital_captures",
        "opponent_city_captures", "opponent_capital_captures")}
    count = len(rows)
    result.update(games=float(count),
        model_a_wins=float(sum(r["score_a"] == 1 for r in rows)),
        model_a_losses=float(sum(r["score_a"] == 0 for r in rows)),
        draws=float(sum(r["score_a"] == .5 for r in rows)),
        truncated_games=float(sum(r["truncated"] for r in rows)),
        terminated_games=float(sum(not r["truncated"] for r in rows)))
    result.update(completion_rate=result["terminated_games"] / count,
        mean_moves=result["moves"] / count,
        model_a_city_captures_per_game=result["model_a_city_captures"] / count)
    return result


def run_mixed_arenas(model_a, model_b, device, training, schedule, *, record, position=None,
                     model_a_label="policy A", model_b_label="model B", visible_action_history=0,
                     num_threads=None, max_actions=1024, max_live_games=64):
    """Advance bounded cohorts together, retaining exact seeds and paired seats.

    The limit is resident games, rather than additional evaluation games. Chunk
    boundaries and inference buckets never change a domain's map allocation.
    """
    from ..training.flat_experiments import model_fingerprint
    if not 0 <= visible_action_history <= 1024 or max_actions < 1 or training.max_moves < 1:
        raise ValueError("invalid mixed evaluation history, action capacity or move limit")
    if max_live_games < 2 or max_live_games % 2:
        raise ValueError("mixed evaluation needs an even resident game limit")
    seed_everything(training.seed)
    model_a.eval()
    model_b.eval()
    same_policy = model_a is model_b or model_fingerprint(model_a, policy_only=True) == model_fingerprint(model_b, policy_only=True)
    capacity = max_live_games // 2
    chunk = max(1, capacity // len(schedule))
    pending = deque((group, first, min(chunk, item[2] - first))
                    for first in range(0, max(item[2] for item in schedule), chunk)
                    for group, item in enumerate(schedule) if first < item[2])
    active, rows_by_group = [], [[] for _ in schedule]
    stats = dict(model_forwards=0, inference_rows=0, max_inference_batch=0, max_action_tokens=0,
                 padded_tile_tokens=0, real_tile_tokens=0, max_resident_games=0)
    games = sum(2 * item[2] for item in schedule)
    completed, turns = 0, 0
    maximum_moves = 0
    with tqdm(total=games, desc=f"evaluate mixed {model_a_label} vs {model_b_label}",
              unit="game", dynamic_ncols=True) as progress:
        while pending or active:
            resident = sum(arena.maps for arena in active)
            while pending and resident + pending[0][2] <= capacity:
                group, first, count = pending.popleft()
                domain, seed, _, _, _ = schedule[group]
                active.append(_Arena(group, domain, seed, first, count, training, device,
                    num_threads=num_threads, max_actions=max_actions, history=visible_action_history, position=position))
                resident += count
            stats["max_resident_games"] = max(stats["max_resident_games"], 2 * resident)
            choices = _policy_actions(model_a, model_b, active, device, same_policy, stats)
            for arena, actions in zip(active, choices):
                previous = int(arena.done.sum())
                arena.advance(actions)
                maximum_moves = max(maximum_moves, int(arena.counts["moves"].max()))
                newly_done = int(arena.done.sum()) - previous
                completed += newly_done
                progress.update(newly_done)
                if arena.done.all():
                    rows = arena.game_rows()
                    rows_by_group[arena.group].extend(rows)
                    for row in rows:
                        record(arena.group, row)
            # Release the loop's last arena reference as well as list entries
            # before allocating the next wave of large native action buffers.
            del arena
            active = [arena for arena in active if not arena.done.all()]
            turns += 1
            if turns % 10 == 0 or not active:
                progress.set_postfix(live=sum(int((~a.done).sum()) for a in active),
                    pending=games-completed-sum(int((~a.done).sum()) for a in active),
                    batch=stats["max_inference_batch"],
                    moves=maximum_moves)
    statistics = [(_summary(rows), item[3]) for rows, item in zip(rows_by_group, schedule)]
    mean_batch = stats["inference_rows"] / max(1, stats["model_forwards"])
    padding = stats["padded_tile_tokens"] / max(1, stats["real_tile_tokens"])
    print(f"mixed evaluation batching | mean_batch={mean_batch:.1f} | peak_batch={stats['max_inference_batch']} "
          f"| forwards={stats['model_forwards']} | tile_padding={padding:.2f}x "
          f"| shared_policy={int(same_policy)}", flush=True)
    return statistics, {"flat_evaluation_" + key: value for key, value in stats.items()} | {
        "flat_evaluation_mean_inference_batch": mean_batch,
        "flat_evaluation_tile_padding_ratio": padding,
        "flat_evaluation_shared_policy": same_policy,
    }
