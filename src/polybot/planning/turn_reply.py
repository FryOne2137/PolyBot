"""Score candidate first actions after a full reply in detached belief worlds."""

from dataclasses import dataclass

import numpy as np
import torch

from ..data.polyenv_features import event_action_history, packets_to_tensors, visible_events_or_empty
from ..training.belief_recovery import particle_temperature
from .belief_mcts import _public_city_constraint


def _outcome(winner: int, player: int) -> float:
    return 0.0 if winner < 0 else (1.0 if winner == player else -1.0)


def _step_world(world, action: int) -> int | None:
    valid, done, _, winner, _ = world.step_fast(action)
    if not valid:
        raise RuntimeError(f"PolyEnv rejected legal turn-reply action {action}")
    return int(winner) if done else None


@dataclass
class TurnReplyDecision:
    action_id: int
    accepted_particles: int = 0
    comparable_particles: int = 0
    simulated_actions: int = 0
    unfinished_branches: int = 0


class TurnReplySearch:
    def __init__(
        self, model, opponent, device, *, candidates, particles, visible_action_history,
        max_rollout_actions, belief_retries, belief_temperature,
    ):
        self.model = model
        self.opponent = opponent
        self.device = device
        self.candidates = candidates
        self.particles = particles
        self.visible_action_history = visible_action_history
        self.max_rollout_actions = max_rollout_actions
        self.belief_retries = belief_retries
        self.belief_temperature = belief_temperature

    def _forward(self, model, worlds, *, include_belief=False):
        packets = [world.model_request_numpy() for world in worlds]
        # Match the native MCTS model inputs. Its packets omit combat previews
        # and use a neutral event summary plus the compact visible history.
        for packet in packets:
            actions = packet["actions"]
            for field in ("damage_dealt", "damage_received"):
                actions[field] = np.full_like(actions["action_id"], -1)
        tensors = packets_to_tensors(packets, self.device, include_event_summary=False)
        histories, masks = zip(*[
            event_action_history(visible_events_or_empty(world), self.visible_action_history)
            for world in worlds
        ], strict=True)
        output = model(
            tensors.map_tokens, tensors.global_features, tensors.action_categorical, tensors.action_continuous,
            tensors.action_mask, tensors.event_features,
            action_history=torch.as_tensor(np.stack(histories), dtype=torch.float32, device=self.device),
            action_history_mask=torch.as_tensor(np.stack(masks), dtype=torch.bool, device=self.device),
            include_belief=include_belief,
        )
        return packets, tensors, output

    @torch.inference_mode()
    def policy_actions(self, worlds, *, opponent=False) -> list[int]:
        packets, tensors, output = self._forward(self.opponent if opponent else self.model, worlds)
        rows = output.policy_logits.masked_fill(~tensors.action_mask, -torch.inf).argmax(dim=1).tolist()
        return [int(packet["actions"]["action_id"][row]) for packet, row in zip(packets, rows, strict=True)]

    def _accepted_worlds(self, env, packet, tensors, logits, root_player, root_actions):
        _, city_count = _public_city_constraint(packet)
        constraints = dict(
            perspective_players=torch.tensor([root_player], device=self.device),
            known_city_counts=None if city_count is None else torch.tensor([city_count], device=self.device),
            action_categorical=tensors.action_categorical, action_mask=tensors.action_mask,
        )

        def materialize(tokens):
            try:
                world = env.make_belief_env(np.ascontiguousarray(tokens[0].cpu().numpy(), dtype=np.int32),
                                            perspective=root_player)
            except (ValueError, RuntimeError):
                return None
            if (world.is_done() or world.current_player() != root_player
                    or not np.array_equal(world.legal_action_ids_fast(), root_actions)):
                return None
            return world

        worlds = []
        for attempt in range(self.belief_retries + 1):
            for particle in range(self.particles - len(worlds)):
                tokens = self.model.sample_completed_map_batch_from_belief_logits(
                    tensors.map_tokens, logits, temperature=particle_temperature(self.belief_temperature, attempt, particle),
                    **constraints,
                )
                world = materialize(tokens)
                if world is not None:
                    worlds.append(world)
            if len(worlds) == self.particles:
                return worlds
        if not worlds:
            tokens = self.model.conservative_completed_map_batch(
                tensors.map_tokens, suppress_unlisted_bridges=True, **constraints,
            )
            world = materialize(tokens)
            if world is not None:
                worlds.append(world)
        return worlds

    def _rollout_branches(self, worlds, actions, root_player):
        # All candidates start from clones of the same hypotheses. The live
        # simulator is only used to build the detached worlds.
        branches = [world.clone() for world in worlds for _ in actions]
        scores = np.full(len(branches), np.nan, dtype=np.float64)
        observed_reply = np.zeros(len(branches), dtype=bool)
        ready = np.zeros(len(branches), dtype=bool)
        for index, branch in enumerate(branches):
            winner = _step_world(branch, int(actions[index % len(actions)]))
            if winner is not None:
                scores[index] = _outcome(winner, root_player)

        simulated_actions = len(branches)
        for depth in range(1, self.max_rollout_actions + 1):
            candidate_rows, opponent_rows = [], []
            for index in np.flatnonzero(np.isnan(scores) & ~ready):
                if branches[index].current_player() != root_player:
                    observed_reply[index] = True
                    opponent_rows.append(index)
                elif observed_reply[index]:
                    ready[index] = True
                else:
                    candidate_rows.append(index)
            if depth == self.max_rollout_actions or not (candidate_rows or opponent_rows):
                break
            for rows, opponent in ((candidate_rows, False), (opponent_rows, True)):
                if not rows:
                    continue
                chosen = self.policy_actions([branches[index] for index in rows], opponent=opponent)
                for index, action in zip(rows, chosen, strict=True):
                    winner = _step_world(branches[index], action)
                    if winner is not None:
                        scores[index] = _outcome(winner, root_player)
                simulated_actions += len(rows)
        return scores.reshape(len(worlds), len(actions)), ready.reshape(len(worlds), len(actions)), branches, simulated_actions

    def _score_branches(self, worlds, actions, root_player):
        scores, ready, branches, steps = self._rollout_branches(worlds, actions, root_player)
        if ready.any():
            _, _, output = self._forward(self.model, [branches[index] for index in np.flatnonzero(ready)])
            values = output.value.reshape(-1).tolist()
            if not np.isfinite(values).all():
                raise RuntimeError("nonfinite value after opponent reply")
            scores[ready] = values
        return scores, steps

    @torch.inference_mode()
    def decide(self, env) -> TurnReplyDecision:
        packets, tensors, output = self._forward(self.model, [env], include_belief=True)
        packet = packets[0]
        legal = packet["actions"]["action_id"]
        logits = output.policy_logits[0, :len(legal)].float().cpu().numpy()
        actions = legal[np.argsort(-logits, kind="stable")[:self.candidates]]
        fallback = int(actions[0])
        if len(actions) == 1:
            return TurnReplyDecision(fallback)

        root_player = env.current_player()
        worlds = self._accepted_worlds(env, packet, tensors, output.belief_logits, root_player, legal)
        if not worlds:
            return TurnReplyDecision(fallback)
        scores, steps = self._score_branches(worlds, actions, root_player)
        # Dropping an incomplete branch for only one candidate changes the
        # sampled world distribution. Keep the whole world or drop it.
        comparable = np.isfinite(scores).all(axis=1)
        chosen = int(actions[scores[comparable].mean(axis=0).argmax()]) if comparable.any() else fallback
        return TurnReplyDecision(chosen, len(worlds), int(comparable.sum()), steps, int(np.isnan(scores).sum()))
