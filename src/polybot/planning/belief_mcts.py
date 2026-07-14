"""Policy-guided PUCT using only detached PolyEnv belief worlds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..config import MCTSConfig
from ..data.polyenv_features import (
    packet_to_tensors,
    packets_to_tensors,
    strategic_action_bias,
    visible_events_or_empty,
)
from ..models.policy_value_belief import PolicyValueBeliefNet
from .batched_inference import BatchedInference


@dataclass
class SearchNode:
    priors: Dict[int, float] = field(default_factory=dict)
    children: Dict[int, "SearchNode"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    # Temporary reservations prevent all leaves in one GPU batch from selecting
    # the same unexpanded child before any of their values are backed up.
    virtual_visits: int = 0
    to_play: Optional[int] = None

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def select(self, c_puct: float, root_player: int) -> int:
        parent_scale = float(np.sqrt(max(1, self.visits + self.virtual_visits)))
        best_action, best_score = None, -float("inf")
        for action_id, prior in self.priors.items():
            child = self.children.get(action_id)
            child_visits = (child.visits + child.virtual_visits) if child else 0
            child_value = child.value if child else 0.0
            # Values are stored from root_player's perspective.  At an
            # opponent node, choose the action worst for root_player.
            if self.to_play is not None and self.to_play != root_player:
                child_value = -child_value
            score = child_value + c_puct * prior * parent_scale / (1.0 + child_visits)
            if score > best_score:
                best_action, best_score = action_id, score
        if best_action is None:
            raise RuntimeError("cannot select from an unexpanded node")
        return best_action


@dataclass
class MCTSDecision:
    action_id: int
    action_row: int
    visit_counts: np.ndarray
    priors: np.ndarray
    valid_particles: int


@dataclass
class PendingLeaf:
    """A selected leaf waiting for a batched neural-network evaluation."""

    node: SearchNode
    path: List[SearchNode]
    packet: Dict[str, object]
    acting_player: int


class BeliefMCTS:
    """MCTS that never calls clone() on the authoritative environment.

    ``clone()`` is used only after ``make_belief_env()`` has already produced a
    detached particle world.  Such clones contain predicted state, never the
    source environment's hidden state.
    """

    def __init__(
        self,
        model: Optional[PolicyValueBeliefNet],
        device: Optional[torch.device],
        config: MCTSConfig,
        inference: Optional[BatchedInference] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.config = config
        self.inference = inference
        self._reported_particle_errors: set[str] = set()

    @torch.no_grad()
    def _evaluate_packet(
        self,
        packet: Dict[str, object],
        events: Dict[str, object] | None = None,
    ) -> Tuple[Dict[int, float], float]:
        return self._evaluate_packets([packet], [events])[0]

    @torch.no_grad()
    def _evaluate_packets(
        self,
        packets: List[Dict[str, object]],
        events: List[Dict[str, object] | None] | None = None,
    ) -> List[Tuple[Dict[int, float], float]]:
        if self.inference is not None:
            return self.inference.evaluate(packets, events)
        return self._evaluate_packets_direct(packets, events)

    @torch.no_grad()
    def _evaluate_packets_direct(
        self,
        packets: List[Dict[str, object]],
        events: List[Dict[str, object] | None] | None = None,
    ) -> List[Tuple[Dict[int, float], float]]:
        """Evaluate variable-action MCTS leaves in one GPU forward pass."""
        if self.model is None or self.device is None:
            raise RuntimeError("direct MCTS inference requires a local model and device")
        tensors = packets_to_tensors(packets, self.device, events)
        output = self.model(
            tensors.map_tokens,
            tensors.global_features,
            tensors.action_categorical,
            tensors.action_continuous,
            tensors.action_mask,
            tensors.event_features,
        )
        logits = output.policy_logits.clone()
        for row, packet in enumerate(packets):
            bias = strategic_action_bias(
                packet,
                capture_city_bonus=self.config.capture_city_prior_bonus,
                capture_capital_bonus=self.config.capture_capital_prior_bonus,
                found_city_bonus=self.config.found_city_prior_bonus,
                attack_bonus=self.config.attack_prior_bonus,
                end_turn_penalty=self.config.end_turn_prior_penalty,
            )
            logits[row, :bias.size] += torch.as_tensor(
                bias, dtype=logits.dtype, device=logits.device
            )
        probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        values = output.value.detach().cpu().numpy()
        result: List[Tuple[Dict[int, float], float]] = []
        for row, packet in enumerate(packets):
            action_ids = np.asarray(packet["actions"]["action_id"], dtype=np.int64)
            result.append((
                {
                    int(action_id): float(prior)
                    for action_id, prior in zip(action_ids, probabilities[row, :action_ids.size])
                },
                float(values[row]),
            ))
        return result

    @torch.no_grad()
    def _particles(self, packet: Dict[str, object], events: Dict[str, object] | None = None) -> List[np.ndarray]:
        remote_sample = getattr(self.inference, "sample_particles", None)
        if callable(remote_sample):
            return remote_sample(
                packet,
                events,
                self.config.particles,
                max(self.config.temperature, 0.35),
            )

        def sample() -> List[torch.Tensor]:
            if self.model is None or self.device is None:
                raise RuntimeError("local particle sampling requires a model and device")
            tensors = packet_to_tensors(packet, self.device, events)
            return self.model.sample_completed_maps(
                tensors.map_tokens,
                tensors.global_features,
                particles=self.config.particles,
                temperature=max(self.config.temperature, 0.35),
                event_features=tensors.event_features,
            )

        particles = self.inference.exclusive(sample) if self.inference is not None else sample()
        return [particle.detach().cpu().numpy().astype(np.int32) for particle in particles]

    def decide(
        self,
        env: object,
        on_progress: Optional[Callable[[int, int, int], None]] = None,
    ) -> MCTSDecision:
        """Search one real-game decision without exposing its hidden state."""
        root_packet = env.model_request_numpy()
        action_ids = np.asarray(root_packet["actions"]["action_id"], dtype=np.int64)
        if action_ids.size == 0:
            raise RuntimeError("PolyEnv returned no legal actions")
        root_events = visible_events_or_empty(env)
        root_priors, _ = self._evaluate_packet(root_packet, root_events)
        # Rollouts may prune/adjust a node.  Keep the policy evaluated on the
        # authoritative observation immutable for final targets and fallback.
        root_player = int(root_packet["obs"].get("current_player", env.current_player()))
        root = SearchNode(priors=dict(root_priors), to_play=root_player)
        simulations_per_particle = max(1, self.config.simulations // self.config.particles)
        total_simulations = simulations_per_particle * self.config.particles
        completed_simulations = 0
        valid_particles = 0

        inference_batch_size = max(1, int(self.config.inference_batch_size))
        for completed in self._particles(root_packet, root_events):
            try:
                # The only branch creation from the authoritative env.
                belief_env = env.make_belief_env(completed, perspective=root_player)
            except Exception as error:
                # Report each distinct validation failure once.  This is a
                # terminal diagnostic only, not a saved game/log artifact.
                reason = str(error) or type(error).__name__
                if reason not in self._reported_particle_errors:
                    print(f"belief particle rejected: {reason}", flush=True)
                    self._reported_particle_errors.add(reason)
                # Discard it; never fall back to env.clone(), which would leak.
                completed_simulations += simulations_per_particle
                if on_progress:
                    on_progress(completed_simulations, total_simulations, valid_particles)
                continue
            valid_particles += 1
            remaining = simulations_per_particle
            while remaining:
                selected = min(remaining, inference_batch_size)
                pending: List[PendingLeaf] = []
                for _ in range(selected):
                    leaf = self._select_leaf(belief_env, root, root_player)
                    if leaf is not None:
                        pending.append(leaf)
                if pending:
                    evaluations = self._evaluate_packets([leaf.packet for leaf in pending])
                    for leaf, (priors, local_value) in zip(pending, evaluations):
                        leaf.node.priors = priors
                        root_value = local_value if leaf.acting_player == root_player else -local_value
                        self._backup(leaf.path, root_value, reserved=True)
                completed_simulations += selected
                remaining -= selected
                if on_progress:
                    on_progress(completed_simulations, total_simulations, valid_particles)

        visits = np.asarray(
            [root.children.get(int(action), SearchNode()).visits for action in action_ids],
            dtype=np.float64,
        )
        priors = np.asarray([root_priors.get(int(action), 0.0) for action in action_ids], dtype=np.float64)
        if visits.sum() == 0:
            # Safe cold-start fallback: choose from policy only, never source truth.
            row = int(priors.argmax())
        elif self.config.temperature <= 1e-5:
            row = int(visits.argmax())
        else:
            weights = np.power(visits, 1.0 / self.config.temperature)
            row = int(np.random.choice(len(action_ids), p=weights / weights.sum()))
        return MCTSDecision(int(action_ids[row]), row, visits, priors, valid_particles)

    def _select_leaf(
        self,
        belief_env: object,
        root: SearchNode,
        root_player: int,
    ) -> Optional[PendingLeaf]:
        """Select one leaf on CPU; its inference is batched later on GPU."""
        env = belief_env.clone()
        node = root
        path = [node]
        for _ in range(self.config.max_depth):
            if env.is_done():
                winner = int(env.model_request_numpy()["obs"].get("winner", -1))
                self._backup(path, 1.0 if winner == root_player else -1.0)
                return None
            packet = env.model_request_numpy()
            acting_player = int(packet["obs"].get("current_player", env.current_player()))
            if node.to_play is None:
                node.to_play = acting_player
            if not node.priors:
                self._reserve(path)
                return PendingLeaf(node, path, packet, acting_player)
            action_id = node.select(self.config.c_puct, root_player)
            ok, done, reward, winner, _ = env.step_fast(action_id)
            if not ok:
                # A legal action in the real observation can be illegal in a
                # particular hypothesis.  Penalise that branch in this search
                # but never remove it from shared priors or another particle.
                child = node.children.setdefault(action_id, SearchNode())
                path.append(child)
                self._backup(path, -1.0)
                return None
            child = node.children.setdefault(action_id, SearchNode())
            node = child
            path.append(node)
            if done:
                self._backup(path, 1.0 if int(winner) == root_player else -1.0)
                return None
        self._backup(path, 0.0)

        return None

    @staticmethod
    def _reserve(path: List[SearchNode]) -> None:
        for visited in path:
            visited.virtual_visits += 1

    @staticmethod
    def _backup(path: List[SearchNode], value: float, reserved: bool = False) -> None:
        for visited in path:
            if reserved:
                visited.virtual_visits -= 1
                if visited.virtual_visits < 0:
                    raise RuntimeError("MCTS virtual visit underflow")
            visited.visits += 1
            visited.value_sum += value
