"""Policy-guided PUCT using only detached PolyEnv belief worlds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..config import MCTSConfig
from ..data.polyenv_features import packet_to_tensors, strategic_action_bias, visible_events_or_empty
from ..models.policy_value_belief import PolicyValueBeliefNet


@dataclass
class SearchNode:
    priors: Dict[int, float] = field(default_factory=dict)
    children: Dict[int, "SearchNode"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    to_play: Optional[int] = None

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def select(self, c_puct: float, root_player: int) -> int:
        parent_scale = float(np.sqrt(max(1, self.visits)))
        best_action, best_score = None, -float("inf")
        for action_id, prior in self.priors.items():
            child = self.children.get(action_id)
            child_visits = child.visits if child else 0
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


class BeliefMCTS:
    """MCTS that never calls clone() on the authoritative environment.

    ``clone()`` is used only after ``make_belief_env()`` has already produced a
    detached particle world.  Such clones contain predicted state, never the
    source environment's hidden state.
    """

    def __init__(self, model: PolicyValueBeliefNet, device: torch.device, config: MCTSConfig) -> None:
        self.model = model
        self.device = device
        self.config = config
        self._reported_particle_errors: set[str] = set()

    @torch.no_grad()
    def _evaluate_packet(
        self,
        packet: Dict[str, object],
        events: Dict[str, object] | None = None,
    ) -> Tuple[Dict[int, float], float]:
        tensors = packet_to_tensors(packet, self.device, events)
        output = self.model(
            tensors.map_tokens,
            tensors.global_features,
            tensors.action_categorical,
            tensors.action_continuous,
            tensors.action_mask,
            tensors.event_features,
        )
        logits = output.policy_logits[0].clone()
        bias = strategic_action_bias(
            packet,
            capture_city_bonus=self.config.capture_city_prior_bonus,
            capture_capital_bonus=self.config.capture_capital_prior_bonus,
            found_city_bonus=self.config.found_city_prior_bonus,
            attack_bonus=self.config.attack_prior_bonus,
            end_turn_penalty=self.config.end_turn_prior_penalty,
        )
        logits = logits + torch.as_tensor(bias, dtype=logits.dtype, device=logits.device)
        priors = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        action_ids = np.asarray(packet["actions"]["action_id"], dtype=np.int64)
        action_priors = {
            int(action_id): float(prior)
            for action_id, prior in zip(action_ids, priors)
        }
        return action_priors, float(output.value.item())

    @torch.no_grad()
    def _particles(self, packet: Dict[str, object], events: Dict[str, object] | None = None) -> List[np.ndarray]:
        tensors = packet_to_tensors(packet, self.device, events)
        particles = self.model.sample_completed_maps(
            tensors.map_tokens,
            tensors.global_features,
            particles=self.config.particles,
            temperature=max(self.config.temperature, 0.35),
            event_features=tensors.event_features,
        )
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
            for _ in range(simulations_per_particle):
                self._simulate(belief_env, root, root_player)
                completed_simulations += 1
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

    def _simulate(self, belief_env: object, root: SearchNode, root_player: int) -> None:
        env = belief_env.clone()
        node = root
        path = [node]
        value: Optional[float] = None
        for _ in range(self.config.max_depth):
            if env.is_done():
                winner = int(env.model_request_numpy()["obs"].get("winner", -1))
                value = 1.0 if winner == root_player else -1.0
                break
            packet = env.model_request_numpy()
            acting_player = int(packet["obs"].get("current_player", env.current_player()))
            if node.to_play is None:
                node.to_play = acting_player
            if not node.priors:
                priors, local_value = self._evaluate_packet(packet)
                node.priors = priors
                value = local_value if acting_player == root_player else -local_value
                break
            action_id = node.select(self.config.c_puct, root_player)
            ok, done, reward, winner, _ = env.step_fast(action_id)
            if not ok:
                # A legal action in the real observation can be illegal in a
                # particular hypothesis.  Penalise that branch in this search
                # but never remove it from shared priors or another particle.
                child = node.children.setdefault(action_id, SearchNode())
                path.append(child)
                value = -1.0
                break
            child = node.children.setdefault(action_id, SearchNode())
            node = child
            path.append(node)
            if done:
                value = 1.0 if int(winner) == root_player else -1.0
                break
        if value is None:
            value = 0.0
        for visited in path:
            visited.visits += 1
            visited.value_sum += value
