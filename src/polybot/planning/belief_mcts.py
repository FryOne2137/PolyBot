"""Policy-guided PUCT using only detached PolyEnv belief worlds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..config import MCTSConfig
from ..data.polyenv_features import (
    native_batch_to_tensors,
    packet_to_tensors,
    packets_to_tensors,
    strategic_action_bias,
    visible_events_or_empty,
)
from ..models.policy_value_belief import PolicyValueBeliefNet
from .batched_inference import BatchedInference, NativeLeafInference


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
        native_leaf_inference: Optional[NativeLeafInference] = None,
        map_size: int = 11,
        map_type: str = "lakes",
    ) -> None:
        self.model = model
        self.device = device
        self.config = config
        self.inference = inference
        self.native_leaf_inference = native_leaf_inference
        self.map_size = map_size
        self.map_type = map_type
        self._reported_particle_errors: set[str] = set()

    @torch.inference_mode()
    def _evaluate_packet(
        self,
        packet: Dict[str, object],
        events: Dict[str, object] | None = None,
    ) -> Tuple[Dict[int, float], float]:
        return self._evaluate_packets([packet], [events])[0]

    @torch.inference_mode()
    def _evaluate_packets(
        self,
        packets: List[Dict[str, object]],
        events: List[Dict[str, object] | None] | None = None,
    ) -> List[Tuple[Dict[int, float], float]]:
        if self.inference is not None:
            return self.inference.evaluate(packets, events)
        return self._evaluate_packets_direct(packets, events)

    @torch.inference_mode()
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
            include_belief=False,
        )
        return self._decode_evaluations(packets, output.policy_logits, output.value)

    def _decode_evaluations(
        self,
        packets: List[Dict[str, object]],
        policy_logits: torch.Tensor,
        values_tensor: torch.Tensor,
    ) -> List[Tuple[Dict[int, float], float]]:
        """Apply visible-only prior shaping and move compact outputs to CPU."""
        logits = policy_logits.clone()
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
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        values = values_tensor.cpu().numpy()
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

    @torch.inference_mode()
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

    @torch.inference_mode()
    def _root_priors_and_particles(
        self,
        packet: Dict[str, object],
        events: Dict[str, object] | None,
    ) -> Tuple[Dict[int, float], List[np.ndarray]]:
        """Evaluate and sample the root with one shared Transformer pass.

        A root policy/value output already contains belief logits.  Reusing it
        removes a formerly separate policy-less root forward pass.
        """
        if self.inference is not None:
            priors, _ = self._evaluate_packet(packet, events)
            return priors, self._particles(packet, events)
        if self.model is None or self.device is None:
            raise RuntimeError("local MCTS inference requires a model and device")
        tensors = packet_to_tensors(packet, self.device, events)
        output = self.model(
            tensors.map_tokens,
            tensors.global_features,
            tensors.action_categorical,
            tensors.action_continuous,
            tensors.action_mask,
            tensors.event_features,
        )
        priors, _ = self._decode_evaluations([packet], output.policy_logits, output.value)[0]
        particles = self.model.sample_completed_maps_from_belief_logits(
            tensors.map_tokens,
            output.belief_logits,
            self.config.particles,
            max(self.config.temperature, 0.35),
        )
        return priors, [particle.cpu().numpy().astype(np.int32) for particle in particles]

    def decide(
        self,
        env: object,
        on_progress: Optional[Callable[[int, int, int], None]] = None,
    ) -> MCTSDecision:
        """Use native PolyEnv MctsPool when available, else the safe fallback."""
        try:
            from PolyEnv import MctsPool  # type: ignore[import-not-found]
        except ImportError:
            return self._decide_python(env, on_progress)
        return self._decide_native(env, MctsPool, on_progress)

    def _native_prior_bias(
        self,
        categorical: torch.Tensor,
        continuous: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorised visible-action curriculum bias for MctsPool logits.

        ``native_batch_to_tensors`` maps PolyEnv's 17 action columns to the
        model's 12 categorical + 4 continuous columns.  Keep these indices in
        that *adapted* schema: tile_action is categorical column 9 and
        damage_dealt is continuous column 2.
        """
        action_type = categorical[..., 0]
        tile_action = categorical[..., 9]
        damage = continuous[..., 2].clamp_min(0)
        bias = torch.zeros_like(action_type, dtype=torch.float32)
        bias = bias + ((action_type == 8) & (tile_action == 14)).float() * self.config.capture_city_prior_bonus
        bias = bias + ((action_type == 8) & (tile_action == 11)).float() * self.config.found_city_prior_bonus
        bias = bias + (action_type == 1).float() * (
            self.config.attack_prior_bonus + damage.clamp_max(10.0) * 0.12
        )
        bias = bias + ((action_type == 8) & ((tile_action == 10) | (tile_action == 12))).float() * 0.8
        bias = bias - (action_type == 3).float() * self.config.end_turn_prior_penalty
        return bias.masked_fill(~mask, 0.0)

    @torch.inference_mode()
    def _evaluate_native_leaves_direct(self, leaves: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray]:
        """One dense model forward for native MctsPool leaves."""
        if self.model is None or self.device is None:
            raise RuntimeError("native MCTS requires a local model and device")
        tensors = native_batch_to_tensors(leaves, self.device, self.map_size, self.map_type)
        output = self.model(
            tensors.map_tokens, tensors.global_features,
            tensors.action_categorical, tensors.action_continuous,
            tensors.action_mask, tensors.event_features,
            include_belief=False,
        )
        logits = output.policy_logits + self._native_prior_bias(
            tensors.action_categorical, tensors.action_continuous, tensors.action_mask
        ).to(output.policy_logits.dtype)
        return logits.float().cpu().numpy(), output.value.float().cpu().numpy()

    @torch.inference_mode()
    def _decide_native(
        self,
        env: object,
        mcts_pool_type: object,
        on_progress: Optional[Callable[[int, int, int], None]],
    ) -> MCTSDecision:
        """Run PUCT selection, branching and backup in PolyEnv's C++ pool."""
        if self.model is None or self.device is None:
            raise RuntimeError("native MCTS requires a local model and device")
        root_packet = env.model_request_numpy()
        action_ids = np.asarray(root_packet["actions"]["action_id"], dtype=np.int64)
        if action_ids.size == 0:
            raise RuntimeError("PolyEnv returned no legal actions")
        root_events = visible_events_or_empty(env)
        root_priors, completed_particles = self._root_priors_and_particles(root_packet, root_events)
        root_player = int(root_packet["obs"].get("current_player", env.current_player()))
        belief_envs: List[object] = []
        for completed in completed_particles:
            try:
                belief_envs.append(env.make_belief_env(completed, perspective=root_player))
            except Exception as error:
                reason = str(error) or type(error).__name__
                if reason not in self._reported_particle_errors:
                    print(f"belief particle rejected: {reason}", flush=True)
                    self._reported_particle_errors.add(reason)
        if not belief_envs:
            priors = np.asarray([root_priors.get(int(action), 0.0) for action in action_ids], dtype=np.float64)
            row = int(priors.argmax())
            return MCTSDecision(int(action_ids[row]), row, np.zeros_like(priors), priors, 0)

        # Passing detached belief worlds is the documented fog-safe MctsPool
        # entry point. The original GameEnv is never given to the pool.
        pool = mcts_pool_type(
            belief_envs,
            num_threads=self.config.native_threads,
            max_actions=self.config.native_max_actions,
            c_puct=self.config.c_puct,
        )
        evaluations = 0
        target = max(1, self.config.simulations)
        while evaluations < target:
            leaves = pool.select_leaves()
            leaf_ids = np.asarray(leaves["leaf_id"])
            if leaf_ids.size == 0:
                break
            if self.native_leaf_inference is not None:
                logits, values = self.native_leaf_inference.evaluate(leaves)
            else:
                logits, values = self._evaluate_native_leaves_direct(leaves)
            pool.expand_and_backup(
                leaf_ids,
                logits,
                values,
            )
            evaluations += int(leaf_ids.size)
            if on_progress:
                on_progress(min(evaluations, target), target, len(belief_envs))

        roots = pool.root_policy(temperature=1.0)
        root_actions = np.asarray(roots["action_id"], dtype=np.int64)
        root_mask = np.asarray(roots["action_mask"], dtype=bool)
        root_visits = np.asarray(roots["visit_count"], dtype=np.float64)
        visit_by_action: Dict[int, float] = {}
        for actions_row, mask_row, visits_row in zip(root_actions, root_mask, root_visits):
            for action, valid, visits in zip(actions_row, mask_row, visits_row):
                if valid:
                    visit_by_action[int(action)] = visit_by_action.get(int(action), 0.0) + float(visits)
        visits = np.asarray([visit_by_action.get(int(action), 0.0) for action in action_ids], dtype=np.float64)
        priors = np.asarray([root_priors.get(int(action), 0.0) for action in action_ids], dtype=np.float64)
        if visits.sum() == 0:
            row = int(priors.argmax())
        elif self.config.temperature <= 1e-5:
            row = int(visits.argmax())
        else:
            weights = np.power(visits, 1.0 / self.config.temperature)
            row = int(np.random.choice(len(action_ids), p=weights / weights.sum()))
        return MCTSDecision(int(action_ids[row]), row, visits, priors, len(belief_envs))

    def _decide_python(
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
        root_priors, completed_particles = self._root_priors_and_particles(root_packet, root_events)
        # Rollouts may prune/adjust a node.  Keep the policy evaluated on the
        # authoritative observation immutable for final targets and fallback.
        root_player = int(root_packet["obs"].get("current_player", env.current_player()))
        root = SearchNode(priors=dict(root_priors), to_play=root_player)
        simulations_per_particle = max(1, self.config.simulations // self.config.particles)
        total_simulations = simulations_per_particle * self.config.particles
        completed_simulations = 0
        valid_particles = 0

        inference_batch_size = max(1, int(self.config.inference_batch_size))
        belief_envs: List[object] = []
        for completed in completed_particles:
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
            belief_envs.append(belief_env)

        # Select from all particle worlds before invoking the network.  These
        # worlds are independent hypotheses but share the root tree, so their
        # leaves can safely be scored together.  This is the main way to make
        # particle MCTS feed a worthwhile CUDA/MPS batch without exposing any
        # hidden state or trying to clone the authoritative environment.
        remaining = [simulations_per_particle] * len(belief_envs)
        next_particle = 0
        while any(count > 0 for count in remaining):
            pending: List[PendingLeaf] = []
            selected = 0
            while selected < inference_batch_size and any(count > 0 for count in remaining):
                particle_index = next_particle % len(belief_envs)
                next_particle += 1
                if remaining[particle_index] <= 0:
                    continue
                leaf = self._select_leaf(belief_envs[particle_index], root, root_player)
                remaining[particle_index] -= 1
                selected += 1
                if leaf is not None:
                    pending.append(leaf)
            if pending:
                evaluations = self._evaluate_packets([leaf.packet for leaf in pending])
                for leaf, (priors, local_value) in zip(pending, evaluations):
                    leaf.node.priors = priors
                    root_value = local_value if leaf.acting_player == root_player else -local_value
                    self._backup(leaf.path, root_value, reserved=True)
            completed_simulations += selected
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
