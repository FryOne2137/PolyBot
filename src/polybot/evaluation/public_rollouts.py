"""Batched raw policy rollouts on independently seeded live games."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..config import TrainingConfig
from ..data.full_actions import exact_vector_env, merge_action_batches
from ..data.polyenv_features import vector_event_action_history
from ..infrastructure.device import select_device
from ..models.policy_value_belief import PolicyValueBeliefNet
from ..training.routed_self_play_pool import routed_public_policy_value


_OBSERVATION_FIELDS = (
    "map_tokens", "state", "own_techs", "own_tribe", "initial_player_count", "player_scores", "player_scores_mask",
    "action_id", "action_features", "action_mask", "total_legal_action_count", "action_truncated",
    "visible_event_features", "visible_event_mask", "terminated", "action_valid", "state_id",
)


class PublicRolloutPool:
    """Step live vector games and keep their observations compatible with search replay."""

    def __init__(
        self,
        training: TrainingConfig,
        models: Sequence[PolicyValueBeliefNet],
        *,
        num_envs: int,
        visible_action_history: int,
    ) -> None:
        from PolyEnv import Bardur, Imperius, VectorGameEnv

        self.training = training
        self.device = select_device(training.device)
        self.models = models
        self.num_envs = num_envs
        # Single-slot environments allow explicit noncontiguous seeds and
        # duplicate seeds for the two branches. Neural inference stays batched.
        # SelfPlayPool does not expose combat previews in its action packets.
        self._envs = [exact_vector_env(
            VectorGameEnv, num_envs=1, num_threads=1, seed=training.seed,
            map_size=training.map_size, map_type=training.map_type, players=(Bardur, Imperius),
            auto_reset=False, include_combat_preview=False, visible_event_history=visible_action_history,
        ) for _ in range(num_envs)]
        self._observations: list[dict[str, np.ndarray]] = [{} for _ in self._envs]

    def _public_packet(self) -> dict[str, np.ndarray]:
        packet = merge_action_batches([
            {key: observed[key] for key in _OBSERVATION_FIELDS} for observed in self._observations
        ])
        history, history_mask = vector_event_action_history(packet)
        del packet["visible_event_features"], packet["visible_event_mask"]
        packet["visible_action_history"] = history
        packet["visible_action_history_mask"] = history_mask.astype(np.uint8)
        # Keeping only the compact history reproduces SelfPlayPool's neutral
        # event summary and the same public inputs used to choose the branches.
        packet["map_seed"] = self._seeds
        packet["to_play"] = packet["state"][:, 1]
        packet["agent_id"] = self._assignments[np.arange(self.num_envs), packet["to_play"]]
        packet["winner"] = packet["state"][:, 3]
        return packet

    def reset(self, seeds: np.ndarray, assignments: np.ndarray) -> dict[str, np.ndarray]:
        self._seeds = np.asarray(seeds, dtype=np.uint32)
        self._assignments = np.asarray(assignments, dtype=np.int32)
        for env, seed, observed in zip(self._envs, self._seeds, self._observations, strict=True):
            env.reset_into(observed, seed=int(seed))
        return self._public_packet()

    def step_public_policy(
        self, actions: np.ndarray, state_ids: np.ndarray, request: dict[str, np.ndarray],
    ) -> None:
        for row in np.flatnonzero(actions >= 0):
            self._envs[row].step_into_checked(actions[row:row + 1], state_ids[row:row + 1], self._observations[row])
        request.update(self._public_packet())

    def direct_policy_value(
        self, request: Mapping[str, np.ndarray], rows: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return routed_public_policy_value(
            request, rows, models=self.models, device=self.device,
            training=self.training, num_envs=self.num_envs,
        )
