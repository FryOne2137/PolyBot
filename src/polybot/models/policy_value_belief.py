"""Joint map encoder, legal-action policy, value, and belief heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import nn

from ..config import ModelConfig
from ..data.token_schema import (
    ACTION_CATEGORICAL_FIELDS,
    ACTION_CONTINUOUS_FIELDS,
    EVENT_SUMMARY_SIZE,
    MAP_FEATURE_MAX,
)


ACTION_FEATURE_MAX: Sequence[int] = (9, 1023, 1023, 1023, 1023, 25, 16, 90, 8, 14, 5, 1)

# These fields describe a hidden tile without introducing cross-tile entity
# identities.  A city/unit decoder needs a separate constrained head because
# city IDs and unit origins must refer to other generated tiles.
SAFE_PARTICLE_FEATURES: Sequence[int] = (7, 8, 14, 18, 19, 20)


def _indices(values: torch.Tensor, largest_value: int) -> torch.Tensor:
    """Map -1 to zero and clamp malformed model/environment values safely."""
    return (values + 1).clamp_(0, largest_value + 1)


@dataclass
class ModelOutput:
    policy_logits: torch.Tensor
    value: torch.Tensor
    belief_logits: List[torch.Tensor]
    tile_states: torch.Tensor


class PolicyValueBeliefNet(nn.Module):
    """Shared Transformer with one score per *currently legal* action.

    The belief decoder predicts token distributions.  It deliberately does not
    receive full-map labels at inference time; those are only a training target.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.d_model
        self.tile_embeddings = nn.ModuleList(
            [nn.Embedding(largest + 2, d_model) for largest in MAP_FEATURE_MAX]
        )
        self.x_embedding = nn.Embedding(config.max_board_side, d_model)
        self.y_embedding = nn.Embedding(config.max_board_side, d_model)
        self.global_encoder = nn.Sequential(
            nn.Linear(9 + EVENT_SUMMARY_SIZE, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.attention_heads,
            dim_feedforward=4 * d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            # Post-norm enables PyTorch's nested-tensor fast path and avoids
            # the harmless warning emitted by the previous pre-norm setting.
            norm_first=False,
        )
        self.map_transformer = nn.TransformerEncoder(block, num_layers=config.transformer_layers)
        self.map_norm = nn.LayerNorm(d_model)

        self.action_embeddings = nn.ModuleList(
            [nn.Embedding(largest + 2, d_model) for largest in ACTION_FEATURE_MAX]
        )
        self.action_continuous = nn.Sequential(
            nn.Linear(len(ACTION_CONTINUOUS_FIELDS), d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.action_to_map = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=config.attention_heads, dropout=config.dropout, batch_first=True
        )
        self.action_norm = nn.LayerNorm(d_model)
        self.policy_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1), nn.Tanh())
        self.belief_heads = nn.ModuleList(
            [nn.Linear(d_model, largest + 2) for largest in MAP_FEATURE_MAX]
        )

    def encode_map(
        self,
        map_tokens: torch.Tensor,
        global_features: torch.Tensor,
        event_features: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, tiles, _ = map_tokens.shape
        encoded = torch.zeros(
            (batch, tiles, self.config.d_model),
            dtype=torch.float32,
            device=map_tokens.device,
        )
        for feature_index, (embedding, largest) in enumerate(zip(self.tile_embeddings, MAP_FEATURE_MAX)):
            encoded = encoded + embedding(_indices(map_tokens[..., feature_index], largest))
        positions = torch.arange(tiles, device=map_tokens.device)
        width = min(self.config.max_board_side, max(1, int(round(tiles ** 0.5))))
        x_positions = self.x_embedding(positions % width)[None]
        y_positions = self.y_embedding(
            (positions // width).clamp_max(self.config.max_board_side - 1)
        )[None]
        encoded = encoded + x_positions + y_positions

        if event_features is None:
            event_features = torch.zeros((batch, EVENT_SUMMARY_SIZE), dtype=torch.float32, device=map_tokens.device)
        all_global_features = torch.cat([global_features, event_features], dim=-1)
        global_token = self.global_encoder(all_global_features).unsqueeze(1)
        transformed = self.map_transformer(torch.cat([global_token, encoded], dim=1))
        transformed = self.map_norm(transformed)
        return transformed[:, 0], transformed[:, 1:]

    def forward(
        self,
        map_tokens: torch.Tensor,
        global_features: torch.Tensor,
        action_categorical: torch.Tensor,
        action_continuous: torch.Tensor,
        action_mask: torch.Tensor,
        event_features: torch.Tensor | None = None,
        *,
        include_belief: bool = True,
    ) -> ModelOutput:
        state, tile_states = self.encode_map(map_tokens, global_features, event_features)
        batch, actions, _ = action_categorical.shape
        action_states = torch.zeros(
            (batch, actions, self.config.d_model),
            dtype=torch.float32,
            device=map_tokens.device,
        )
        for feature_index, (embedding, largest) in enumerate(zip(self.action_embeddings, ACTION_FEATURE_MAX)):
            action_states = action_states + embedding(_indices(action_categorical[..., feature_index], largest))
        action_states = action_states + self.action_continuous(action_continuous)

        # Source and target tiles are particularly informative for movement and combat.
        for action_column in (1, 2):
            indices = action_categorical[..., action_column]
            existing = indices >= 0
            tile_indices = indices.clamp_min(0).clamp_max(tile_states.shape[1] - 1)
            gather_shape = (-1, -1, tile_states.shape[-1])
            gathered = tile_states.gather(
                1,
                tile_indices.unsqueeze(-1).expand(gather_shape),
            )
            action_states = action_states + gathered * existing.unsqueeze(-1)

        # Attention weights are never consumed by the policy. Asking
        # MultiheadAttention not to materialise them enables PyTorch's
        # memory-efficient scaled-dot-product kernels and avoids allocating a
        # [batch, actions, tiles] tensor in every training forward pass.
        attended, _ = self.action_to_map(
            action_states, tile_states, tile_states, need_weights=False
        )
        action_states = self.action_norm(action_states + attended)
        logits = self.policy_head(action_states).squeeze(-1)
        logits = logits.masked_fill(~action_mask, float("-inf"))
        # Simulated MCTS leaves need only policy/value.  Belief decoding is
        # reserved for real observations that seed particle worlds.
        belief_logits = [head(tile_states) for head in self.belief_heads] if include_belief else []
        return ModelOutput(
            policy_logits=logits,
            value=self.value_head(state).squeeze(-1),
            belief_logits=belief_logits,
            tile_states=tile_states,
        )

    @torch.no_grad()
    def sample_completed_maps(
        self,
        map_tokens: torch.Tensor,
        global_features: torch.Tensor,
        particles: int,
        temperature: float = 1.0,
        event_features: torch.Tensor | None = None,
    ) -> List[torch.Tensor]:
        """Sample valid, detached map hypotheses while copying known rows.

        Hidden units and cities cannot be sampled independently: their IDs
        reference other tiles, so an independent categorical decoder produces
        invalid worlds almost every time.  Until a structured entity decoder is
        introduced, particles predict only terrain-like fields and represent
        hidden dynamic entities as absent.  This is conservative, valid and
        never reads hidden source state.
        """
        if map_tokens.shape[0] != 1:
            raise ValueError("MCTS particle generation currently expects a batch of one")
        # This compatibility method has no policy action to score, so avoid a
        # dummy action-attention pass entirely.
        _, tile_states = self.encode_map(map_tokens, global_features, event_features)
        belief_logits = [head(tile_states) for head in self.belief_heads]
        return self.sample_completed_maps_from_belief_logits(
            map_tokens, belief_logits, particles, temperature
        )

    @torch.no_grad()
    def sample_completed_maps_from_belief_logits(
        self,
        map_tokens: torch.Tensor,
        belief_logits: Sequence[torch.Tensor],
        particles: int,
        temperature: float = 1.0,
    ) -> List[torch.Tensor]:
        """Sample particles from belief logits computed during root inference."""
        if map_tokens.shape[0] != 1:
            raise ValueError("MCTS particle generation currently expects a batch of one")
        if len(belief_logits) != len(MAP_FEATURE_MAX):
            raise ValueError("particle sampling requires all belief heads")
        observed = map_tokens[0]
        known = observed[:, 0] == 1
        hidden = ~known
        # A valid empty hidden tile.  The sampled static values below replace
        # its road/building/settlement/resource/terrain/tribe fields.
        empty_tile = torch.tensor(
            [0, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1,
             0, -1, -1, -1, 0, 2, 0, -1, -1],
            dtype=observed.dtype,
            device=observed.device,
        )
        # One softmax/multinomial per field replaces one per field * particle.
        sampled_features: dict[int, torch.Tensor] = {}
        has_hidden = bool(hidden.any())
        for feature_index in SAFE_PARTICLE_FEATURES:
            if has_hidden:
                probabilities = torch.softmax(
                    belief_logits[feature_index][0, hidden] / max(temperature, 1e-4), dim=-1
                )
                sampled_features[feature_index] = (
                    torch.multinomial(probabilities, particles, replacement=True).transpose(0, 1) - 1
                )
            else:
                sampled_features[feature_index] = torch.empty(
                    (particles, 0), dtype=observed.dtype, device=observed.device
                )

        result: List[torch.Tensor] = []
        for particle_index in range(particles):
            completed = empty_tile.expand_as(observed).clone()
            for feature_index in SAFE_PARTICLE_FEATURES:
                completed[hidden, feature_index] = sampled_features[feature_index][particle_index].clamp_min(0)
            # Settlement type 2 is a city and requires globally consistent
            # city id/owner/level fields.  Keep such hidden predictions empty
            # until the structured city head exists.
            completed[hidden & (completed[:, 14] == 2), 14] = 0
            completed[known] = observed[known]
            completed[:, 0] = observed[:, 0]
            self._add_referenced_city_placeholders(completed, hidden)
            result.append(completed)
        return result

    @staticmethod
    def _add_referenced_city_placeholders(completed: torch.Tensor, hidden: torch.Tensor) -> None:
        """Create valid hidden city centers for IDs visible only by reference.

        A revealed territory or own unit can name a city whose center remains
        hidden.  The engine correctly requires that ID to have a city-center
        token, but the observation does not tell us where it is.  Allocate a
        hidden tile deterministically; this is a hypothesis, not source data.
        """
        referenced = torch.cat((completed[:, 6], completed[:, 22]))
        referenced = torch.unique(referenced[referenced >= 0])
        if referenced.numel() == 0:
            return
        existing = completed[completed[:, 14] == 2, 15]
        existing = set(int(city_id) for city_id in existing.tolist() if city_id >= 0)
        candidates = iter(torch.nonzero(hidden & (completed[:, 14] != 2), as_tuple=False).flatten().tolist())
        for city_id in referenced.tolist():
            if int(city_id) in existing:
                continue
            try:
                tile = next(candidates)
            except StopIteration:
                # Let the engine reject an impossible hypothesis rather than
                # alter a visible row to force it through validation.
                return
            completed[tile, 9] = 0   # not capital
            completed[tile, 10] = 0  # no workshop
            completed[tile, 11] = 0  # no wall
            completed[tile, 12] = 0  # no parks
            completed[tile, 13] = 1  # city level
            completed[tile, 14] = 2  # SettlementTypeEnum::City
            completed[tile, 15] = int(city_id)
            completed[tile, 16] = 0  # valid player id; owner is unobserved
            completed[tile, 17] = 0
            existing.add(int(city_id))
