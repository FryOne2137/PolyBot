"""Joint map encoder, legal-action policy, value, and belief heads."""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from typing import List, Sequence, Tuple

import torch
from torch import nn

from ..config import ModelConfig
from ..data.player_context import CONTEXT_SIZE, LEGACY_GLOBAL_SIZE
from .tile_encoder import TileEncoder
from .action_encoder import ActionEncoder
from .history_encoder import HistoryEncoder
from .relational_policy import SpatialStateReadout
from ..data.token_schema import (
    EVENT_SUMMARY_SIZE,
    MAP_FEATURE_MAX,
    VISIBLE_ACTION_HISTORY_FEATURES,
)


ACTION_FEATURE_MAX: Sequence[int] = (9, 1023, 1023, 1023, 1023, 25, 16, 90, 8, 14, 5, 1)

# Runtime IDs are lookup keys, not stable semantic categories. The shared
# action encoder resolves source/target/city pointers through tile states.
FACTOR_OPTION_FEATURES: Sequence[int] = (5, 6, 7, 8, 9, 10, 11)
RUNTIME_MAP_FEATURES = frozenset((6, 15, 22))

# Legal actions are public observations.  These fields describe their semantic
# kind without treating short-lived runtime unit/city IDs as stable categories.
BELIEF_ACTION_FEATURES: Sequence[int] = (0, 3, 4, 5, 6, 7, 8, 9, 11)

# Fields which can be sampled independently.  Cities and units are sampled by
# the constrained completion code below: their related columns have to be
# mutually consistent for ``BeliefWorldBuilder`` to accept the hypothesis.
SAFE_PARTICLE_FEATURES: Sequence[int] = (7, 8, 18, 19, 20)
CITY_PARTICLE_FEATURES: Sequence[int] = (9, 10, 11, 12, 13, 16, 17)
UNIT_PARTICLE_FEATURES: Sequence[int] = (2, 3, 4, 5, 21)

# ``unit_type`` is a sparse enum, not the dense 1..127 range allocated for
# forward-compatible token embeddings.  BeliefWorldBuilder validates the wide
# range, but UnitFactory can only instantiate these concrete template ids.
VALID_BELIEF_UNIT_TYPES = frozenset(
    (*range(1, 14), *range(20, 27), *range(30, 42), *range(50, 54),
     *range(60, 65), *range(70, 82), 90)
)

BRIDGE_ACTION_TYPE = 8
BUILD_BRIDGE_TILE_ACTION = 9
TERRAIN_FEATURE = 19
WATER_TERRAIN = 1
LAND_TERRAIN = 2
LANDISH_TERRAINS = frozenset((2, 3, 4))


def _bridge_neighbour_pairs(tile: int, width: int, tiles: int) -> tuple[tuple[int, int], ...]:
    """Return complete N/S and E/W pairs around a row-major board tile."""
    row, column = divmod(int(tile), int(width))
    result: list[tuple[int, int]] = []
    if row > 0 and row + 1 < width:
        result.append((tile - width, tile + width))
    if column > 0 and column + 1 < width:
        result.append((tile - 1, tile + 1))
    return tuple(result)


def apply_public_bridge_constraints(
    completed: torch.Tensor,
    observed: torch.Tensor,
    action_categorical: torch.Tensor | None,
    action_mask: torch.Tensor | None,
    *,
    suppress_unlisted: bool = False,
) -> None:
    """Project one detached belief onto public Build Bridge constraints.

    A legal bridge action proves that its water tile has a landish N/S or E/W
    pair.  This routine changes hidden terrain only; visible tokens are never
    touched.  Conservative fallback may additionally break land pairs around
    visible water tiles whose bridge action is absent, preventing an all-land
    fallback from inventing public actions.
    """
    if completed.ndim != 2 or observed.shape != completed.shape:
        raise ValueError("bridge projection expects matching [tiles, features] tensors")
    tiles = int(completed.shape[0])
    width = isqrt(tiles)
    if width * width != tiles:
        return
    known = observed[:, 0] == 1
    legal_targets: set[int] = set()
    if action_categorical is not None:
        actions = action_categorical.detach().to(device="cpu", dtype=torch.long)
        if actions.ndim != 2 or actions.shape[1] < 10:
            raise ValueError("action_categorical must have shape [actions, >=10]")
        if action_mask is None:
            valid = torch.ones(actions.shape[0], dtype=torch.bool)
        else:
            valid = action_mask.detach().to(device="cpu", dtype=torch.bool).reshape(-1)
            if valid.numel() != actions.shape[0]:
                raise ValueError("action_mask must match action_categorical")
        bridge_rows = valid & (actions[:, 0] == BRIDGE_ACTION_TYPE) & (
            actions[:, 9] == BUILD_BRIDGE_TILE_ACTION
        )
        for target in actions[bridge_rows, 2].tolist():
            if 0 <= int(target) < tiles:
                legal_targets.add(int(target))

    def is_landish(index: int) -> bool:
        return int(completed[index, TERRAIN_FEATURE]) in LANDISH_TERRAINS

    if suppress_unlisted:
        for target in range(tiles):
            if target in legal_targets:
                continue
            # Only a visible water target can expose a bridge action.  Other
            # prerequisites remain in the submitted public state.
            if (
                not bool(known[target])
                or int(observed[target, TERRAIN_FEATURE]) != WATER_TERRAIN
                or int(observed[target, 7]) != 0
                or int(observed[target, 4]) >= 1
            ):
                continue
            for first, second in _bridge_neighbour_pairs(target, width, tiles):
                if not (is_landish(first) and is_landish(second)):
                    continue
                hidden_member = next(
                    (index for index in (first, second) if not bool(known[index])),
                    None,
                )
                if hidden_member is not None:
                    completed[hidden_member, TERRAIN_FEATURE] = WATER_TERRAIN

    # Apply positive constraints last.  If neighbouring water targets interact,
    # an actual public legal action always takes precedence over suppression.
    for target in sorted(legal_targets):
        candidates: list[tuple[int, tuple[int, int]]] = []
        for pair in _bridge_neighbour_pairs(target, width, tiles):
            if any(
                bool(known[index])
                and int(observed[index, TERRAIN_FEATURE]) not in LANDISH_TERRAINS
                for index in pair
            ):
                continue
            score = sum(is_landish(index) for index in pair)
            candidates.append((score, pair))
        if not candidates:
            # A real legal action cannot contradict two visible axes.  Leave
            # the particle unchanged so PolyEnv diagnostics can reject it.
            continue
        _, selected = max(candidates, key=lambda item: item[0])
        for index in selected:
            if not bool(known[index]):
                completed[index, TERRAIN_FEATURE] = LAND_TERRAIN


def _indices(values: torch.Tensor, largest_value: int) -> torch.Tensor:
    """Map -1 to zero and clamp malformed model/environment values safely."""
    return (values + 1).clamp_(0, largest_value + 1)


@dataclass
class ModelOutput:
    policy_logits: torch.Tensor
    value: torch.Tensor
    belief_logits: List[torch.Tensor]
    state: torch.Tensor
    tile_states: torch.Tensor


class PolicyValueBeliefNet(nn.Module):
    """Shared Transformer with one score per *currently legal* action.

    The belief decoder predicts token distributions.  It deliberately does not
    receive full-map labels at inference time; those are only a training target.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.observation_context_version != 1:
            raise ValueError("unsupported observation_context_version")
        if config.tile_encoder_version != 2:
            raise ValueError("unsupported tile_encoder_version; initialize a new model with the joint tile encoder")
        if config.action_encoder_version != 2:
            raise ValueError("unsupported action_encoder_version")
        if config.history_encoder_version != 2:
            raise ValueError("unsupported history_encoder_version; initialize a new model with semantic history")
        if min(config.action_encoder_dim, config.action_encoder_heads, config.action_latent_count) < 1 or config.action_encoder_dim % config.action_encoder_heads:
            raise ValueError("action encoder width must be positive and divisible by its heads; latents must be positive")
        d_model = config.d_model
        if config.policy_architecture not in {"factorized", "attention", "relational"}:
            raise ValueError("policy_architecture must be 'factorized', 'attention' or 'relational'")
        if config.action_interaction_dim < 1:
            raise ValueError("action_interaction_dim must be positive")
        self.policy_architecture = config.policy_architecture
        self.tile_encoder = TileEncoder(
            d_model,
            ignored_fields=RUNTIME_MAP_FEATURES if self.policy_architecture == "relational" else (),
            max_players=config.max_players,
        )
        self.x_embedding = nn.Embedding(config.max_board_side, d_model)
        self.y_embedding = nn.Embedding(config.max_board_side, d_model)
        self.global_encoder = nn.Sequential(
            nn.Linear(9 + EVENT_SUMMARY_SIZE, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.history_encoder = HistoryEncoder(d_model, config.max_board_side, config.max_players)
        self.history_position = nn.Embedding(1024, d_model)
        history_block = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=config.attention_heads, dim_feedforward=2 * d_model,
            dropout=config.dropout, activation="gelu", batch_first=True, norm_first=False,
        )
        # MPS does not implement the nested-tensor mask probe used by the
        # default fast path; disabling it avoids a CPU fallback every step.
        self.history_transformer = nn.TransformerEncoder(
            history_block, num_layers=1, enable_nested_tensor=False
        )
        self.history_gate = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        nn.init.zeros_(self.history_gate[-1].weight)
        nn.init.zeros_(self.history_gate[-1].bias)
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
        self.map_transformer = nn.TransformerEncoder(
            block, num_layers=config.transformer_layers, enable_nested_tensor=False
        )
        self.map_norm = nn.LayerNorm(d_model)

        self.action_encoder = ActionEncoder(
            d_model, config.action_encoder_dim, config.action_encoder_heads,
            config.action_latent_count, ACTION_FEATURE_MAX, FACTOR_OPTION_FEATURES, config.dropout,
        )
        self.shared_action_policy = nn.Sequential(
            nn.LayerNorm(config.action_encoder_dim), nn.Linear(config.action_encoder_dim, 2*config.action_encoder_dim),
            nn.GELU(), nn.Linear(2*config.action_encoder_dim, 1),
        )
        self.belief_shared_actions = nn.Linear(config.action_encoder_dim, d_model)
        if self.policy_architecture == "relational":
            self.spatial_readout = SpatialStateReadout(d_model, config.action_interaction_dim)
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1), nn.Tanh())
        # The belief decoder receives an explicit, masked summary of the
        # current legal-action set. Source/target tile embeddings make this
        # information spatial, e.g. a legal movement/attack constrains what
        # can plausibly occupy neighbouring hidden tiles.
        self.belief_action_norm = nn.LayerNorm(d_model)
        # A token for an action can tell the decoder that *some* source and
        # target exist, but an explicit tile-local map lets it learn the
        # complementary signal too: an adjacent hidden tile with no matching
        # move target is evidence that it is blocked.  Both source and target
        # channels are public information derived solely from legal actions.
        self.belief_action_evidence = nn.Linear(
            2 * (ACTION_FEATURE_MAX[0] + 2), d_model
        )
        self.belief_action_to_tile = nn.MultiheadAttention(
            d_model, config.attention_heads, dropout=config.dropout, batch_first=True
        )
        self.belief_action_gate = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        # The decoder learns a private projection of shared action tokens.
        # The shared encoder stays frozen during belief-only calibration.
        nn.init.zeros_(self.belief_action_evidence.weight)
        nn.init.zeros_(self.belief_action_evidence.bias)
        # The shared map encoder is deliberately frozen during belief-only
        # calibration: changing it would silently move policy and value too.
        # Give the belief decoder its own spatial transformer instead.  This
        # lets it infer map-wide relations such as a missing movement target
        # being evidence of an unseen blocker, while policy/value retain their
        # exact representation.  A zero gate makes this architecture extension
        # behaviourally identical for existing checkpoints until it is trained.
        belief_block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.attention_heads,
            dim_feedforward=4 * d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.belief_spatial_refiner = nn.TransformerEncoder(
            belief_block,
            num_layers=max(1, int(config.belief_decoder_layers)),
            enable_nested_tensor=False,
        )
        self.belief_spatial_norm = nn.LayerNorm(d_model)
        self.belief_spatial_gate = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.belief_spatial_gate[-1].weight)
        nn.init.zeros_(self.belief_spatial_gate[-1].bias)
        # Entity fields are decoded conditionally: settlement type determines
        # the city attributes, while unit type determines HP/owner/origin
        # attributes. Zero-initialised context tables preserve old checkpoints
        # until the structured decoder learns useful relations.
        self.belief_settlement_context = nn.Embedding(MAP_FEATURE_MAX[14] + 2, d_model)
        self.belief_unit_context = nn.Embedding(MAP_FEATURE_MAX[4] + 2, d_model)
        nn.init.zeros_(self.belief_settlement_context.weight)
        nn.init.zeros_(self.belief_unit_context.weight)
        self.belief_heads = nn.ModuleList(
            [nn.Linear(d_model, largest + 2) for largest in MAP_FEATURE_MAX]
        )
        # Observer/public globals remain a shared addition to the global token.
        self.observation_context = nn.Sequential(
            nn.Linear(CONTEXT_SIZE, d_model), nn.GELU(), nn.Linear(d_model, d_model, bias=False),
        )
        nn.init.zeros_(self.observation_context[-1].weight)

    def encode_map(
        self,
        map_tokens: torch.Tensor,
        global_features: torch.Tensor,
        event_features: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
        action_history_mask: torch.Tensor | None = None,
        *,
        board_sizes: Sequence[int] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, tiles, _ = map_tokens.shape
        if map_tokens.shape[-1] not in (23, 24):
            raise ValueError("expected 23 legacy or 24 population map fields")
        if global_features.shape[-1] not in (LEGACY_GLOBAL_SIZE, LEGACY_GLOBAL_SIZE + CONTEXT_SIZE):
            raise ValueError("incompatible global observation schema")
        context = global_features[..., LEGACY_GLOBAL_SIZE:]
        global_features = global_features[..., :LEGACY_GLOBAL_SIZE]
        padding = None
        if board_sizes is not None:
            if len(board_sizes) != batch or any(s < 1 or s > self.config.max_board_side or s*s > tiles for s in board_sizes):
                raise ValueError("padded batch needs one valid board size per row")
            widths = torch.as_tensor(board_sizes, dtype=torch.long, device=map_tokens.device)
            padding = torch.arange(tiles, device=map_tokens.device)[None] >= widths[:, None].square()
            map_tokens = map_tokens.masked_fill(padding.unsqueeze(-1), -1)
        elif self.policy_architecture == "relational":
            side = isqrt(tiles)
            if side * side != tiles or side > self.config.max_board_side:
                raise ValueError("relational board must be square and fit max_board_side")
        perspective = torch.round(global_features[..., 1] * 16).long()
        encoded = self.tile_encoder(map_tokens, perspective)
        positions = torch.arange(tiles, device=map_tokens.device)
        width = min(self.config.max_board_side, max(1, int(round(tiles ** 0.5)))) if board_sizes is None else widths
        if board_sizes is None:
            x_positions = self.x_embedding(positions % width)[None]
            y_positions = self.y_embedding((positions // width).clamp_max(self.config.max_board_side - 1))[None]
        else:
            x_positions = self.x_embedding(positions[None] % widths[:, None])
            y_positions = self.y_embedding((positions[None] // widths[:, None]).clamp_max(self.config.max_board_side - 1))
        encoded = encoded + x_positions + y_positions

        if event_features is None:
            event_features = torch.zeros((batch, EVENT_SUMMARY_SIZE), dtype=torch.float32, device=map_tokens.device)
        if self.policy_architecture == "relational":
            # The legacy global feature at index 6 is a scaled runtime city
            # pointer, not an amount. City-linked legal actions already name
            # the relevant public tile; do not feed the arbitrary number to
            # the global encoder (ID 0 also aliases "none" in this schema).
            global_features = global_features.clone()
            global_features[..., 6] = 0
        all_global_features = torch.cat([global_features, event_features], dim=-1)
        global_state = self.global_encoder(all_global_features)
        if context.shape[-1]:
            # No data is a zero contribution even after the new branch learns
            # biases; old replay must never mean an empty technology tree.
            present = (context != 0).any(dim=-1, keepdim=True)
            global_state = global_state + self.observation_context(context) * present
        global_state = global_state + self._history_context(
            action_history, action_history_mask, batch, map_tokens.device, board_side=width,
        )
        global_token = global_state.unsqueeze(1)
        transformer_input = torch.cat([global_token, encoded], dim=1)
        if padding is None:
            transformed = self.map_transformer(transformer_input)
        else:
            key_padding = torch.cat([padding.new_zeros((batch, 1)), padding], dim=1)
            transformed = self.map_transformer(transformer_input, src_key_padding_mask=key_padding)
        transformed = self.map_norm(transformed)
        state, tile_states = transformed[:, 0], transformed[:, 1:]
        if self.policy_architecture == "relational":
            state = self.spatial_readout(state, tile_states, map_tokens[..., 0] == 1)
        return state, tile_states

    def _history_context(
        self, history: torch.Tensor | None, history_mask: torch.Tensor | None,
        batch: int, device: torch.device, *, board_side: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if history is None or history.numel() == 0:
            return torch.zeros((batch, self.config.d_model), dtype=torch.float32, device=device)
        if history.ndim != 3 or history.shape[0] != batch or history.shape[2] != VISIBLE_ACTION_HISTORY_FEATURES:
            raise ValueError("visible_action_history must have shape [batch, length, 12]")
        if history_mask is None or history_mask.shape != history.shape[:2]:
            raise ValueError("visible_action_history_mask must have shape [batch, length]")
        valid_mask = history_mask.bool()
        # Positional embeddings describe distance *from the latest observed
        # action*, rather than the column chosen by the largest replay item
        # in this batch.  Thus an old 16-event sequence has exactly the same
        # representation alone and when right-aligned beside a 32-event row.
        # Padded positions are masked before attention, but are clamped to a
        # legal id so they never index the embedding out of range.
        valid_counts = valid_mask.long().sum(dim=1, keepdim=True)
        chronological_index = valid_mask.long().cumsum(dim=1) - 1
        positions = (valid_counts - 1 - chronological_index).clamp_min(0)
        positions = positions.clamp_max(self.history_position.num_embeddings - 1)
        encoded = self.history_encoder(
            history, self.config.max_board_side if board_side is None else board_side,
        ) + self.history_position(positions)
        no_history = ~valid_mask.any(dim=1)
        if bool(no_history.all()):
            return torch.zeros(
                (batch, self.config.d_model), dtype=torch.float32, device=device
            )
        safe_mask = valid_mask.clone()
        if no_history.any():
            safe_mask[no_history, 0] = True
        encoded = self.history_transformer(encoded, src_key_padding_mask=~safe_mask)
        mask = valid_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        context = self.history_gate(pooled)
        # A row without history must be exactly equivalent to omitting the
        # history tensor, regardless of other rows' padded history lengths.
        return torch.where(
            no_history.unsqueeze(-1), torch.zeros_like(context), context
        )

    def forward(
        self,
        map_tokens: torch.Tensor,
        global_features: torch.Tensor,
        action_categorical: torch.Tensor,
        action_continuous: torch.Tensor,
        action_mask: torch.Tensor,
        event_features: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
        action_history_mask: torch.Tensor | None = None,
        *,
        include_belief: bool = True,
        board_sizes: Sequence[int] | None = None,
    ) -> ModelOutput:
        if board_sizes is not None and include_belief:
            raise ValueError("padded map batches support policy/value inference only")
        state, tile_states = self.encode_map(
            map_tokens, global_features, event_features, action_history, action_history_mask,
            board_sizes=board_sizes,
        )
        encoded_actions, value_state = self.action_encoder(
            state, tile_states, map_tokens, action_categorical, action_continuous, action_mask,
            board_sizes=board_sizes,
        )
        # Policy has one scoring path. Raw semantics and endpoints are encoded
        # once by action_encoder and shared with value and belief.
        logits = self.shared_action_policy(encoded_actions).squeeze(-1)
        logits = logits.masked_fill(~action_mask, float("-inf"))
        # Simulated MCTS leaves need only policy/value.  Belief decoding is
        # reserved for real observations that seed particle worlds.
        belief_states = (
            self._belief_tile_states(tile_states, action_categorical, action_mask, encoded_actions)
            if include_belief else None
        )
        belief_logits = self._structured_belief_logits(belief_states) if belief_states is not None else []
        return ModelOutput(
            policy_logits=logits,
            value=self.value_head(value_state).squeeze(-1),
            belief_logits=belief_logits,
            state=value_state,
            tile_states=tile_states,
        )

    def _belief_tile_states(
        self,
        tile_states: torch.Tensor,
        action_categorical: torch.Tensor,
        action_mask: torch.Tensor,
        encoded_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Condition hidden-tile logits on the public legal action set."""
        action_states = self.belief_shared_actions(encoded_actions)
        no_actions = ~action_mask.any(dim=1)
        if action_mask.shape[1] == 0:
            attended = torch.zeros_like(tile_states)
        else:
            safe_mask = action_mask.clone()
            if no_actions.any():
                safe_mask[no_actions, 0] = True
            attended, _ = self.belief_action_to_tile(
                tile_states, action_states, action_states,
                key_padding_mask=~safe_mask, need_weights=False,
            )
        if no_actions.any():
            attended = attended.masked_fill(no_actions[:, None, None], 0.0)
        evidence = self._belief_action_tile_evidence(
            action_categorical, action_mask, tile_states.shape[1]
        )
        action_conditioned = (
            tile_states
            + self.belief_action_evidence(evidence)
            + self.belief_action_gate(self.belief_action_norm(attended))
        )
        refined = self.belief_spatial_refiner(action_conditioned)
        return action_conditioned + self.belief_spatial_gate(
            self.belief_spatial_norm(refined)
        )

    @staticmethod
    def _belief_action_tile_evidence(
        action_categorical: torch.Tensor,
        action_mask: torch.Tensor,
        tiles: int,
    ) -> torch.Tensor:
        """Encode public action source/target incidence at each map tile.

        The representation deliberately says only that an action is public
        and legal.  It never distinguishes why an alternative action is
        absent, so the belief head must infer whether a hidden blocker is a
        unit, terrain, or another game-rule constraint from supervised maps.
        """
        if action_categorical.ndim != 3 or action_mask.shape != action_categorical.shape[:2]:
            raise ValueError("belief action evidence expects [batch, actions, features] and matching mask")
        batch, actions, _ = action_categorical.shape
        type_count = ACTION_FEATURE_MAX[0] + 2
        channels = 2 * type_count
        evidence = torch.zeros(
            (batch, tiles * channels),
            dtype=torch.float32,
            device=action_categorical.device,
        )
        action_types = _indices(action_categorical[..., 0], ACTION_FEATURE_MAX[0])
        for role, column in enumerate((1, 2)):
            indexes = action_categorical[..., column]
            valid = action_mask & (indexes >= 0) & (indexes < tiles)
            if not bool(valid.any()):
                continue
            combined = (
                indexes.clamp(0, max(0, tiles - 1)) * channels
                + action_types.clamp(0, type_count - 1)
                + role * type_count
            )
            evidence.scatter_add_(1, combined, valid.to(evidence.dtype))
        return evidence.view(batch, tiles, channels).clamp_max_(1.0)

    def _structured_belief_logits(self, tile_states: torch.Tensor) -> List[torch.Tensor]:
        """Decode cities and units in a valid dependency order.

        City-only fields are conditioned on predicted settlement type; unit
        attributes are conditioned on predicted unit type. Runtime IDs remain
        deliberately unconditioned because they are not stable categories.
        """
        logits = [head(tile_states) for head in self.belief_heads]
        settlement_logits = logits[14]
        settlement_probs = torch.softmax(settlement_logits, dim=-1)
        settlement_state = tile_states + settlement_probs @ self.belief_settlement_context.weight
        for field in (9, 10, 11, 12, 13, 16, 17):
            logits[field] = self.belief_heads[field](settlement_state)
        unit_logits = logits[4]
        unit_probs = torch.softmax(unit_logits, dim=-1)
        unit_state = tile_states + unit_probs @ self.belief_unit_context.weight
        for field in (2, 3, 5, 21, 22):
            logits[field] = self.belief_heads[field](unit_state)
        return logits

    @torch.no_grad()
    def sample_completed_maps(
        self,
        map_tokens: torch.Tensor,
        global_features: torch.Tensor,
        particles: int,
        temperature: float = 1.0,
        event_features: torch.Tensor | None = None,
        action_categorical: torch.Tensor | None = None,
        action_continuous: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
        action_history_mask: torch.Tensor | None = None,
        perspective_player: int | None = None,
        known_city_count: int | None = None,
    ) -> List[torch.Tensor]:
        """Sample valid, detached map hypotheses while copying known rows.

        Hidden cities and units are inferred too.  Their runtime references are
        generated by a constrained decoder rather than copied from the engine.
        """
        if map_tokens.shape[0] != 1:
            raise ValueError("MCTS particle generation currently expects a batch of one")
        state, tile_states = self.encode_map(
            map_tokens, global_features, event_features, action_history, action_history_mask
        )
        if action_categorical is not None and action_continuous is not None and action_mask is not None:
            encoded_actions, _ = self.action_encoder(
                state, tile_states, map_tokens, action_categorical, action_continuous, action_mask,
            )
            tile_states = self._belief_tile_states(
                tile_states, action_categorical, action_mask, encoded_actions
            )
        belief_logits = self._structured_belief_logits(tile_states)
        return self.sample_completed_maps_from_belief_logits(
            map_tokens, belief_logits, particles, temperature,
            perspective_player=perspective_player,
            known_city_count=known_city_count,
            action_categorical=action_categorical,
            action_mask=action_mask,
        )

    @torch.no_grad()
    def sample_completed_maps_from_belief_logits(
        self,
        map_tokens: torch.Tensor,
        belief_logits: Sequence[torch.Tensor],
        particles: int,
        temperature: float = 1.0,
        *,
        perspective_player: int | None = None,
        known_city_count: int | None = None,
        action_categorical: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
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
             0, -1, -1, -1, 0, 2, 0, -1, -1] + [-1] * (observed.shape[-1] - 23),
            dtype=observed.dtype,
            device=observed.device,
        )
        # Sample a valid world in dependency order. Terrain/static structure
        # comes first, then entity presence, then only the attributes owned by
        # a sampled city/unit. Sampling all heads independently can create a
        # hidden empty tile whose unobserved attributes nevertheless describe
        # a city or unit, which makes later structural repair unnecessarily
        # ambiguous.
        sampled_features: dict[int, torch.Tensor] = {}
        has_hidden = bool(hidden.any())

        def sample_feature(feature_index: int) -> torch.Tensor:
            if has_hidden:
                logits = belief_logits[feature_index][0, hidden]
                if feature_index == 3 and perspective_player is not None:
                    logits = self._hidden_unit_owner_logits(
                        logits, torch.as_tensor(perspective_player, device=logits.device)
                    )
                if temperature <= 0.0:
                    return (logits.argmax(dim=-1) - 1).expand(particles, -1)
                probabilities = torch.softmax(logits / max(temperature, 1e-4), dim=-1)
                return torch.multinomial(
                    probabilities, particles, replacement=True
                ).transpose(0, 1) - 1
            return torch.empty((particles, 0), dtype=observed.dtype, device=observed.device)

        for feature_index in SAFE_PARTICLE_FEATURES:
            sampled_features[feature_index] = sample_feature(feature_index)
        sampled_features[14] = sample_feature(14)
        sampled_features[4] = sample_feature(4)
        # Only cities cannot occupy water. Villages, ruins and lighthouses
        # are valid settlement hypotheses too; erasing every non-city value
        # here made their prediction invisible to MCTS.
        city_present = (sampled_features[14] == 2) & torch.isin(
            sampled_features[TERRAIN_FEATURE],
            torch.as_tensor(tuple(LANDISH_TERRAINS), device=observed.device),
        )
        sampled_features[14] = torch.where(
            (sampled_features[14] != 2) | city_present,
            sampled_features[14],
            torch.zeros_like(sampled_features[14]),
        )
        unit_present = sampled_features[4] >= 1
        for feature_index in CITY_PARTICLE_FEATURES:
            samples = sample_feature(feature_index)
            sampled_features[feature_index] = torch.where(
                city_present, samples, torch.full_like(samples, -1)
            )
        for feature_index in UNIT_PARTICLE_FEATURES:
            if feature_index == 4:
                continue
            samples = sample_feature(feature_index)
            sampled_features[feature_index] = torch.where(
                unit_present, samples, torch.full_like(samples, -1)
            )

        result: List[torch.Tensor] = []
        observed_cpu = observed.to(device="cpu")
        action_rows = None if action_categorical is None else action_categorical.detach().to(device="cpu")
        action_masks = None if action_mask is None else action_mask.detach().to(device="cpu")
        for particle_index in range(particles):
            completed = empty_tile.expand_as(observed).clone()
            for feature_index in SAFE_PARTICLE_FEATURES:
                completed[hidden, feature_index] = sampled_features[feature_index][particle_index].clamp_min(0)
            completed[known] = observed[known]
            completed[:, 0] = observed[:, 0]
            completed_cpu = completed.to(device="cpu")
            hidden_cpu = hidden.to(device="cpu")
            apply_public_bridge_constraints(
                completed_cpu,
                observed_cpu,
                None if action_rows is None else action_rows[0],
                None if action_masks is None else action_masks[0],
            )
            # ``sampled_features`` contains only the compact hidden rows for
            # efficient multinomial sampling.  The structural repair below
            # works in absolute tile coordinates, so expand every field back
            # to a full board before handing it over.  Without this, scalar
            # BeliefMCTS evaluation crashes as soon as a revealed tile makes
            # ``hidden`` shorter than the board.
            full_sampled: dict[int, torch.Tensor] = {}
            for field, values in sampled_features.items():
                full_values = torch.full(
                    (observed.shape[0],),
                    -1,
                    dtype=observed.dtype,
                    device="cpu",
                )
                full_values[hidden_cpu] = values[particle_index].to(device="cpu", dtype=observed.dtype)
                full_sampled[field] = full_values
            self._place_structured_entities(
                completed_cpu, hidden_cpu, full_sampled,
                player_count=2,
                perspective_player=perspective_player,
            )
            self._add_missing_player_capitals(
                completed_cpu, hidden_cpu, player_count=2, perspective_player=perspective_player
            )
            self._validate_perspective_city_count(
                completed_cpu, perspective_player=perspective_player, known_city_count=known_city_count
            )
            result.append(completed_cpu)
        return result

    @torch.no_grad()
    def sample_completed_map_batch_from_belief_logits(
        self,
        map_tokens: torch.Tensor,
        belief_logits: Sequence[torch.Tensor],
        temperature: float = 1.0,
        *,
        perspective_players: torch.Tensor | None = None,
        known_city_counts: torch.Tensor | None = None,
        action_categorical: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Create one valid detached belief completion for every batch row.

        ``SelfPlayPool`` consumes exactly one completion per live environment.
        This is the vectorised counterpart to the scalar particle sampler above:
        it never reads source-world state, preserves every visible token, and
        creates structurally valid hidden city/unit hypotheses.
        """
        if len(belief_logits) != len(MAP_FEATURE_MAX):
            raise ValueError("belief completion requires all belief heads")
        observed = map_tokens
        if observed.ndim != 3:
            raise ValueError("map_tokens must have shape [batch, tiles, features]")
        known = observed[..., 0] == 1
        empty_tile = torch.tensor(
            [0, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1,
             0, -1, -1, -1, 0, 2, 0, -1, -1] + [-1] * (observed.shape[-1] - 23),
            dtype=observed.dtype,
            device=observed.device,
        )
        completed = empty_tile.view(1, 1, -1).expand_as(observed).clone()
        batch, tiles, _ = observed.shape
        hidden = ~known
        sampled_features: dict[int, torch.Tensor] = {}

        def sample_feature(feature_index: int) -> torch.Tensor:
            logits = belief_logits[feature_index]
            if feature_index == 3 and perspective_players is not None:
                logits = self._hidden_unit_owner_logits(
                    logits, perspective_players.to(device=logits.device).reshape(batch, 1)
                )
            if temperature <= 0.0:
                return logits.argmax(dim=-1) - 1
            scaled = logits / max(float(temperature), 1e-4)
            probabilities = torch.softmax(scaled, dim=-1).reshape(-1, scaled.shape[-1])
            return torch.multinomial(probabilities, 1).reshape(batch, tiles) - 1

        for feature_index in SAFE_PARTICLE_FEATURES:
            samples = sample_feature(feature_index)
            sampled_features[feature_index] = samples
            if feature_index in SAFE_PARTICLE_FEATURES:
                completed[..., feature_index] = torch.where(
                    hidden, samples.to(completed.dtype).clamp_min(0), completed[..., feature_index]
                )
        sampled_features[14] = sample_feature(14)
        sampled_features[4] = sample_feature(4)
        # See the scalar sampler above: condition only city presence on
        # terrain before sampling city attributes. Other settlement types are
        # public game objects in their own right and must reach the engine.
        city_present = (sampled_features[14] == 2) & torch.isin(
            sampled_features[TERRAIN_FEATURE],
            torch.as_tensor(tuple(LANDISH_TERRAINS), device=observed.device),
        )
        sampled_features[14] = torch.where(
            (sampled_features[14] != 2) | city_present,
            sampled_features[14],
            torch.zeros_like(sampled_features[14]),
        )
        unit_present = sampled_features[4] >= 1
        for feature_index in CITY_PARTICLE_FEATURES:
            samples = sample_feature(feature_index)
            sampled_features[feature_index] = torch.where(
                city_present, samples, torch.full_like(samples, -1)
            )
        for feature_index in UNIT_PARTICLE_FEATURES:
            if feature_index == 4:
                continue
            samples = sample_feature(feature_index)
            sampled_features[feature_index] = torch.where(
                unit_present, samples, torch.full_like(samples, -1)
            )
        completed[known] = observed[known]
        completed[..., 0] = observed[..., 0]
        # SelfPlayPool needs a CPU ``int32`` packet. Move once before the
        # small structured-city repair: doing ``tolist()`` for every GPU row
        # would otherwise synchronise CUDA once per live environment.
        completed_cpu = completed.to(device="cpu")
        observed_cpu = observed.to(device="cpu")
        hidden_cpu = hidden.to(device="cpu")
        perspectives = self._batch_constraint_values(perspective_players, batch, "perspective_players")
        city_counts = self._batch_constraint_values(known_city_counts, batch, "known_city_counts")
        action_rows = None if action_categorical is None else action_categorical.detach().to(device="cpu")
        action_masks = None if action_mask is None else action_mask.detach().to(device="cpu")
        for row in range(batch):
            apply_public_bridge_constraints(
                completed_cpu[row],
                observed_cpu[row],
                None if action_rows is None else action_rows[row],
                None if action_masks is None else action_masks[row],
            )
            self._place_structured_entities(
                completed_cpu[row], hidden_cpu[row],
                {field: values[row].to(device="cpu") for field, values in sampled_features.items()},
                player_count=2,
                perspective_player=perspectives[row],
            )
            self._add_missing_player_capitals(
                completed_cpu[row], hidden_cpu[row], player_count=2, perspective_player=perspectives[row]
            )
            self._validate_perspective_city_count(
                completed_cpu[row], perspective_player=perspectives[row], known_city_count=city_counts[row]
            )
        return completed_cpu

    @staticmethod
    def _hidden_unit_owner_logits(
        logits: torch.Tensor, perspective: torch.Tensor
    ) -> torch.Tensor:
        """Own units are always visible in the supported two-player samplers."""
        owners = torch.arange(logits.shape[-1], device=logits.device) - 1
        known_perspective = (perspective >= 0) & (perspective < 2)
        allowed = (owners >= 0) & (owners < 2) & (owners != perspective[..., None])
        return logits.masked_fill(known_perspective[..., None] & ~allowed, -torch.inf)

    @staticmethod
    def _batch_constraint_values(
        values: torch.Tensor | None,
        batch: int,
        name: str,
    ) -> list[int | None]:
        """Move public per-perspective constraints into the CPU repair path."""
        if values is None:
            return [None] * batch
        flat = values.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        if flat.numel() != batch:
            raise ValueError(f"{name} must provide one value per belief row")
        return [int(value) for value in flat.tolist()]

    @staticmethod
    def _place_structured_entities(
        completed: torch.Tensor,
        hidden: torch.Tensor,
        sampled: dict[int, torch.Tensor],
        player_count: int,
        perspective_player: int | None = None,
    ) -> None:
        """Materialise sampled cities and units without invalid runtime IDs.

        The network predicts *what* occupies a hidden tile.  This method owns
        ephemeral IDs and enforces the engine invariants: every city gets a
        unique centre/id, units have legal owner/HP, and an unknown unit has no
        fictitious origin-city reference.  It intentionally does not invent
        territory/origin pointers because those are relational facts the tile
        decoder cannot identify reliably yet.
        """
        PolicyValueBeliefNet._add_referenced_city_placeholders(
            completed, hidden,
            player_count=player_count,
            perspective_player=perspective_player,
            sampled_settlements=sampled[14],
        )
        open_settlement_tiles = hidden & (completed[:, 14] == 0)
        city_candidates = torch.nonzero(
            open_settlement_tiles & (sampled[14] == 2), as_tuple=False
        ).flatten().tolist()
        # A predicted capital must be considered before ordinary city rows;
        # otherwise the city cap can discard the only capital hypothesis.
        city_candidates.sort(
            key=lambda tile: int(sampled[9][tile]) == 1,
            reverse=True,
        )
        references = torch.cat((completed[:, 6], completed[:, 22]))
        # Settlement IDs are map-wide, not city-centre-only.  A visible
        # village already owns the ID of the city that ``found_city`` will
        # create.  Reusing it for a hidden hypothesis makes the belief engine
        # see that village as an existing opposing city and removes a public,
        # legal Found City action.
        used_ids = [int(value) for value in completed[:, 15].tolist() if int(value) >= 0]
        used_ids.extend(int(value) for value in references.tolist() if int(value) >= 0)
        next_city_id = max(used_ids, default=-1) + 1

        # Settlement values 1/3/4 are villages, ruins and lighthouses. They
        # used to be silently discarded because only cities were materialised.
        # A village reserves a future city ID; the other object types have no
        # city runtime identity and therefore retain the valid empty defaults
        # for city fields.
        non_city_settlement_candidates = torch.nonzero(
            open_settlement_tiles & (sampled[14] > 0) & (sampled[14] != 2),
            as_tuple=False,
        ).flatten().tolist()
        for tile in non_city_settlement_candidates:
            settlement_type = int(sampled[14][tile])
            if settlement_type < 1 or settlement_type > MAP_FEATURE_MAX[14]:
                continue
            completed[tile, 14] = settlement_type
            if settlement_type == 1:  # SettlementTypeEnum::Village
                completed[tile, 15] = next_city_id
                next_city_id += 1

        # A fresh, untrained network assigns roughly equal mass to every
        # category.  A cap prevents it from fabricating a whole board of cities
        # while retaining particle diversity once the decoder has learned them.
        city_limit = max(player_count, completed.shape[0] // 20)
        placed_cities = 0
        capital_owners = set(
            int(owner)
            for owner in completed[
                (completed[:, 14] == 2) & (completed[:, 9] == 1), 16
            ].tolist()
            if 0 <= int(owner) < player_count
        )
        for tile in city_candidates:
            if placed_cities >= city_limit:
                break
            owner = int(sampled[16][tile]) % max(player_count, 1)
            # A player always knows every city they own.  A city owned by the
            # perspective on a hidden row is therefore not uncertainty: it is
            # an impossible hypothesis that would corrupt tech prices, spawns
            # and territory actions.
            if perspective_player is not None and owner == perspective_player:
                continue
            is_capital = int(sampled[9][tile]) == 1 and owner not in capital_owners
            completed[tile, 9] = int(is_capital)
            completed[tile, 10] = min(1, max(0, int(sampled[10][tile])))
            completed[tile, 11] = min(1, max(0, int(sampled[11][tile])))
            completed[tile, 12] = max(0, int(sampled[12][tile]))
            completed[tile, 13] = max(1, int(sampled[13][tile]))
            completed[tile, 14] = 2
            completed[tile, 15] = next_city_id
            completed[tile, 16] = owner
            completed[tile, 17] = max(0, int(sampled[17][tile]))
            if is_capital:
                capital_owners.add(owner)
            next_city_id += 1
            placed_cities += 1

        unit_candidates = torch.nonzero(hidden & (sampled[4] >= 1), as_tuple=False).flatten().tolist()
        # As with cities, a cap keeps early random checkpoints syntactically
        # useful instead of filling every hidden tile with a unit.
        unit_limit = max(4, completed.shape[0] // 8)
        placed_units = 0
        for tile in unit_candidates:
            if placed_units >= unit_limit:
                break
            unit_type = int(sampled[4][tile])
            if unit_type not in VALID_BELIEF_UNIT_TYPES:
                # Sparse enum holes have no UnitFactory template. They must
                # mean "no hypothesis here", never an arbitrary remapping.
                continue
            health = max(1, int(sampled[2][tile]))
            max_health = max(health, int(sampled[21][tile]), 1)
            completed[tile, 2] = health
            completed[tile, 3] = int(sampled[3][tile]) % max(player_count, 1)
            completed[tile, 4] = unit_type
            completed[tile, 5] = max(-1, int(sampled[5][tile]))
            completed[tile, 21] = max_health
            completed[tile, 22] = -1
            placed_units += 1

    @torch.no_grad()
    def conservative_completed_map_batch(
        self,
        map_tokens: torch.Tensor,
        player_count: int = 2,
        *,
        perspective_players: torch.Tensor | None = None,
        known_city_counts: torch.Tensor | None = None,
        action_categorical: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        suppress_unlisted_bridges: bool = False,
    ) -> torch.Tensor:
        """Return a legal low-information fallback for rejected belief rows.

        This is never an engine-state fallback: hidden tiles are empty
        hypotheses, while visible tokens are copied verbatim.  It only repairs
        structural city references required for the engine to instantiate a
        detached belief world.
        """
        if map_tokens.ndim != 3:
            raise ValueError("map_tokens must have shape [batch, tiles, features]")
        observed = map_tokens
        known = observed[..., 0] == 1
        empty_tile = torch.tensor(
            [0, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1,
             0, -1, -1, -1, 0, 2, 0, -1, -1] + [-1] * (observed.shape[-1] - 23),
            dtype=observed.dtype,
            device=observed.device,
        )
        completed = empty_tile.view(1, 1, -1).expand_as(observed).clone()
        completed[known] = observed[known]
        completed[..., 0] = observed[..., 0]
        completed_cpu = completed.to(device="cpu")
        observed_cpu = observed.to(device="cpu")
        hidden_cpu = (~known).to(device="cpu")
        perspectives = self._batch_constraint_values(
            perspective_players, observed.shape[0], "perspective_players"
        )
        city_counts = self._batch_constraint_values(
            known_city_counts, observed.shape[0], "known_city_counts"
        )
        action_rows = None if action_categorical is None else action_categorical.detach().to(device="cpu")
        action_masks = None if action_mask is None else action_mask.detach().to(device="cpu")
        for row in range(observed.shape[0]):
            apply_public_bridge_constraints(
                completed_cpu[row],
                observed_cpu[row],
                None if action_rows is None else action_rows[row],
                None if action_masks is None else action_masks[row],
                suppress_unlisted=suppress_unlisted_bridges,
            )
            self._add_referenced_city_placeholders(
                completed_cpu[row], hidden_cpu[row],
                player_count=player_count, perspective_player=perspectives[row],
            )
            self._add_missing_player_capitals(
                completed_cpu[row], hidden_cpu[row],
                player_count=player_count, perspective_player=perspectives[row],
            )
            self._validate_perspective_city_count(
                completed_cpu[row], perspective_player=perspectives[row], known_city_count=city_counts[row]
            )
        return completed_cpu

    @staticmethod
    def _add_referenced_city_placeholders(
        completed: torch.Tensor,
        hidden: torch.Tensor,
        *,
        player_count: int = 2,
        perspective_player: int | None = None,
        sampled_settlements: torch.Tensor | None = None,
    ) -> None:
        """Create valid hidden city centers for IDs visible only by reference.

        A revealed territory or own unit can name a city whose center remains
        hidden. Reserve a hidden tile before placing sampled settlements. If
        the prediction offers no land, the public reference justifies turning
        one hidden tile into land for its center.
        """
        referenced = torch.cat((completed[:, 6], completed[:, 22]))
        referenced = torch.unique(referenced[referenced >= 0])
        if referenced.numel() == 0:
            return
        existing = set(completed[completed[:, 14] == 2, 15].tolist())
        missing_ids = [city_id for city_id in referenced.tolist() if city_id not in existing]
        if not missing_ids:
            return
        candidates = torch.nonzero(hidden & (completed[:, 14] == 0), as_tuple=False).flatten()
        terrain = completed[candidates, TERRAIN_FEATURE]
        land = torch.isin(terrain, completed.new_tensor(tuple(LANDISH_TERRAINS)))
        priority = land.to(torch.int64) * 2
        if sampled_settlements is not None:
            priority += sampled_settlements[candidates] == 0
        ordered = torch.argsort(priority, descending=True, stable=True)
        for city_id, tile in zip(missing_ids, candidates[ordered].tolist()):
            if int(completed[tile, TERRAIN_FEATURE]) not in LANDISH_TERRAINS:
                completed[tile, TERRAIN_FEATURE] = LAND_TERRAIN
            completed[tile, 9] = 0   # not capital
            completed[tile, 10] = 0  # no workshop
            completed[tile, 11] = 0  # no wall
            completed[tile, 12] = 0  # no parks
            completed[tile, 13] = 1  # city level
            completed[tile, 14] = 2  # SettlementTypeEnum::City
            completed[tile, 15] = city_id
            # The owner is genuinely unknown, but it cannot be the
            # perspective: their own city centre would already be known.
            owner = next(
                (candidate for candidate in range(player_count) if candidate != perspective_player), 0
            )
            completed[tile, 16] = owner
            completed[tile, 17] = 0

    @staticmethod
    def _add_missing_player_capitals(
        completed: torch.Tensor,
        hidden: torch.Tensor,
        player_count: int,
        perspective_player: int | None = None,
    ) -> None:
        """Keep every public player alive in a detached fog hypothesis.

        At an early turn the opposing capital is normally hidden.  A completed
        map with no city for that public player is syntactically valid, but the
        engine correctly considers that player defeated and makes the detached
        MCTS root terminal.  Allocate one conservative *hidden* capital for
        every owner without a city.  Its location is a model hypothesis; the
        number of players and their identities are public game metadata.
        """
        if player_count < 1:
            return
        city_rows = completed[:, 14] == 2
        capital_owners = set(
            int(owner)
            for owner in completed[
                city_rows & (completed[:, 9] == 1), 16
            ].tolist()
            if 0 <= int(owner) < player_count
        )
        # A village's settlement ID is reserved for its future city too.
        # Never allocate a hypothetical capital with that ID.
        existing_ids = completed[:, 15]
        next_city_id = max((int(city_id) for city_id in existing_ids.tolist() if city_id >= 0), default=-1) + 1
        # A fallback capital may occupy only an otherwise empty land
        # hypothesis; it must not erase a sampled object or invent a city on
        # water merely to keep a detached particle non-terminal.
        candidates = iter(torch.nonzero(
            hidden
            & (completed[:, 14] == 0)
            & torch.isin(
                completed[:, TERRAIN_FEATURE],
                torch.as_tensor(
                    tuple(LANDISH_TERRAINS),
                    dtype=completed.dtype,
                    device=completed.device,
                ),
            ),
            as_tuple=False,
        ).flatten().tolist())
        for owner in range(player_count):
            if owner in capital_owners:
                continue
            # Missing own cities are never a fog hypothesis. Their absence is
            # public information and must remain absent in the completion.
            if perspective_player is not None and owner == perspective_player:
                continue
            # A non-capital city can already exist for this owner. Promote a
            # deterministic one rather than inventing a duplicate city; this
            # is only the engine-valid fallback when the decoder gave no
            # capital prediction.
            # Revealed tokens are immutable public evidence. A missing
            # capital may only promote a hidden city hypothesis; never turn a
            # visible ordinary city into a different map token.
            existing = torch.nonzero(
                hidden & city_rows & (completed[:, 16] == owner), as_tuple=False
            ).flatten()
            if existing.numel():
                completed[int(existing[0]), 9] = 1
                capital_owners.add(owner)
                continue
            try:
                tile = next(candidates)
            except StopIteration:
                return
            completed[tile, 9] = 1   # capital
            completed[tile, 10] = 1  # workshop, valid at level one
            completed[tile, 11] = 0
            completed[tile, 12] = 0
            completed[tile, 13] = 1
            completed[tile, 14] = 2  # SettlementTypeEnum::City
            completed[tile, 15] = next_city_id
            completed[tile, 16] = owner
            completed[tile, 17] = 0
            next_city_id += 1

    @staticmethod
    def _validate_perspective_city_count(
        completed: torch.Tensor,
        *,
        perspective_player: int | None,
        known_city_count: int | None,
    ) -> None:
        """Assert the public invariant that a player knows all own cities."""
        if perspective_player is None or known_city_count is None:
            return
        city_rows = completed[:, 14] == 2
        actual_count = int((city_rows & (completed[:, 16] == perspective_player)).sum().item())
        if actual_count != known_city_count:
            raise ValueError(
                "belief completion violates the public own-city invariant: "
                f"perspective={perspective_player}, expected_cities={known_city_count}, "
                f"completed_cities={actual_count}"
            )
