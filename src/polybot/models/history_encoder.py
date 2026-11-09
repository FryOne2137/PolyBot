"""Semantic encoding of the public 12-field visible-event journal."""

from __future__ import annotations

import torch
from torch import nn

from ..data.token_schema import VISIBLE_ACTION_HISTORY_FEATURES


class HistoryEncoder(nn.Module):
    """Encode categories, board coordinates and observer-local ID presence.

    Category IDs are not ordinal quantities. Unknown/out-of-schema values use
    a separate embedding, rather than clipping high valid unit types together.
    Runtime observed-unit IDs remain presence bits, preserving ID-renaming
    invariance without leaking spawn order or assuming globally stable IDs.
    """

    def __init__(self, d_model: int, max_board_side: int, max_players: int):
        super().__init__()
        self.max_board_side = max_board_side
        width = min(32, d_model)
        self.fields = (
            ("event_type", 0, 16),
            ("actor_player", 1, max_players - 1),
            ("actor_tribe", 2, 12),
            ("visibility", 3, 15),
            ("source_unit_type", 6, 127),
            ("target_unit_type", 7, 127),
            ("detail_kind", 10, 4),
        )
        self.categories = nn.ModuleDict({
            name: nn.Embedding(largest + 2, width)
            for name, _, largest in self.fields
        })
        # Detail IDs have different meanings for tile actions, buildings,
        # spawned units and upgrades. Never share their category namespace.
        self.details = nn.ModuleList([nn.Embedding(129, width) for _ in range(4)])
        self.x = nn.Embedding(max_board_side + 1, width)
        self.y = nn.Embedding(max_board_side + 1, width)
        self.fusion = nn.Sequential(
            nn.Linear(12 * width + 2, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )

    @staticmethod
    def _indices(values: torch.Tensor, largest: int) -> torch.Tensor:
        known = (values >= 0) & (values <= largest)
        return torch.where(known, values + 1, 0)

    def forward(self, history: torch.Tensor, board_side: int | torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[-1] != VISIBLE_ACTION_HISTORY_FEATURES:
            raise ValueError("history encoder expects [batch, length, 12]")
        if isinstance(board_side, torch.Tensor):
            if board_side.shape != history.shape[:1]:
                raise ValueError("history needs one board side per batch row")
            board_side = board_side[:, None]
        elif not 1 <= board_side <= self.max_board_side:
            raise ValueError("history board side must fit max_board_side")
        # Replay can store these categorical values as float32. Integer
        # conversion here keeps scalar, vector and replay paths identical.
        history = history.long()
        parts = [
            self.categories[name](self._indices(history[..., column], largest))
            for name, column, largest in self.fields
        ]
        detail = torch.zeros_like(parts[0])
        detail_index = self._indices(history[..., 11], 127)
        for kind, embedding in enumerate(self.details, start=1):
            detail = detail + embedding(detail_index) * (history[..., 10] == kind).unsqueeze(-1)
        parts.append(detail)
        for column in (4, 5):
            tile = history[..., column]
            known = (tile >= 0) & (tile < board_side * board_side)
            # Zero is unknown; actual coordinate zero maps to embedding one.
            x = torch.where(known, tile.remainder(board_side) + 1, 0)
            y = torch.where(known, tile.div(board_side, rounding_mode="floor") + 1, 0)
            parts.extend((self.x(x), self.y(y)))
        parts.append((history[..., 8:10] >= 0).to(parts[0].dtype))
        return self.fusion(torch.cat(parts, dim=-1))
