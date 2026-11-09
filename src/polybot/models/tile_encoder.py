"""Joint per-tile encoding of categorical state and signed city population."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from ..data.player_context import UNKNOWN_POPULATION
from ..data.token_schema import MAP_FEATURE_MAX


class TileEncoder(nn.Module):
    """Turn one complete 24-field tile into one shared-width representation.

    Population is normalized and concatenated with the categorical embedding
    sum *before* the shared nonlinear fusion. It has no separate learned branch.
    City level is already categorical column 13; no duplicate next-level input
    is needed. Runtime identifiers can remain lookup keys in relational models.
    """

    def __init__(self, width: int, *, ignored_fields: Iterable[int] = (), max_players: int = 16):
        super().__init__()
        self.width = width
        self.max_players = max_players
        self.ignored_fields = frozenset(ignored_fields)
        self.embeddings = nn.ModuleList([
            nn.Identity() if index in self.ignored_fields else nn.Embedding(largest + 2, width)
            for index, largest in enumerate(MAP_FEATURE_MAX)
        ])
        # One extra value and its availability mask augment the semantic
        # embedding. All features, including city level, share this same MLP.
        self.fusion = nn.Sequential(
            nn.Linear(width + 2, width), nn.GELU(),
            nn.Linear(width, width), nn.LayerNorm(width),
        )

    def forward(self, tokens: torch.Tensor, perspective_players: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] not in (23, 24):
            raise ValueError("tile encoder expects [batch,tiles,24] or legacy 23-field tokens")
        if perspective_players.shape != tokens.shape[:1]:
            raise ValueError("tile encoder needs one observer identity per batch row")
        categorical = torch.zeros((*tokens.shape[:2], self.width),
                                  dtype=torch.float32, device=tokens.device)
        for index, (embedding, largest) in enumerate(zip(self.embeddings, MAP_FEATURE_MAX)):
            if index not in self.ignored_fields:
                categorical = categorical + embedding((tokens[..., index] + 1).clamp(0, largest + 1))

        population = torch.zeros(tokens.shape[:2], dtype=torch.float32, device=tokens.device)
        known = torch.zeros_like(population, dtype=torch.bool)
        if tokens.shape[-1] == 24:
            valid_observer = (perspective_players >= 0) & (perspective_players < self.max_players)
            known = (
                valid_observer[:, None]
                & tokens[..., 0].eq(1) & tokens[..., 14].eq(2)
                & tokens[..., 16].eq(perspective_players[:, None])
                & tokens[..., 23].ne(UNKNOWN_POPULATION)
            )
            # A real -1 population is valid. Only the explicit legacy sentinel
            # denotes unknown; hidden/foreign values are masked before use.
            population = torch.asinh(torch.where(known, tokens[..., 23], 0).float() / 8.0)
        joint = torch.cat([categorical, population.unsqueeze(-1), known.unsqueeze(-1)], dim=-1)
        return self.fusion(joint)
