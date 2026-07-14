"""Inspectable, small potential-based reward shaping for self-play."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..config import TrainingConfig
from ..data.polyenv_features import _scalar


@dataclass
class RewardBreakdown:
    terminal: float = 0.0
    stars: float = 0.0
    income: float = 0.0
    cities: float = 0.0
    capture_city: float = 0.0
    capture_capital: float = 0.0

    @property
    def non_terminal(self) -> float:
        return self.stars + self.income + self.cities + self.capture_city + self.capture_capital

    @property
    def total(self) -> float:
        return self.terminal + self.non_terminal

    def as_dict(self) -> Dict[str, float]:
        return {
            "reward_terminal": self.terminal,
            "reward_stars": self.stars,
            "reward_income": self.income,
            "reward_cities": self.cities,
            "reward_capture_city": self.capture_city,
            "reward_capture_capital": self.capture_capital,
            "reward_total": self.total,
        }


class RewardShaper:
    """Keeps reward components transparent instead of hiding them in the env.

    The shaping is deliberately weak.  It makes early training less sparse but
    is never a replacement for the terminal win/loss objective.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.previous: Dict[int, Dict[str, float]] = {}

    @staticmethod
    def _state(obs: Dict[str, Any]) -> Dict[str, float]:
        return {
            "stars": _scalar(obs.get("player_stars", obs.get("stars"))),
            "income": _scalar(obs.get("next_turn_star_income", obs.get("income"))),
            "cities": _scalar(obs.get("own_cities", obs.get("cities"))),
        }

    def observe(
        self,
        player_id: int,
        obs: Dict[str, Any],
        terminal: float = 0.0,
        captured_city: bool = False,
        captured_capital: bool = False,
    ) -> RewardBreakdown:
        current = self._state(obs)
        old = self.previous.get(player_id, current)
        self.previous[player_id] = current
        return RewardBreakdown(
            terminal=terminal,
            stars=(current["stars"] - old["stars"]) * self.config.stars_weight,
            income=(current["income"] - old["income"]) * self.config.income_weight,
            cities=(current["cities"] - old["cities"]) * self.config.cities_weight,
            capture_city=self.config.city_capture_reward if captured_city else 0.0,
            capture_capital=self.config.capital_capture_reward if captured_capital else 0.0,
        )
