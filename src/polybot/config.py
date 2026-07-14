"""Central, serialisable configuration objects."""

from dataclasses import dataclass
@dataclass
class ModelConfig:
    """Network size. The defaults fit an Apple Silicon laptop."""

    d_model: int = 192
    transformer_layers: int = 8
    attention_heads: int = 8
    dropout: float = 0.08
    max_board_side: int = 32
    max_players: int = 16
    max_runtime_id: int = 512


@dataclass
class MCTSConfig:
    """PUCT search budget for a single decision."""

    simulations: int = 128
    particles: int = 8
    c_puct: float = 1.5
    max_depth: int = 96
    temperature: float = 1.0
    # Applied only to already legal, visible capture actions in MCTS.
    capture_city_prior_bonus: float = 2.0
    capture_capital_prior_bonus: float = 4.0
    found_city_prior_bonus: float = 3.0
    attack_prior_bonus: float = 0.8
    end_turn_prior_penalty: float = 2.0


@dataclass
class TrainingConfig:
    """Self-play and optimisation settings."""

    seed: int = 2137
    device: str = "auto"  # auto, mps, cuda, cpu
    map_size: int = 11
    map_type: str = "lakes"
    batch_size: int = 16
    replay_capacity: int = 20_000
    # Safety cap for one self-play game. A game that reaches it is a draw.
    # Keeps early smoke runs bounded when an untrained policy loops.
    max_moves: int = 300
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    policy_weight: float = 1.0
    value_weight: float = 1.0
    belief_weight: float = 0.35

    # Small economy shaping plus explicit strategic capture bonuses. Terminal
    # win/loss always remains +/-1 and is the dominant value target.
    stars_weight: float = 0.004
    income_weight: float = 0.025
    cities_weight: float = 0.12
    city_capture_reward: float = 1.0
    capital_capture_reward: float = 4.0
    shaping_target_weight: float = 0.20
    reward_discount: float = 0.997
    # During early self-play, occasionally execute a strategic legal action
    # instead of the current MCTS choice to create non-draw training data.
    exploration_rate: float = 0.20
    # Fraction of each training batch reserved for rare strategic/terminal
    # examples, while the rest stays uniformly sampled for stability.
    priority_replay_fraction: float = 0.35
