"""Central, serialisable configuration objects."""

from dataclasses import dataclass
from typing import Optional


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
    # ``factorized`` scores a legal action as type + source tile + target
    # tile + option fields.  It avoids materialising attention from every
    # legal action to every map tile, which is both faster and less prone to
    # memorising runtime ids.  ``attention`` remains for old checkpoints.
    policy_architecture: str = "factorized"


@dataclass
class MCTSConfig:
    """PUCT search budget for a single decision."""

    simulations: int = 128
    particles: int = 8
    c_puct: float = 1.5
    max_depth: int = 96
    temperature: float = 1.0
    # Number of selected MCTS leaves evaluated in one neural-network forward
    # pass.  It changes throughput, not the information exposed to search.
    inference_batch_size: int = 32
    # A global self-play decision creates a short-lived MctsPool. ``None``
    # resolves to one C++ thread there, avoiding OS resource exhaustion from
    # repeatedly creating a hardware-sized native worker pool.
    native_threads: Optional[int] = None
    native_max_actions: int = 1024
    # Applied only to already legal, visible capture actions in MCTS.
    capture_city_prior_bonus: float = 2.0
    capture_capital_prior_bonus: float = 4.0
    found_city_prior_bonus: float = 3.0
    attack_prior_bonus: float = 0.8
    end_turn_prior_penalty: float = 2.0
    # Scale the hand-written visible-action curriculum independently from the
    # network. Training starts with this aid and anneals it away; evaluation
    # sets it to zero so a match measures the checkpoint itself.
    strategic_prior_weight: float = 1.0
    # Blend a small amount of the legal-action teacher into early MCTS priors.
    # It is a training aid only; the neural network still scores every leaf.
    teacher_prior_weight: float = 0.20


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
    # Keep dense hints well below the exact terminal +/-1 return. Larger
    # values can make repeated captures outweigh actually winning the game.
    shaping_target_weight: float = 0.05
    reward_discount: float = 0.997
    # MCTS value targets interpolate between a search bootstrap at the next
    # decision and the later return.  1.0 is Monte Carlo; lower values put
    # more trust in search/value estimates.
    td_lambda: float = 1.0
    # Preserve a small amount of the productive legal-action teacher in MCTS
    # policy targets while early search is still shallow/noisy.
    teacher_policy_target_weight: float = 0.10
    # During early self-play, occasionally execute a strategic legal action
    # instead of the current MCTS choice to create non-draw training data.
    exploration_rate: float = 0.20
    # Fraction of each training batch reserved for rare strategic/terminal
    # examples, while the rest stays uniformly sampled for stability.
    priority_replay_fraction: float = 0.35
