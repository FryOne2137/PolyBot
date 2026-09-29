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
    # AlphaZero-style self-play exploration.  It is consumed explicitly by
    # training loops only; evaluators never mix noise into a root prior.
    root_dirichlet_epsilon: float = 0.25
    # Total, not per-action, Dirichlet concentration.  A fixed total remains
    # well behaved when PolyEnv exposes a variable number of legal actions.
    root_dirichlet_concentration: float = 10.0
    # Sample from MCTS visits in the opening, then choose the best visit.  A
    # value of zero preserves the old single-temperature behaviour.
    temperature_drop_move: int = 20
    late_temperature: float = 0.0
    # A global self-play decision creates a short-lived MctsPool. ``None``
    # resolves to one C++ thread there, avoiding OS resource exhaustion from
    # repeatedly creating a hardware-sized native worker pool.
    native_threads: Optional[int] = None
    native_max_actions: int = 1024
    # PolyBot-only candidate-filtering experiment. Zero keeps the old full
    # policy. A positive value submits top-K policy actions plus tactical
    # mandatory actions and a small random legal sample. PolyEnv still owns
    # and validates the full legal-action list.
    search_top_actions: int = 0
    search_random_actions: int = 0
    # When a weak policy is flat, top-K alone can discard a good move before
    # MCTS has any opportunity to inspect it.  A positive floor expands top-K
    # until the retained neural prior mass reaches this value.
    search_min_prior_mass: float = 0.0
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
    # Safety cap for one self-play game. A truncated training game keeps its
    # policy/belief labels but is masked from value loss; evaluation counts it
    # as a draw. This keeps early runs bounded without inventing value labels.
    max_moves: int = 300
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    policy_weight: float = 1.0
    value_weight: float = 1.0
    belief_weight: float = 0.35
    # Geometry-aware auxiliary belief objective.  In particular, a visible
    # Build Bridge action constrains the hidden cardinal neighbours of its
    # water tile; learning the relation directly is much more sample-efficient
    # than treating every terrain token as independent.
    bridge_constraint_weight: float = 1.0

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
    exploration_rate: float = 0.05
    # Fraction of each training batch reserved for rare strategic/terminal
    # examples, while the rest stays uniformly sampled for stability.
    priority_replay_fraction: float = 0.35
    # Positions containing a legal bridge or supervised hidden bridge geometry
    # are rare but disproportionately important for action-safe ISMCTS.
    bridge_replay_fraction: float = 0.25
    # Bound repeated sampling when belief failures leave only a small replay.
    # Six means that, on average, one newly collected example can contribute
    # to at most six optimiser-batch exposures in this training phase.
    max_replay_epochs: float = 6.0
