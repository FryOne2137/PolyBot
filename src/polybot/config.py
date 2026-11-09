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
    # All heads share one legal-action encoder and policy has one token scorer.
    # Legacy names remain as map-encoding ablations: "relational" removes
    # runtime-ID features and adds spatial readout; the other two names use
    # the base map/global readout. They never select a second action scorer.
    policy_architecture: str = "relational"
    # Spatial state-readout width (historical serialized field name).
    action_interaction_dim: int = 64
    # Private spatial capacity for the fog-of-war decoder.  It is deliberately
    # separate from ``map_transformer`` so belief calibration can learn how
    # hidden tiles relate to one another without perturbing policy/value.
    belief_decoder_layers: int = 2
    observation_context_version: int = 1
    # Version 2 fuses all tile semantics and population inside one encoder.
    # It starts a new experiment; older tile weights are not interchangeable.
    tile_encoder_version: int = 2
    # Version 2 removes every legacy parallel policy scorer.
    action_encoder_version: int = 2
    action_encoder_dim: int = 64
    action_encoder_heads: int = 4
    action_latent_count: int = 16
    # Semantic event categories and board coordinates replace clipped ID scalars.
    history_encoder_version: int = 2


@dataclass
class MCTSConfig:
    """PUCT search budget for a single decision."""

    simulations: int = 128
    particles: int = 8
    c_puct: float = 1.5
    max_depth: int = 96
    temperature: float = 1.0
    # This controls completed fog-map sampling, not root action sampling.
    # Keeping it separate prevents a high self-play exploration temperature
    # from turning belief particles into implausibly random worlds.
    belief_particle_temperature: float = 0.15
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


@dataclass(frozen=True)
class FlatCurriculumConfig:
    """Opt-in flat-only domain mixture; budgets count native vector steps."""

    tribes: tuple[int, ...] = tuple(range(1, 13))
    map_sizes: tuple[int, ...] = (11, 12, 13, 14, 15, 16)
    map_types: tuple[str, ...] = ("lakes", "drylands", "water_world")
    map_type_weights: tuple[float, ...] = (0.60, 0.25, 0.15)
    small_map_probability: float = 0.70
    intro_steps: int = 8_000
    expansion_steps: int = 24_000
    pools: int = 3
    segment_steps: int = 512
    episode_max_moves: int = 3200
    evaluation_groups: int = 6
    domain_score_floor: float = 0.40
    domain_min_games: int = 20

    def __post_init__(self):
        if not self.tribes or len(set(self.tribes)) != len(self.tribes) or any(t not in range(1, 13) for t in self.tribes):
            raise ValueError("flat curriculum tribes must be distinct regular tribe IDs 1..12")
        if not self.map_sizes or len(set(self.map_sizes)) != len(self.map_sizes) or any(s < 11 or s > 32 for s in self.map_sizes):
            raise ValueError("flat curriculum map sizes must be distinct values in 11..32")
        if not self.map_types or len(set(self.map_types)) != len(self.map_types) or any(t not in {"lakes", "drylands", "water_world"} for t in self.map_types):
            raise ValueError("invalid flat curriculum map types")
        from math import isfinite
        if len(self.map_type_weights) != len(self.map_types) or any(not isfinite(w) or w <= 0 for w in self.map_type_weights):
            raise ValueError("flat map-type weights must be finite, positive and match map types")
        if not 0 < self.small_map_probability < 1 or not 0 <= self.intro_steps <= self.expansion_steps:
            raise ValueError("invalid flat curriculum schedule")
        if min(self.pools, self.segment_steps, self.evaluation_groups, self.episode_max_moves) < 1:
            raise ValueError("flat curriculum pool, segment and evaluation counts must be positive")
        if not 0 <= self.domain_score_floor <= 1 or self.domain_min_games < 2:
            raise ValueError("invalid flat domain score floor or sample count")


@dataclass
class TrainingConfig:
    """Self-play and optimisation settings."""

    seed: int = 2137
    device: str = "auto"  # auto, mps, cuda, cpu
    map_size: int = 11
    map_type: str = "lakes"
    # Used by flat mixed-domain collection/evaluation. Other stages keep their
    # existing default two-player setup and never enable the curriculum.
    player_tribes: tuple[int, int] = (3, 2)
    flat_curriculum: Optional[FlatCurriculumConfig] = None
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
    # Legal actions are a public fog observation.  Give hidden tiles named by
    # those actions extra belief supervision so the decoder learns, for
    # example, that a missing/available move constrains an unseen destination.
    legal_action_belief_weight: float = 2.0
    # PolyEnv can compare a sampled belief particle with the public legal
    # action set of the live SelfPlayPool state.  The resulting action
    # differences focus an additional supervised belief loss on hidden action
    # endpoints; the engine itself never receives model losses or gradients.
    engine_action_constraint_weight: float = 0.25
    # Geometry-aware auxiliary belief objective.  In particular, a visible
    # Build Bridge action constrains the hidden cardinal neighbours of its
    # water tile; learning the relation directly is much more sample-efficient
    # than treating every terrain token as independent.
    bridge_constraint_weight: float = 1.0
    # Rare hidden cities and units determine routes, captures and builds. A
    # focal term keeps their presence heads from winning by predicting empty
    # tiles everywhere; the consistency term prevents structurally impossible
    # combinations such as a predicted city on predicted water.
    belief_presence_focal_gamma: float = 1.5
    belief_entity_consistency_weight: float = 0.35

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
    engine_action_constraint_replay_fraction: float = 0.25
    # Bound repeated sampling when belief failures leave only a small replay.
    # Six means that, on average, one newly collected example can contribute
    # to at most six optimiser-batch exposures in this training phase.
    max_replay_epochs: float = 6.0
