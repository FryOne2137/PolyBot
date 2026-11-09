# Complete legal actions and shared action encoder

Every real legal candidate must reach policy, value and belief. Padding is not
an action. Candidate filtering for MCTS happens after the neural forward and
does not remove observations from the shared encoder.

## Native buffers

The pinned PolyEnv vector/pool buffer capacities are immutable. PolyBot reserves
the complete stable `ActionSpace`, whose size is obtained from the engine for
the board size. SelfPlayPool uses its documented `max_actions=0` exact mode and
`require_all_actions=True`. VectorGameEnv and MctsPool use the full action-space
capacity rather than interpreting the CLI hint as a truncation limit.

Reusable full-capacity buffers stay on the host. Returned/model packets remove
only trailing padding, keeping every legal entry in its original order. Packets
are validated against `action_truncated` and `total_legal_action_count`; partial
packets fail before inference or replay capture. External buffer dictionaries
can resize their compact action arrays after a step without changing native game
states, RNG, seeds, event history or checked position IDs.
Only two native leaf-buffer capacities are cached, preventing host memory from
growing with every historical active-forest size during a long run.

The full native allocation increases host memory usage. For the pinned engine,
an 11x11 board's stable space contains 45,522 IDs; this is not the number of
legal actions processed by the network on every move. The network receives the
actual list (14 actions at the tested initial position). `max_actions` settings
remain capacity hints/minima, not neural observation limits. Start large native
forests with modest environment/pending-leaf counts and measure host RAM.

PPO, objective/PPG replay and terminal value reservoirs retain compact complete
observations. Mixed-length examples are padded only when batching. Independent
native MCTS inference requests are also padded together, then split back to
each request's action width. Native backup receives its required full-capacity
logit buffer, with unused entries at negative infinity.

Training emits `full_legal_actions` events with the native capacity. The adapter
tracks the peak number of actual legal actions seen. Do not reuse old replay
whose lists may already have been truncated: missing actions cannot be recovered
from a stored partial observation. Start a new experiment/replay directory.

## Shared learned representation

`action_encoder_version=2` uses default width 64, four attention heads and 16
learned latent tokens. The map Transformer remains width 192 with eight layers.

Each action combines semantic type/options, normalized cost/economy/combat
features, shared map state and source/target/city tile representations. Runtime
unit/city IDs resolve pointers; action-list indexes have no learned position
embeddings. Source-target geometry is explicit.

Induced set attention reads **all A legal actions** into 16 latent tokens and
writes context back to **all A action tokens**. Its attention cost is O(A*16),
instead of an A-by-A self-attention matrix. Latent aggregation is learned
compression; complete input does not mean lossless internal representation.
Candidate permutations preserve policy correspondence and leave value/belief
unchanged in evaluation, apart from floating point rounding.

- Policy applies only `LayerNorm -> Linear(64,128) -> GELU -> Linear(128,1)`
  to each contextual action token, then masks padding. No independent raw-action
  scorer or sum of parallel logits remains. Raw type/options, costs and endpoints
  enter only the shared action encoder. The private belief endpoint-evidence
  masks still describe which public actions refer to each tile.
- Value attends to the full contextual action set, combines it with map state,
  then uses the existing scalar head. Head-only fitting, terminal replay and PPG
  consume this same action-conditioned representation.
- Belief projects the shared tokens to map width, uses tile-to-action attention
  plus public endpoint evidence, then its private spatial refiner and semantic
  heads. It no longer maintains a separate raw-action embedding encoder.

All three losses can train shared action features during joint training.
Belief-only calibration freezes `action_encoder` and policy/value parameters;
belief-specific projections/refinement remain trainable. Head-only value fitting
changes only `value_head`. Empty action sets use a neutral masked path with
finite gradients, rather than softmax over entirely masked keys.

## Initialization

This is a new architecture requiring random initialization. Checkpoints record
both `tile_encoder_version=2` and `action_encoder_version=2`, plus action width,
heads and latent count. Semantic history additionally records
`history_encoder_version=2`; see [the history schema](history_encoder.md).
Earlier weights and optimizer continuation files are
rejected explicitly. `init_model.py` exposes `--action-encoder-dim`,
`--action-encoder-heads` and `--action-latent-count` and saves them in experiment
artifacts. The default relational network has 6,945,185 parameters
(4,980,674 on the flat path and 1,964,511 belief-specific). The old parallel
relational scorer and legacy factorized/attention scorers have been removed.
The historical `policy_architecture` names now select map-encoding profiles
only: `relational` uses runtime-ID-invariant inputs and spatial readout;
`factorized` and `attention` both use the base map/global profile. All profiles
use the same single shared-action scoring path. Playing-strength improvements still require controlled evaluation.
