# Architecture and information flow

![Neural model and decision paths](assets/architecture.png)

## Shared public-state encoder

The model receives a player-view map with 23 categorical fields per tile,
nine global features, a 12-field visible-event summary and an optional bounded
history of compact 12-field event tokens. The history encoder pools visible
events into the global token before the map Transformer. This is a bounded
history model, not a recurrent hidden state carried between network calls.

The Transformer returns a global state and contextual tile representations.
The relational variant adds mean, maximum and learned attention pooling over
revealed tiles to the global state. Its default initializer uses width 192,
eight map layers, eight attention heads and action interaction width 64.

## Checkpoint-selected policy architectures

| Configuration | Action scorer | Intended use |
| --- | --- | --- |
| `relational` | Joint nonlinear context/type/options/source/target/city interactions | Explicit new experiments initialized with `init_model.py` |
| `factorized` | Sum of type, tile and option scores | Existing checkpoints and ablations; still the generic `ModelConfig` default |
| `attention` | Per-action cross-attention over map tiles | Compatibility with older checkpoints |

All variants produce one score per currently legal candidate, followed by a
padding/legality mask. The size of the output is not a fixed vocabulary of
every possible game action.

The relational scorer gathers low-dimensional source and target features,
resolves public city/unit pointers, and combines them with action semantics
and geometry. It does not build an action-by-map attention matrix. Runtime
unit/city/origin IDs are omitted from its map embeddings; their magnitudes
are not learned features. See the [detailed scorer](relational_architecture.md).

## Value and belief

Value is a scalar in `[-1, 1]`. Its terminal targets use candidate-relative
outcomes: win `+1`, draw `0`, loss `-1`. A learned estimate is not a calibrated
win probability unless evaluated as such. Extra head-only critic fitting can
update value without changing the encoder or policy.

The optional belief branch conditions tile representations on the public legal
action set, refines them spatially and predicts semantic map distributions.
Structured sampling handles terrain, settlements, units and their attributes.
Observed tiles must be preserved; public action constraints reject inconsistent
world proposals. Runtime IDs for sampled entities are materialization details,
not hidden semantic labels the policy is asked to memorize.

The three branches belong to one model. Dedicated training stages choose which
parameters receive gradients; flat forward calls normally skip belief decoding.

## Flat decision path

Public observation → shared encoder → legal-action scores → selected legal
action → real `VectorGameEnv` step. Belief reconstruction and MCTS are not
prerequisites for this path.

## Optional belief-MCTS path

![Belief sampling, search and supervision boundary](assets/belief_mcts.png)

Public observation and legal actions seed belief particles. Accepted detached
worlds are searched with policy priors and value estimates, using PolyEnv's
native pools for tree selection, simulation and backup. Visit distributions
provide policy targets; completed outcomes and configured bootstraps provide
value supervision. Selective MCTS can search a subset of training positions.

`root_shield` provides additional root-action mapping protection. It does not
make a world with an inconsistent public legal-action list valid. Strict
calibration, rejection statistics and raw-versus-search preflight remain needed.

## Ground-truth boundary

The engine owns the actual hidden state. `full_map_numpy()` and native
ground-truth labels are training-only supervision, not neural observation
features or oracle search worlds. The policy and value consume player-view
features; MCTS uses detached hypotheses rather than cloning the true hidden map.

Some older standalone training helpers also use authoritative economy for a
training-only horizon target. This does not turn it into an inference input.
These legacy helpers are not equivalent to the terminal-only supervision of
the dedicated flat collector. Keep training modes explicit in experiments.

## Engine versus learner

PolyEnv determines legal actions, executes rules, materializes detached worlds
and reports action differences. PolyBot owns neural inference, losses,
optimization, sampling configuration, replay, league selection and metrics.
An action-discrepancy mask focuses supervised map learning; it is not a
differentiable loss computed inside the engine.
