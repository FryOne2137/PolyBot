# Observation context v1 and joint tile encoder v2

The same adapter feeds scalar GameEnv, VectorGameEnv, SelfPlayPool and native
MCTS leaves. The acting observer supplies their own technology vector and tribe;
per-player belief requests retain each observer's separate perspective.

## Global input

The first nine legacy features keep their original order and meaning. The
following 126 fields are appended, giving 135 global features in total:

| Context offset | Meaning |
| --- | --- |
| 0–24, 25 | All 25 owned technology bits, then a known flag |
| 26–37, 38 | Own tribe one-hot over IDs 1–12, then a known flag |
| 39, 40 | Initial player count divided by 16, then a known flag |
| 41–56, 57 | Own global PlayerId one-hot, then a known flag |
| 58–73, 74–89 | Active flags for 16 player slots, then per-slot known masks |
| 90–105, 106–121 | Public scores transformed as `asinh(score / 1000)`, then known/participation masks |
| 122–124, 125 | `lakes`, `drylands`, `water_world` one-hot, then a known flag |

Only own technologies and own tribe are consumed. The current engine exports
public scores and the original participant mask, but no public opponent-status
array. Accordingly, opponent status is **unknown**. Neither a zero score nor a
set participation bit is interpreted as elimination or survival. The acting
observer is known active before game over. An explicitly supplied public
`player_status` array can provide known statuses; missing entries remain unknown.

A known empty technology tree differs from missing technology data. Scores are
stored as float32 context in compact PPO/PPG/value replay, avoiding int16 score
overflow. Context is also retained in supervised belief/MCTS examples.

## City tiles

Map columns 0–22 remain the existing categorical schema. Column 23 is signed
`own_city_population`. A single `TileEncoder` consumes the complete tile:

1. Encode categorical fields, including city level, with field embeddings.
2. Normalize the signed population as `asinh(population / 8)` and attach its
   availability mask for a revealed own city.
3. Concatenate the categorical embedding sum and these two values, then run
   **one joint MLP: `(d_model + 2) -> d_model -> d_model`, GELU and LayerNorm**.
4. Add X/Y positions and send the tile vector to the shared map Transformer.

At width 192, the joint fusion is `194 -> 192 -> 192`. Population has no separate
learned projection, addition or branch. The MLP can combine population with city
level and other tile semantics before attention. The next-level threshold is
derivable from the existing city level (`level + 1` under the pinned rules), so
it is no longer duplicated as an extra input. Negative population is valid.

Population on foreign or hidden tiles is ignored, even if malformed input
supplies a number. Full-map labels never become observation features. Population
is an auxiliary continuous input, not an additional hidden-map classification
head. Belief materialization preserves observed population and the packet width.
Relational models continue to use runtime unit/city/origin IDs only as pointer
keys, not learned semantic categories; renaming IDs cannot change predictions.

## Architecture versions and training

The joint tile encoder is initialized normally and learns from the first update.
The observer/public context still has its global-token MLP with zero-initialized
output. Policy, value and belief share the same tile encoder in every inference
path, including scalar/vector collection, native MCTS leaves and reanalysis.
This applies to every map-encoding profile; policy has one shared-token scorer.

Checkpoints record `observation_context_version=1` and `tile_encoder_version=2`.
The [shared action encoder](shared_actions.md) additionally records
`action_encoder_version=2`. The [semantic history encoder](history_encoder.md)
records `history_encoder_version=2`. Both are required by all current training
entry points.
Weights and optimizer state from previous tile/history encoders or parallel
scorers are incompatible.
Initialize a new model with `scripts/init_model.py`, use a separate checkpoint
directory and league database, and train from scratch. Loading previous weights
fails explicitly rather than silently substituting a new randomly initialized
encoder. The map Transformer and action scorer retain their existing sizes.

Legacy replay with nine globals and 23 map columns is padded with unknown
context and a distinct population sentinel (`-32768`); it is not interpreted as
a known empty tech tree or zero population. Legacy data remains usable even
though the old network weights are not interchangeable with the new architecture.

Normal policy/value training can update the shared input encoders. Belief-only
training retains its existing frozen shared encoder; head-only value fitting
also retains its existing freeze boundary. These stages do not train the shared
tile encoder by themselves.

Input support is separate from collection curricula: this change does not enable
mixed-size batches, multiplayer value training or random tribe/map rotation.
The native flat population collector still uses its supported `lakes` path.
