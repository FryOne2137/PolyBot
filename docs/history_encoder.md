# Semantic visible-event history

`history_encoder_version=2` consumes the same public 12-field event records as
before. The scalar, vector, replay and native MCTS adapters do not change their
journal layout. Full hidden-state labels never enter this encoder.

## Fields and missing values

- Event type, actor player, actor tribe, visibility flags, source/target unit
  type and detail kind have separate categorical embedding tables.
- The detail value uses separate tables for tile actions, buildings, spawned
  units and unit upgrades. Matching numeric IDs in different namespaces are
  not the same event feature.
- Source/target tile indexes are decoded with the actual board width into X/Y.
  Coordinate zero is a real position, distinct from unknown (`-1`).
- Observer-local unit IDs contribute presence bits only. Consistent ID renaming
  cannot change a prediction; numeric IDs are not stable semantic categories.
- Unknown/out-of-schema categories use an explicit unknown embedding. Valid
  unit types throughout the supported 0..127 range are distinct; the old
  division by 32 and clamp incorrectly merged types 31 and above.

At map width 192, categorical/coordinate embeddings have width 32. Their
concatenation and two ID-presence bits pass through `386 -> 192 -> 192`, GELU
and LayerNorm. The existing one-layer history Transformer and latest-relative
position embeddings then pool valid events into the global map token.

The history gate starts at zero contribution and learns to open. Padding is
masked inside history attention and pooling. A row with no valid events returns
exactly zero even when other batch rows have long histories or a trained gate
has nonzero biases.

## Training and compatibility

All training entry points use this encoder through `PolicyValueBeliefNet`.
Joint policy/value training can learn it. Existing belief-only calibration and
head-only value fitting continue to freeze shared parameters.

Checkpoints and flat optimizer continuation record `history_encoder_version=2`.
Weights without this field or with a previous version are rejected before any
partial restore. Reinitialize the model and bootstrap into a new experiment
directory. Old raw event records remain readable; old history weights are not
silently reinterpreted as categorical embeddings.

Tests cover high unit IDs and their gradients, typed detail categories, board
widths, unknown positions, ID renaming, full-model history sensitivity, padding,
mixed history lengths and checkpoint round trips.
