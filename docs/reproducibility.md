# Reproducibility and model card

## Scope and intended use

PolyBot is research software for learning strategies in the unofficial PolyEnv
environment. The code supports neural policy/value inference, a structured
hidden-map decoder and optional search. It is not a released competitive
Polytopia player, a reproduction of a complete AlphaZero/AlphaStar system,
or a guarantee of continual improvement.

No pretrained weights or independently established playing-strength results are
shipped in the source tree. Architecture expressiveness tests establish a
specific capability of the scorer, not tournament performance.

## Input and training data

- Inputs: player-visible map fields, public global features, visible history,
  and descriptions of the current legal actions.
- Flat data: engine-generated self-play against frozen model opponents, with
  optional supervised heuristic bootstraps and configured shaping.
- Belief labels: full semantic map labels obtained only for training, plus
  public legal-action discrepancy evidence.
- Search labels: accepted MCTS visit distributions and configured value targets
  from detached hypothetical worlds and game outcomes.

Native flat population training currently supports the `lakes` map path. Do not
assume that a generic CLI `--map-type` choice implies support in every collector.
Most current training examples use two players, Bardur/Imperius and 11×11 maps.
Support for a larger input shape does not establish generalization to that size,
different tribes, longer histories or another ruleset.

## Minimum experiment record

Record the PolyBot commit and uncommitted changes, PolyEnv revision, Python,
PyTorch/NumPy versions, device, architecture, initial weights, initialization and
collection seeds, evaluation schedules, opponents and sampling weights, move
limits, rollout lengths, all losses/weights and optimization budgets.

`init_model.py` preserves random initialization. Normal runs save `args.json`,
`config.json`, `metrics.jsonl`, CSV, event records, summaries and plots when
metrics exist. Flat scientific artifacts add source snapshots, model
fingerprints, initial/selected weights, validation positions and per-block
optimization state. Check the actual run directory; filenames and stage nesting
depend on the entry point.

These artifacts can contain absolute paths, environment details and private
experiment information. Keep them out of source control. Publish a deliberately
reviewed experiment bundle separately from the source release.

## What an evaluation means

Match score is `wins + 0.5 × draws`, divided by games. Paired arenas swap the
candidate's player side on each map. Report opponents, maps, game counts,
truncated games, policy/search mode and move limit together with the score.

`lower95` is a lower confidence diagnostic, not the win rate itself. Some
diagnostics use approximate map-level intervals. Both games on a shared map
are correlated; do not count them as independent maps. Tiny samples, repeated
selection and adaptive development gates limit the interpretation of intervals.

Development gates protect training, but repeated decisions can overfit them.
Reserve new final maps and opponents for the report's final test. Use several
training seeds and pair comparison workloads across variants.

Low belief action mismatch does not mean every hidden tile is correct.
Collection-time value MAE measures earlier predictions, not a fresh validation
of the selected checkpoint. Lower losses, more updates, an EMA checkpoint or
promotion against a small population do not establish broad improvement.

## Resuming

Model `.pt` files contain weights and `ModelConfig`. A champion alone can
initialize a new experiment, but does not preserve the opponent archive,
optimizer, replay or exact training continuation.

Back up compatible league models, the SQLite database and any intended replay
and continuation artifacts. Stop writers before copying a live database, or use
SQLite's backup API. Its paths may need attention after moving machines.

`--flat-resume-state` restores working weights and optimization/RNG state under
compatibility checks, with a new curriculum budget. Native in-flight games and
local opponents restart. It is not a bitwise continuation of the whole process.
See [the detailed limits](flat_improvements.md#continuation-and-artifacts).

## Existing benchmark

[relational_cpu.json](benchmarks/relational_cpu.json) records a synthetic CPU
tensor benchmark with explicit dimensions and software/platform metadata. It
does not measure engine time, validation cost, GPU throughput or playing strength.
Use `benchmark_architecture.py` on the target hardware before allocating a long
run. Publish both raw measurements and the exact command.
