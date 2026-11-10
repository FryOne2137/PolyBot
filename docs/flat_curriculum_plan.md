# Diverse flat training curriculum

`train_flat.py --flat-curriculum` enables one learner across the twelve regular
tribes, variable square board sizes and three map types. This is an opt-in
flat-only collector. The default flat command remains the two-tribe baseline;
belief, MCTS and teacher entry points do not enable this package.

## Start a run

Use a current, compatible model and an independent league/run directory:

```bash
python scripts/train_flat.py --champion-in models/demo/teacher.pt --checkpoint-dir models/mixed-flat --checkpoint-prefix flat --league-db runs/mixed-flat/league.sqlite --project runs/mixed-flat --name curriculum --seed 314159 --evaluation-seed 1531842 --rounds 6 --flat-curriculum --flat-tribes all --flat-map-sizes 11,12,13,14,15,16 --flat-map-types lakes,drylands,water_world --flat-map-type-weights 0.60,0.25,0.15 --flat-domain-pools 3 --flat-domain-segment-steps 512 --flat-domain-max-moves 3200 --flat-domain-evaluation-groups 6 --direct-population-envs 8 --direct-population-steps 24000 --direct-population-max-steps 48000 --post-bootstrap-population-steps 24000 --post-bootstrap-population-max-steps 48000 --direct-population-block-steps 4000 --direct-population-gate-games 100 --direct-population-diagnostic-gate-games 60 --promotion-games 400 --historical-games 80 --direct-population-ppo-rollout-steps 128 --direct-population-ppo-epochs 1 --direct-population-ppo-minibatch-size 128 --direct-population-ppo-top-advantage-fraction 1.0 --direct-population-ppo-terminal-advantage-weight 0.75 --direct-population-shaping-weight 0.05 --direct-population-shaping-final-weight 0.0 --flat-population-terminal-replay-value-weight 0.0 --flat-population-regression-confirm-games 80 --flat-value-validation-games 12 --no-breakthrough-exploiter --num-threads 8 --learning-rate 0.00001 --max-moves 1600 --visible-action-history 32 --device cuda
```

Initialization and supervised bootstrap are described in [training](training.md).
All current model schema checks still apply, including semantic history v2.
No additional neural architecture change is required by the mixed collector.
PPO must use persistent optimization and synchronous rollouts (already enabled
by the dedicated flat entry point).

## Domain schedule

The sampler balances game starts separately for both seats across regular tribe
IDs 1..12; names/IDs can restrict `--flat-tribes`. Mirror matches are allowed.
The candidate seat and opponent checkpoint are independent of tribe selection.
All selected tribes are eligible from the first group.

The stage uses cumulative collected vector steps, preserved across main league
rounds in one process and in optimizer continuation files:

| Steps | Eligible sizes | Eligible map types |
| --- | --- | --- |
| Before `--flat-curriculum-intro-steps` (8000) | Smallest requested size | Lakes, or first requested type |
| Before `--flat-curriculum-expansion-steps` (24000) | Requested sizes at most 13 | All requested types |
| Later | All requested sizes | All requested types |

With both small and large sizes eligible, 70% of the size prior is assigned
uniformly to sizes at most 13, and 30% to larger sizes. Change this through
`--flat-small-map-probability`. Map-type weights default to lakes 60%, drylands
25%, water_world 15%. Set both stage thresholds to zero to expose the full
mixture immediately. The schedule is exposure-based, not an assertion of
playing-strength readiness. Phase changes affect new groups; live games drain
under the configuration in which they started.

## Native collectors and budgets

`VectorGameEnv` fixes size, type and ordered tribes at construction. A bounded
set of native groups therefore runs homogeneous batches. Groups are advanced
in consecutive segments using one shared model, AdamW and monotonic policy
version. Each segment freezes collection until its PPO update; another group's
in-flight game is paused, never reset by a PPO block. Opponents already assigned
to live games remain available until those games finish.

`--direct-population-envs` is the width of each group. Three pools of eight
create at most 24 resident games; a segment advances one group at a time.
Step budgets count the sum of group vector steps. Logs additionally record
actual environment steps, including partially drained groups. A deficit
scheduler steers exposure among available size/type groups; finite pool counts
and unfinished games can temporarily differ from the configured prior.

Completed slots are stopped using checked native partial steps. Once every
slot has finished, the group can receive newly sampled tribes, size and type.
An explicit `--flat-domain-max-moves` horizon also retires long unresolved
slots. Such episodes bootstrap PPO/GAE, are reported as truncated, and create
no terminal value labels. A block/segment boundary alone never creates an
outcome. Both true outcomes and truncations are recorded in game callbacks.

Legal lists remain complete. Training groups of different map sizes are inferred
in separate batches; auxiliary replay stays local to
its homogeneous group. Terminal critic replay spans domains, preserves domain
metadata, and is grouped by tile count for inference without changing sample
order or map-type features.

## Evaluation and promotion

Every flat arena uses a deterministic weighted domain schedule, independent
of the learner's current curriculum stage. Paired games share a map and tribe
configuration and swap the candidate seat. At least one group per requested
map type is allocated when the game budget permits; extra groups follow the
type prior. Sizes use stratified draws from the full size prior. Balanced tribe
coverage means six groups cover all twelve candidate tribes when all are
selected. Larger `--flat-domain-evaluation-groups` increases domain coverage
and native setup/inference overhead. A small diagnostic is a sampled benchmark,
not exhaustive coverage of all size/type/tribe combinations.

Arena groups advance together instead of waiting for each domain to finish.
Inference combines live games across tribes/types and nearby sizes. Smaller
boards use masked trailing storage with explicit per-row board sizes; real tile
indexes, coordinates, history and action endpoints retain their original meaning.
Padding does not become visible or hidden map content. This changes no parameters
or checkpoint schema, and belief/MCTS do not use this padded inference path.

`--flat-evaluation-batch-games` defaults to 64 resident games, not 64 extra test
games. Larger promotion arenas run bounded chunks of the same seed schedule.
Completed rows are excluded from inference. Size buckets limit additional tile
padding to 50%, and a separate action-token budget splits inference batches
without truncating legal lists. Identical policy weights share one inference
route even if value heads differ. For a different memory/throughput tradeoff,
set an even resident-game limit (for example 32 or 128) and measure on the target
GPU. No hardware-independent speedup is guaranteed.

Public CPU packets are now combined before transfer: each actual model forward
receives one compact native input batch. CUDA staging reuses contiguous pinned
allocations and retains int32 action/map storage until conversion on the GPU.
This avoids per-domain H2D copies followed by GPU row-selection/concatenation.
For identical paired policies, a decision can be reused in the other seat's game
only after checking equality of its public inputs; both native games still advance
and contribute their own outcome/position records.

`--flat-evaluation-size-bucketing nearby` retains the padded size buckets.
`--flat-evaluation-size-bucketing exact` batches only equal sizes, enabling the
ordinary unpadded map-transformer path. The faster option depends on the GPU,
batch population and legal-action lengths. Compare them on the actual checkpoint:

```bash
python scripts/benchmark_flat_evaluation.py --model models/mixed-flat/bootstrap.pt --device cuda --games 40 --max-moves 100 --num-threads 4 --batch-games 64 --output runs/mixed-flat/evaluation-throughput.json
```

The probe warms each case, then measures single-board evaluation, a three-type
mixture restricted to 11x11, and the full size mixture with each bucketing mode.
It writes wall time, actual environment steps/s, average game length, runtime
versions, model fingerprint and GPU/VRAM details. Short truncated games are a
throughput diagnostic, not evidence of playing strength. Compare steps/s together
with domain sizes and steps performed: seconds/game alone mixes software cost
with longer games on larger maps.

One progress bar covers the whole arena. Results also record mean/peak inference
batch size, model forward count, resident games, tile-padding ratio and whether
the policy route was shared. These counters describe batching, not GPU utilization
or training progress.
The progress bar and final `mixed evaluation results` line show disjoint counts:
`wins`, `losses`, terminal `draws`, and `truncated` games stopped at the arena
move limit. They update as individual games finish, even while other games in
their native cohort remain live. Their sum equals completed games. Matching
`flat_evaluation_wins`, `flat_evaluation_losses`, `flat_evaluation_terminal_draws`
and `flat_evaluation_truncated_games` counters are returned for run artifacts.
The existing aggregate `draws`/score contract still counts a move-limit game as
half a point; promotion thresholds and game outcomes are unchanged.
Timing diagnostics additionally separate CPU preparation, policy input/inference,
native engine/position callbacks and arena setup. The inference interval includes
H2D copies and the action-result synchronization; it is not isolated kernel time.

The arena horizon is calibrated to 11x11: a limit of 1600 becomes
`ceil(1600 * size^2 / 121)` on each board, about 3386 on 16x16. Each game
records its actual limit. The independent training episode cap remains the
explicit `--flat-domain-max-moves` value.

The aggregate score uses recorded group weights, not a raw average over uneven
stratum counts. Confidence diagnostics cluster seat swaps by map and use a
weighted approximation with a Wilson guard against degenerate all-win samples.
Paired rollback requires matching domain identities and weights as well as map
seeds. Scores, completion and counts are reported by map type, size and own tribe.

`--flat-domain-score-floor` (0.40) prevents final selection/promotion when a
sufficiently sampled size or map-type stratum is clearly below that floor.
`--flat-domain-min-games` defaults to 20; the upper confidence diagnostic must
also lie below the floor before the stratum is flagged. The existing champion
and historical promotion thresholds still apply.

Finite training groups reserve the lower half of the positive signed seed
range; mixed evaluation uses the upper half. This prevents training/evaluation
map-seed collisions even when user base seeds overlap. Actual seeds and domains
are saved. Development schedules rotate across league rounds; keep independent
final-test arenas for publication.

## Continuation and logs

`latest_state.pt` includes domain settings, sampler RNG, stage/exposure counters
and tribe-start counts. In-process league rounds retain live groups; process
resume restarts native games explicitly. Main optimizer state and the shared
critic replay resume under domain-signature checks. Per-group live traces and
local auxiliary replay restart with the native games. No exact whole-process
restoration is claimed. Model-only continuation starts a new curriculum budget.

Metrics/events include group creation, actual domains, phase, native seed,
actor sample shares, real endings and truncations. Match records include size,
type, tribe tuple and evaluation weight. Nested domain-stage logs keep all
segment measurements; only block summaries render plots to avoid plotting
hundreds of tiny segment runs.

The optional existing main exploiter can use the same machinery with its own
session. Start with it disabled to measure the mixed main collector's effect.
The implementation remains two-player: multiplayer turn/value/search handling
is a separate change.
