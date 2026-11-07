# Flat optimization and experiment controls

`train_flat.py` enables `--flat-improvements` by default. Generic league,
belief and MCTS entry points do not enable it by default, and the league rejects
using the package outside the flat stage. `--no-flat-improvements` selects the
previous package behavior while preserving independent flat options.

## Head-only value fitting

A separate bounded replay stores public positions with true final outcomes:
win `+1`, draw `0`, loss `-1`. Full-game reservoirs preserve earlier
positions independently of PPO fragment boundaries.

After a block, the additional critic fit updates **only the value head**.
The encoder, relational spatial readout and policy do not receive its gradients.
Defaults are eight updates, batch 128, capacity 16,384 and learning rate 0.0001.
Opening/mid/late phase buckets are sampled evenly when present; wins and losses
are not artificially balanced.

Validation positions are copied from the first configured games of the baseline
development arena against the first opponent. Both player sides are included;
truncated games produce no terminal labels. A requested validation count larger
than the diagnostic gate increases the baseline arena accordingly.

The same frozen public positions measure before/after MAE, RMSE, bias and
phase metrics. A worse RMSE or nonfinite gradient restores the previous head.
An accepted external head update clears that head's outdated main-AdamW moments.
The log explicitly distinguishes a fit, a rollback and insufficient data.

**Interpretation:** these labels describe outcomes under the baseline reference
policy, not an oracle value for every later policy. They are a fitting diagnostic.
Repeated development-set selection can overfit. Independent game-strength and
final test arenas remain necessary.

Controls: `--flat-value-updates`, `--flat-value-batch-size`,
`--flat-value-learning-rate`, `--flat-value-capacity`,
`--flat-value-validation-games`. Zero updates disable this extra fit.
The older `--flat-population-terminal-replay-value-weight` is separate; set
it to 0.0 consistently when isolating the new critic fit in an experiment.

## Rollout horizon

Fragments can grow from `--direct-population-ppo-rollout-steps` toward
`--flat-rollout-steps-final` (default 256) over the maximum curriculum budget.
Weights remain fixed inside each fragment. A final horizon of zero disables
annealing. Longer fragments reduce update frequency and need empirical testing.

## Paired rollback confirmation

Before throwing away an apparent population regression, candidate and previous
best play the weakest matchup on the same additional maps with swapped sides.
Differences are clustered by map. The interval is an approximation, not an exact
statistical guarantee. Fewer than eight complete pairs or mismatched evidence
cannot confirm regression.

An unconfirmed loss permits working-weight continuation without replacing the
protected best checkpoint. A clearly failed champion screen may stop earlier.
Promotion thresholds remain separate.

Use `--flat-population-regression-confirm-games` for the confirmation budget.
`--no-flat-paired-rollback` selects the older point-score comparison.

## Evaluation cache

Identical raw evaluations can be reused within a curriculum. Keys include
actual model/configuration fingerprints, map schedule, history, move limit,
game count and execution settings. Changing weights or conditions invalidates
the match. A different population weighting can reuse individual match results.

Cached match evidence still reaches the audit callbacks. Collecting new
validation positions bypasses the score-only cache. Existing round-level seed
rotation still applies. Disable with `--no-flat-cache-evaluations`.

## Main exploiter and anchors

With `--breakthrough-exploiter` and `--flat-exploiter-objective`, the
exploiter's fixed objective is the current main. Its independent archive
admission checks that matchup, rather than requiring broad historical strength.
It does not replace the main candidate. Its policy anchor retains the
exploiter's own setting.

A weaker main anchor is an optional ablation, not an automatic guarantee of
improvement. `--no-flat-exploiter-objective` restores the previous objective.

## Continuation and artifacts

Alongside normal run records, scientific artifacts include:

| File | Purpose |
| --- | --- |
| `experiment.json` | Code/environment identity, settings and model fingerprints |
| `source_snapshot.json` | Actual Python sources, including uncommitted edits |
| `initial_model.pt` | Weights at curriculum entry |
| `value_validation.pt` | Frozen public reference positions and outcomes |
| `events.jsonl` | Rollback, critic-fit and final-selection decisions |
| `latest_state.pt` | Post-block working weights, best weights, AdamW/scaler, policy version, terminal replay and RNG |
| `selected_model.pt` | Model selected by the final curriculum gate |

`--flat-resume-state PATH` restores compatible working weights, AdamW,
learning rate and RNG for one main branch. It checks architecture, public
inputs and champion fingerprint. Changed arena conditions prevent reuse of
the saved terminal replay.

This starts a **new curriculum budget**. Native in-flight games and local
opponents restart, and gates/high-water selection are re-established. It is
not bitwise whole-process restoration. Post-block working state can differ
from final selected weights. Use `--challenger-in` for ordinary model-only
continuation from the accepted checkpoint.

Keep original opponent files referenced by the league database. Disable extra
scientific files with `--no-flat-scientific-artifacts`.

## Multi-seed ablations

```bash
python scripts/flat_experiment_matrix.py --output runs/ablations-01 --seeds 101,202,303 --variants control,critic,critic_credit,full -- python scripts/train_flat.py --champion-in models/demo/teacher.pt --league-db runs/demo/league.sqlite --direct-population-steps 8000 --direct-population-max-steps 16000 --flat-population-terminal-replay-value-weight 0.0
```

Without `--execute`, this writes a plan and directories only. With it, trials
run sequentially with paired training seeds and common evaluation maps. Each
trial has independent models, logs and a copy of the same initial league.
Database copies begin only on execution and can require substantial disk space.

Variants: `control`, `critic`, `critic_credit`, `full`, optionally
`weak_anchor` and `exploiter`. Each trial runs one round with the base
command's step budgets. Optimizer warm-start arguments are rejected to keep
initial conditions comparable. Full stdout/stderr goes to `console.log`;
`plan.json` preserves commands and exit status.

See [reproducibility](reproducibility.md) for validation interpretation and
final hold-out tests.
