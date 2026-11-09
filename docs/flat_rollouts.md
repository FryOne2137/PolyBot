# Fresh flat-policy rollouts

The dedicated flat trainer enables `--flat-population-synchronous-rollouts`.
Compatible checkpoints retain their scorer architecture. The shared tile encoder
is version 2 and requires a new experiment if upgrading from earlier weights.
The other training stages keep their own collectors.

## Policy fragments

The model collects `--direct-population-ppo-rollout-steps` vector steps without
changing the behavior weights, then finishes GAE fragments and runs PPO and
scheduled PPG phases. Unfinished games continue. Fragment and block boundaries
use value bootstraps, not invented draws.

Policy age is measured before updating. Synchronous fragments should enter with
age zero. The monotonic policy version advances after an accepted PPO epoch or
PPG phase; a rejected PPO epoch does not advance it. Old actor data is not made
fresh by resetting a block counter.

## Full-game outcomes

Each environment separately keeps a bounded reservoir of public positions from
the whole game, including early fragments already consumed by PPO. A true
terminal assigns candidate-relative outcomes `+1 / 0 / -1`.

These labels can train PPG with policy-KL protection and configured value-only
replay. They are not reused as old actor targets. In synchronous mode the older
terminal replay updates the value head only. Rollback clears rejected actor and
bootstrapped auxiliary data while preserving genuine terminal critic labels
where configured. Live-game reservoirs continue across compatible blocks.

Earlier policy decisions learn through GAE/bootstraps and improved value
representations. Increasing rollout length can extend direct terminal coverage;
it does not retroactively label old PPO decisions as current-policy samples.

## Diagnostics

The `direct population learning` log reports:

- `actor`: unique observed samples eligible for policy loss;
- `applied`: actor sample presentations versus all accepted PPO presentations;
- `stale`: samples rejected by the age limit;
- `terminal_actor`: samples with terminal-return supervision;
- `age mean/p95/max`: the age distribution before updating.

Several epochs can present a sample more than once. Coverage metrics distinguish
unique samples from presentations and include opening/mid/late terminal coverage.

`direct population terminal value (collection predictions)` reports MAE, RMSE,
bias and correlation against true outcomes, with PPG terminal-label coverage.
These are collection-time predictions, not a new validation of selected weights.
No completed game means no terminal calibration measurement, not zero error.

## Controls

```text
--flat-population-synchronous-rollouts --direct-population-ppo-rollout-steps 128
```

`--no-flat-population-synchronous-rollouts` selects the older collector for an
explicit ablation. The new flat package can anneal the fragment horizon through
`--flat-rollout-steps-final`; see [flat improvements](flat_improvements.md).

Tests verify freshness and continuation semantics. A playing-strength benefit
still needs paired experiments with fixed evaluation conditions.
