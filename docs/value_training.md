# Training a value head for Polytopia

The turn-start experiment trains a **value head on player-visible observations
at the start of a turn** and uses it to score candidate plays **after the
opponent's full reply**. Train the head against completed game outcomes while
keeping policy, belief and the shared encoder fixed. Compare the resulting
search on paired maps and measure elapsed time for each arm. Run it with
`train_value.py --position-source mcts-turn-start`. The original
`mcts-terminal` mode samples positions throughout the game.

The implemented pilot samples the first decision after control passes to the
apprentice. At the start of each live apprentice turn, the planner compares
the top legal first actions under the frozen policy. It plays each through the
remainder of the turn and the opponent's reply in the same accepted belief
worlds, using greedy policies for both players. It scores the next apprentice
turn or the exact terminal outcome. The chosen first action is executed in
the real game. The frozen policy handles subsequent decisions within the turn
using their current observations. A world contributes to ranking only when
all candidate branches reach their scoring point. If none qualify, the
first action comes from the frozen policy. The report compares completed
map pairs with the native MCTS baseline and records these policy-only turns.

## Why change the positions that value sees?

A player can take several actions before ending a turn. A search budget of
32 individual action expansions can therefore spend much of its work on one
side's decisions before seeing a response. For example, a head might favor an
attack immediately after it captures a city. Scoring after the opponent's turn
gives the simulator a chance to show the recapture. The number of expansions that
reach the opponent's first decision is worth measuring in PolyBot. The turn
structure and branching problem are documented in the Polytopia-inspired
[Tribes research framework][tribes].

Kevin Zhang's Polytopia project reports a concrete precedent. Its planner
samples candidate turns from a frozen policy, simulates an opponent reply
and scores the next player turn using a critic trained on fogged inputs.
In the author's experiments, this search gained strength over the raw
policy, with gains varying by checkpoint. The project also used a separate
full-state critic for PPO training. These results come from the author's
engineering report and a different engine, policy and compute budget.
[Project report][zhang] · [Companion repository][zhang-code]

PolyBot's own results motivate the same comparison. The terminal-only
experiment improved selection RMSE from 0.877 to 0.868 over 16 games. On a
fresh matched arena, value at 0.25 won 7/24 games against 11/24 with the
candidate's value disabled. On a second, smaller arena, those counts were
6/12 and 4/12. In the four replayed first divergences, the two heads chose
the same move three times. The one position on which they chose different
moves yielded three wins from either move. These runs leave the source of
the gameplay difference unresolved.

## The value target

For the two-player zero-sum setting, let `h_i` be player `i`'s observation
and visible action history, and let `z_i` be that player's terminal result.
Train `V(h_i)` to predict the expected `z_i` under the frozen policies used
for continuation. Use +1 for a win, 0 for a terminal draw and -1 for a loss.
Exclude games cut off by the move limit from terminal supervision. A budget
of several actions within one turn retains the same player perspective.
When backing up estimates in a two-player tree, change sign when the
perspective changes. Terminal outcome regression provides the control used
in the AlphaZero family. [AlphaZero][alphazero]

The input is the same player-view packet used at inference, including the
available legal actions and visible history. A sampled hidden map can drive
simulation. Each decision still uses the observation that the current
simulated player would receive. The true hidden map may supply supervision
offline. The deployed value head reads the masked packet, as described in the
[model architecture](architecture.md#ground-truth-boundary). Public-belief
methods such as ReBeL and Student of Games additionally condition their
values on distributions over both players' information states. They give
a principled direction for future work when one scalar from the current
observation is insufficient. [ISMCTS][ismcts] · [ReBeL][rebel] · [Student of Games][sog]

## Experimental controls

1. Collect seat-swapped full games with fixed checkpoints. Start with
   Bardur versus Imperius on 11×11 Lakes and the same frozen opponent in
   every arm. The current terminal collector can provide full-game outcomes.
   Record which decisions occur immediately after control passes to the
   player. Sample across opening, middle and late game and keep a cap per
   game so longer games do not dominate. Record the behavior policies and
   map seeds along with each sampled position.
2. Fit two heads initialized from the same checkpoint. The control samples
   decisions throughout the game, as in `train_value.py --position-source
   mcts-terminal`. The proposed head samples the start of the player's
   turn. Match the number of samples per phase and the update budget.
   Regress both heads on the same type of terminal label with MSE, and hold
   out complete maps, including both seat assignments.
3. On a small set of held-out decisions, simulate plausible plays through
   the end of the current turn and through the opponent's full reply. Score
   at the start of the original player's next turn, or use the exact result
   when a branch terminates sooner. Compare candidates across the same
   accepted belief samples and pair random streams for replies. Sample
   candidate plays from the fixed policy using legal observations. Average
   evidence for each candidate first action across belief worlds before
   choosing it. A follow-up can replan within the turn as new information
   arrives. Compare the existing heads at this horizon to estimate how much
   comes from the search change.
   [Polytopia search experiment][zhang]
4. Benchmark the approaches on fresh maps with both seat assignments.
   Within each search implementation, compare head variants while holding
   policy, belief, opponent and candidate budget fixed. Include the current
   `(0,1)` MCTS@32 reference, where 0 disables apprentice leaf value and 1
   keeps the opponent's leaf value inside the tree. Compare planners at
   similar total search time per real turn. Report wins, draws, losses, truncations and
   paired score differences. Use the map as the sampling unit when
   reporting uncertainty. Select settings on development maps and verify
   the selected setup once on an independent set.

Two diagnostic measures make the benchmark easier to explain. Report RMSE
and bias by phase and on positions at the intended search horizon, with a
constant-zero reference. Also record the score difference between competing
moves under a shared continuation policy. When both moves repeatedly reach
the same result, that position provides little evidence about their ranking.
The existing `mcts-value-regression` mode provides a baseline for these
checks at the individual-action horizon. [Training entry points](training.md)

## Matched terminal and soft-Z targets

`train_value.py --position-source mcts-soft-z` collects completed paired
games with MCTS on the apprentice and a fixed raw opponent. It samples
apprentice turn starts across all three phases. Both heads start from the
same checkpoint and use the same frozen inputs, minibatch draws and MSE
update budget. Checkpoint selection uses separate maps and the corresponding
validation target for each head.

The soft-Z teacher compares the highest-prior first actions on shared,
action-equivalent belief particles. Both policies play greedily through the
rest of the current turn and the opponent's full reply. The teacher runs
deeper native MCTS from each live successor after the reply and uses exact
outcomes for terminal branches. Worlds interrupted by the rollout limit are
excluded for all candidates. The teacher averages each action's scores over
the remaining worlds, allocates root visits by PUCT over these action
values, and uses their visit-weighted mean as the soft-Z target. The training
set includes only positions for which both targets are available.
[Off-policy AlphaZero targets][off-policy]

The arena compares the two fitted heads and the zero-value control at
the same search budget (`--simulations 32`), with both player assignments
on each new map and one fixed raw opponent. The opponent's value inside the
search tree is held at `--opponent-value-scale`. Match deltas use only maps
whose two games reached a terminal result in every arm.

A separate ranking diagnostic picks two high-prior legal actions per
held-out position. It scores both actions after the reply in the same
belief worlds for every head. The reference runs repeated raw continuations
on the original map with matching random seeds for the two actions. It
reports preference accuracy on decisive mean returns and gap RMSE across
positions with paired terminal continuations. A predicted tie counts as a
miss when the reference prefers a move. Individual labels and continuations
are written to JSONL, with both comparisons in `summary.json`.

## Same-particle disagreement dataset

`evaluate.py --mode mcts-value-disagreements` disables and enables the
same apprentice value head at identical roots with identical accepted
particles. Roots are sampled from fixed raw games with seat swaps, across
all three phases. The diagnostic keeps only different first actions. Each
action gets paired continuations on the original map with the same frozen
MCTS (0,1) candidate and a sampled raw opponent. The opponent uses matching
random streams across the two branches.

The output directory stores native leaf packets and shared particles as
compressed NumPy archives, root action prefixes and visits in
`pairs.jsonl`, paired terminal results in `rollouts.jsonl`, and per-map
score changes in `summary.json`. The real nonterminal positions immediately
after the forced actions are saved separately for supervised use. Their
labels and perspective signs are in `labelled_pairs.jsonl`.
Rollout results belong to these continuations. The native leaf packet
currently exposes the tree and leaf IDs without the traversed action path,
so these results cannot serve as labels for individual simulated leaves.

## Fit a head from saved disagreement pairs

`train_value.py --position-source saved-mcts-pairs --saved-pairs DIRECTORY`
loads the real successor observations from that diagnostic. It verifies the
saved terminal labels against completed paired rollouts, checks the player
perspective and re-encodes each observation with the frozen input checkpoint.
The encoded features must agree with the saved features. Input settings for
map size, map type, tribes and visible history must match the dataset.

The head learns mean terminal results through MSE. An additional MSE term
compares the difference between both branch values in the original player's
perspective. Gaps are divided by two before squaring, keeping their scale
comparable with the individual value targets. The rest of the network is
frozen. Both branches and every position from a map stay in the same split.
By default 25% of map seeds are held out for checkpoint selection. The split
uses `--selection-seed` to break ties between map groups. It prioritizes
coverage of all available phases and pairs with different terminal returns
in both splits, then balances their counts. This assignment uses only
dataset metadata. The report lists any categories that cannot be covered
with the requested number of maps. The combined held-out loss selects the
checkpoint, including the initial head as a reference.

When a head update improves the selection loss, the selected head and a
zero-value control then play MCTS at
`--simulations` against the fixed raw opponent. Every new validation map is
played from both seats. The validation schedule is rejected if it overlaps
any source map from the saved diagnostic. Only complete map pairs common to
both arms contribute to their score difference.
If the initial head remains best, the default run finishes after fitting.
`--benchmark-unchanged-head` explicitly runs the arena in that case.

`config.json` records the exact pair IDs and map seeds in each split,
dataset checksums and the loss settings. `summary.json` reports the offline
RMSE, gap RMSE, preference accuracy, chosen update and paired arena results.
It reports `head_updated`, `benchmark_performed` and `beats_zero`
separately. A missing arena result leaves `beats_zero` null. A positive
point-score difference against zero is recorded without claiming training
improved upon the original checkpoint.

`metrics.jsonl` and `metrics.csv` contain full-training-set and
held-out value MSE, gap MSE, combined loss, RMSE and preference accuracy
before training and after every update. They also record minibatch loss,
gradient norm, learning rate and the best update so far. Console output
prints losses every `--fit-log-interval` updates, defaulting to 16. The
summary keeps the metrics for the selected head separate from those of the
last attempted update. The selected checkpoint is saved before the arena
starts.

## Offline loss and feature diagnostic

`scripts/diagnose_value_pairs.py` reuses the saved successors and the same
whole-map split to compare MSE, normalized gap MSE, and their weighted sum.
Every variant starts with identical head weights and uses identical phase
sampling and minibatch draws. It selects updates with the corresponding
held-out objective and reports the last attempted update separately from
the selected one. These three fits run offline. The script restores the
input network after collecting its measurements.

Before fitting, the diagnostic measures each pair's feature distance,
root-perspective predictions, target gap, gradient of the normalized gap,
and gradient of its squared error. It flags an informative pair with
identical features as an input collision only when both branches have the
same perspective sign. Opposite signs can produce a nonzero root gap from
identical own-perspective values.

The aggregate gradients follow the learner's phase-balanced objective.
Their norms, weighted ratio and cosine help distinguish a weak gap signal
from conflicting updates. Identical-input collisions, zero sensitivity and
tanh saturation are recorded separately. An input distance by itself does
not establish whether the head can learn the ranking.

The root `summary.json` compares training and held-out RMSE, gap RMSE and
preference accuracy for all losses. Each loss has its own
`metrics.jsonl`, `metrics.csv` and detailed `summary.json`.
`pair_diagnostics.jsonl` records individual feature and gradient
measurements before training and after each variant's last update.
Gap-only fitting diagnoses relative ordering. Its absolute value
calibration requires a separate check before use in search.

For a memorization check, add `--memorization-test`. It keeps the same
whole-map split, removes tied-return pairs from training and fits gap loss
on every remaining training pair in one full batch. Held-out examples never
enter this batch. The report records the effective pair IDs, highest
training preference accuracy and first update to reach a perfect training
ranking. Gradient diagnostics use the full-batch objective in this mode.
This check measures whether the head can learn these training rankings.
Transfer to held-out maps and value calibration remain separate measurements.

## Calibrated ranking fit

`scripts/train_ranked_value.py` fits only `value_head`. Use this after the
memorization diagnostic shows that the head can learn branch rankings but
gap-only updates damage absolute calibration. It accepts the same saved
successor datasets through `--saved-pairs`. Repeat that option to combine
additional collections. Pair IDs receive separate ranges, while all pairs
from the same map stay together in the train or selection split. The
ranking loss uses only pairs whose mean paired terminal returns differ.

The script also collects separate complete games for absolute value MSE.
By default the learner uses MCTS with zero leaf values, against the frozen
raw opponent. Sampling covers thirds of each completed game and both
seats. Truncated games supply no terminal labels. The train calibration
maps, calibration selection maps, source-pair maps and final arena maps are
disjoint, including any replacement maps attempted during collection.
`--calibration-policy raw` switches only this collection policy.

Complete collections are cached under `--calibration-cache` as compressed
NPZ files with encoded states, terminal targets, map seeds, player IDs and
phase labels. Their signatures include both checkpoints' fingerprints,
device and collection settings. Head-only reruns can reuse these data.
A cache with a currently reserved map is rejected. Incomplete collections
are reported and stop the run before fitting. They never become cached
terminal training sets.

The training objective is normalized root-perspective gap MSE plus
`calibration_weight * terminal_MSE`. Every weight in
`--calibration-weights`, default `0.01,0.05,0.2,1.0`, starts from the same
head and uses identical phase-balanced calibration minibatches. Informative
training pairs form a full batch. Policy, belief and both shared encoders
remain frozen.

Each update is checked against the original head on held-out maps. Admission
requires improved preference accuracy (or preservation of perfect accuracy),
lower gap RMSE, no absolute RMSE regression by default, absolute RMSE at
least as good as predicting zero, and no increase in absolute bias beyond
`--calibration-bias-regression`, default `0.02`. The RMSE tolerance defaults
to zero. Among admitted updates, selection prioritizes ranking accuracy,
then gap RMSE, absolute RMSE and absolute bias. This is checkpoint selection
on validation data. Its metrics are not an independent estimate of strength.

If no update qualifies, the run records the rejection reasons and skips
the arena. Otherwise it saves a separate checkpoint before playing MCTS
with the selected head, the original head, and zero leaf value against the
same raw opponent. Arena maps are fresh, each played from both seats. Only
complete map pairs shared by all three arms enter score differences.
`beats_zero` denotes a higher observed score for the selected head, without
claiming statistical significance or automatic champion promotion.

Each `calibration-WEIGHT` directory contains per-update `metrics.jsonl`,
`metrics.csv` and `summary.json`. The root config records source checksums,
map splits, checkpoint fingerprints, loss weights and gate settings. The
root summary separates the original, selected and last attempted metrics,
records cache paths, and reports arena validity and score differences.
The number of informative pairs is printed before collection. A small
number of informative selection pairs remains weak evidence even when
all offline gates pass. Add independently collected pair directories to
expand that part of the dataset.

## Collect confirmed new pairs before the calibrated fit

`train_ranked_value.py --collect-decisive-pairs 50 --collected-pairs-dir DIRECTORY`
adds a resumable collection before head fitting. It first loads the terminal
calibration data. Add `--require-calibration-cache` to require the existing
matching cache and fail immediately rather than collect those games again.
Keep its original input checkpoint, device, seeds, map settings, history,
game counts, positions per game and pool settings when reusing the cache.

Source maps exclude every map in the old pair datasets, both calibration
collections including attempted replacements, and the final arena. New
raw source games play from both seats. Probes cover opening, midgame and
endgame thirds. MCTS with value zero and one uses the same root particles
to propose competing actions. Real successor observations and the particles
are saved; this mode does not collect simulated-leaf training labels.

For each candidate, both forced actions receive paired MCTS(0,1) versus
sampled-raw continuations. The opponent RNG seed is identical between
branches for a repeat and distinct across repeats. Only completed pairs
count. The default schedule checks after 4, 8, 12 and 16 completed paired
repeats. Uncertain cases receive more repeats. An exact two-sided paired
sign test ignores tied repeats, using a per-look alpha of `(1-confidence)`
divided by the number of scheduled looks. The direction must also agree
with the mean score difference. Four identical nonzero differences do not
pass the default test. After eight all-tied repeats, the candidate is set
aside as having no observed difference; this does not establish equivalence.
The test screens repeatable preferences, not a precise confidence interval
for the mean gap or a claim of league strength. The look correction is per
candidate, not across the whole mined dataset. See the [NIST paired sign
test](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/signtest.htm)
and [exact binomial test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).

At most two confirmed pairs per source map are selected by default. Only
confirmed pairs with two nonterminal successor packets enter
`labelled_pairs.jsonl`. All candidates, paired scores, RNG seed tuples and
uncertainty decisions remain available in `pairs.jsonl`, `rollouts.jsonl`
and `decisions.json`. Source reservations and a pending-probe queue are
checkpointed in `collection-state.json`. Reusing the directory resumes
missing repeats without replaying completed ones. The directory verifies
checkpoint fingerprints and collection settings; the time allowance can
change between invocations. A partial trailing journal record from an
interrupted write is recovered, while earlier corruption is rejected.

The `50` target counts confirmed pairs. `--pair-collection-hours 8` bounds
collection at the next native batch boundary, so the final in-flight batch
may extend beyond that limit. Reaching 50 within the time allowance is not
guaranteed. `--minimum-collected-pairs 20` explicitly allows fitting a
smaller completed collection after a time or source-game limit. Without
that option the minimum equals the target. Below the minimum the run saves
progress and skips fitting. Offline gates and an optional arena follow
collection and take additional time.

When this collector is enabled, ranking training uses the new confirmed
dataset. Previously supplied `--saved-pairs` directories remain recorded
and reserve their maps. New pairs are split by whole map seed and checked
against every calibration and arena reservation again before fitting.
The original calibration and head-admission gates still apply.

## Pool candidates and retain uncertain mean outcomes

Use `--pair-collection-mode pooled` to change the collection schedule and
retain soft training targets. `--pair-candidate-pool 100` first scouts a
larger set of competing actions before starting new continuations. Scouting
uses at most `--pair-scout-fraction`, default 0.2, of this invocation's
collection allowance. It can finish sooner when the pool is full or source
games are exhausted. Existing pending probes are reused.

Screening takes `--pair-screen-rollouts 2` completed paired continuations
per candidate. Pairs with different numbers of saved attempts can share a
batch. Each position still receives the opponent RNG seed for its own
next repeat, duplicated between both branches. This fills the available
pair slots while resuming old work. Once enough usable pairs are available,
the remaining budget refines nonzero mean gaps that remain uncertain. It
does not spend extra repeats on all-tied cases. Refinement priorities
include gap size, a regularized estimate of its standard error, and the
number of completed repeats. This is a collection heuristic.

In this mode `--collect-decisive-pairs 50` targets 50 **usable** pairs.
`confirmed_pairs` separately retains the stricter sign-test classification.
A pair becomes usable when both nonterminal successor packets exist and
enough complete paired rollouts have been saved. Truncated or belief-failed
attempts never supply terminal targets. The selected states keep their
actual mean terminal outcomes, converted to each successor player's
perspective. Tied outcomes are useful for point-value MSE and gap equality.

Uncertainty weights are bounded between zero and one. Two fixed pseudo-
observations from the uniform joint distribution over paired win/draw/loss
outcomes regularize the moment estimate, preventing zero estimated
uncertainty from a short constant sample. Point and gap weights use
`n/(n+2)/(1+SE)` for their respective regularized standard errors. These
weights are heuristics, not confidence intervals. The loader recomputes
them from the complete rollout journal before accepting saved labels.

The head now learns weighted gap MSE plus
`--pair-value-loss-weight * weighted_pair_MSE` plus the existing weighted
terminal calibration MSE. The pair MSE weight defaults to one for this mode
and zero for the original confirmed-only mode. Losses average weighted
errors over a fixed sample count, so normalization does not cancel a low
confidence weight. Each training map receives equal total pair sampling
mass before applying the uncertainty weights. Whole-map selection splits,
the independent calibration guards and the final game arena are retained.
Offline acceptance metrics remain the unweighted measurements on the
fixed held-out dataset, reported separately from the weighted train losses.

To reuse a previous confirmed-only night, keep its checkpoint, rollout
settings and `--collected-pairs-dir`, and add the pooled options. The
collector permits this explicit scheduling upgrade and verifies the
original checkpoint fingerprints, device, environment and search settings.
It saves `legacy-config.json`, `legacy-summary.json` and
`legacy-decisions.json` before upgrading. The candidate and rollout journals
and successor observations are reused without repeating recorded attempts.
`weighted_pairs.jsonl` contains the new targets; the original
`labelled_pairs.jsonl` remains intact. Switching that directory back to the
old scheduling mode is rejected.

Keep the old thirty-pair dataset in `--saved-pairs` to reserve its maps; put
the resumable night directory in `--collected-pairs-dir`. Keep
`--require-calibration-cache` and the original calibration settings to reuse
its existing train and selection caches. New logs show `usable/target`,
`confirmed`, collection phase and `mean_pair_batch`. A minimum of 20 now
means 20 usable pairs, including reused pairs from the previous invocation.
The collection limit excludes head fitting and any later arena.

## Reconstruct an explicitly requested logged update

`scripts/reconstruct_ranked_value.py --source-run DIRECTORY --update 448
--calibration-weight 1.0 --model-out CHECKPOINT` replays one saved run's
initial value head through the requested optimizer update. It loads the
original model paths, training settings, split and collection parameters
from the run artifacts. Input paths and the calibration-cache directory
can be overridden, but checkpoint fingerprints and source dataset checksums
must match. The original run directory and input checkpoints are protected
against accidental overwrites.

No source games or branch continuations are collected. Both matching
terminal caches must already exist. The replay uses the same head
initialization, weighted targets, map balancing and seeded sequence of
calibration minibatches. Replaying the first 448 updates of the 1.0 arm is
equivalent to that prefix of the original sweep because every arm started
with a fresh optimizer and identical minibatches. The requested state is
captured explicitly, independently of the best update admitted by the
normal gates.

Before saving or starting games, the script compares the initial and
requested-update metrics against their original `metrics.jsonl` records.
The default absolute and relative tolerance is `0.0001`. A mismatch is
reported with the recorded and replayed values and skips the checkpoint
and arena. Different GPU kernels or software versions can affect numeric
reproduction; passing these checks is a metric-level verification rather
than proof of bitwise identity with an unsaved GPU checkpoint.

`gap_diagnostics` separates all, tied-target and informative-target gap
errors for training and held-out pairs, for the original head, requested
head and zero control. Tied means equal **mean returns of both branches**,
including both winning or both losing. It does not mean a drawn game or
proven action equivalence. Both predictions and targets use the original
player's perspective. `gap_pairs.jsonl` records each pair's target gap and
the three predicted gaps, map, phase, group and supervision weights.

After verification the requested checkpoint is saved as **diagnostic**,
with any original gate failures listed in the summary. This does not
change training admission or league promotion. MCTS then compares the
requested head, original head and zero against the same frozen raw
opponent on fresh seat-swapped map pairs. Original source, calibration,
replacement and even planned original arena seeds are reserved. A
24-game arena means 12 maps from both seats per arm, 72 games across the
three arms. Only complete map pairs common to all arms contribute to score
differences. The diagnostic checkpoint and offline report are saved before
these games so an arena interruption does not discard the recovered head.

## Attenuate the audited head using training maps only

`scripts/calibrate_value_scale.py --source-run TRAIN_RUN --audit-run AUDIT_RUN`
loads the verified checkpoint from the reconstruction audit and the original
frozen-encoder caches. It checks the source dataset hashes and cache hashes,
requires the checkpoint's non-value weights to equal the original network,
and compares its held-out metrics with the audit. It collects no games for
training and does not change any neural-network weights.

The script fits a nonnegative multiplier in `[0, --maximum-scale]`, with
the maximum limited to one. This is attenuation of the leaf value used by
MCTS. Positive scaling preserves the head's pairwise ordering; it changes
how strongly the values guide search relative to its priors. It should not
be interpreted as a newly trained or automatically calibrated probability
predictor. Scaled and unscaled outcome errors are both reported.

Candidate fitting objectives include gap-only MSE and joint gap plus pair
point MSE with each `--calibration-weights` terminal-MSE weight. The pair
point weight comes from the original run. Each objective has a closed-form
bounded scalar solution using map-balanced, uncertainty-weighted pairs and
map-balanced terminal labels. Weights stay outside fixed-count means.

`--folds 4` partitions **only the original training maps**, separately for
pairs and terminal calibration. All positions from a map stay in one fold,
and each training map is held out once. The original selection pairs and
selection calibration games do not fit scales or select objectives. The
common cross-validation criterion minimizes map-balanced, uncertainty-
weighted gap MSE, subject to the scaled terminal RMSE beating or matching
zero. Each candidate is judged with this same criterion, rather than its
own fitting loss. The report lists the scale range and standard deviation,
per-fold winning objectives, and pooled out-of-fold errors. If no method
improves upon the zero gap control, selection returns zero. Otherwise the
chosen objective is fitted once on all original training maps. The
original selection sets then evaluate this fixed choice without retuning.

This cross-validation checks **scalar calibration of a fixed head**. That
head was previously trained on these training maps. It is not an unbiased
cross-validation estimate of the original head training algorithm or of
playing strength. New game maps remain the independent measurement.

`gap_errors.jsonl` ranks each split's pairs by unscaled squared gap error.
The summary lists the largest errors and regressions relative to the
original head, including the fractions of total squared error contributed
by the worst pair and worst three pairs. Equal observed mean outcomes are
reported separately from nonzero gaps and do not establish equivalence.

`value_scale.json` saves the selected multiplier, checkpoint fingerprint,
objective and planned arena map seeds before any games start. The value
checkpoint remains unchanged. For a nonzero selection, MCTS compares only
that scale with zero against the original frozen raw opponent on new
seat-swapped map pairs, with the same opponent-tree value scale. Source,
calibration, original arena and reconstruction-audit arena maps are
reserved. A zero selection skips the redundant arena. `--offline-only`
runs all analysis without any game evaluation. With 24 games per arm, a
nonzero selection plans 48 games on 12 fresh maps.

## Offline comparison of branch-MSE weights

`scripts/compare_pair_mse.py --source-run TRAIN_RUN` reuses the original
saved pairs, whole-map split and terminal calibration caches. This is a
head-only training experiment, with no game collection or MCTS arena. A
missing cache is an error; source checkpoint fingerprints, source dataset
hashes and the original split must match. Run it on the same device as the
original collection, because device is part of the cache identity.

The objective is:

```
normalized_root_gap_MSE + pair_mse_weight * branch_MSE
                        + calibration_weight * terminal_cache_MSE
```

The gap coefficient is always one. `--pair-mse-weights 0,0.01,0.05,0.1,1.0`
changes only the branch-MSE coefficient. `--calibration-weight 1.0` stays
fixed across variants, so pair weight zero still trains on terminal-cache
outcomes. Confidence weights and map balancing follow the source run, with
fixed-count normalization. Both branch predictions and their targets are
converted to the same root-player perspective when calculating the gap.

Every variant starts from the **original source run's initial checkpoint**,
not the head reconstructed at update 448. Each gets a fresh AdamW and the
same phase-sampled calibration minibatches, learning rate, batch size and
seed. The original training and selection maps retain their roles; selection
positions supply no gradients. The default budget is 512 updates per weight.

The source run's branch-MSE weight must be included. It runs first, and its
metrics at update zero and `--control-update 448` are compared with the
source history. Failure stops the sweep before the remaining weights and
prevents saving a proposed candidate. The control step must exist in the
original history and fit inside the new update budget. This makes it
possible to attribute differences to the branch-MSE coefficient, rather
than a different initialization, dataset or optimizer schedule.

Each `pair-mse-WEIGHT/calibration-1.0` directory contains the complete
per-update JSONL and CSV curves, including training and selection ranking,
train branch/gap losses, selection value RMSE and bias. The parent summary
separates the final attempted update, the best held-out gap (including
update zero), and the best update admitted by the original calibration
gate. A best-gap checkpoint that breaks calibration remains explicitly
diagnostic. Small head-only snapshots are stored next to each variant's
summary. `gap_pairs.jsonl` and `gap_diagnostics` in the root summary compare
the original head, zero, each last update and each selected update, with
equal-mean and informative pairs separated.

If a variant passes the original common admission rules, the best admitted
candidate is saved as `candidate.pt` in the new run directory, or at a new
`--model-out` path. Otherwise no full-network candidate is saved. Admission
prioritizes ranking accuracy, gap RMSE, value RMSE and absolute bias; it does
not compare the variants' differently weighted training objectives. The
original network and checkpoints remain unchanged. Reused selection metrics
are development measurements: this experiment neither establishes a gain
over zero-value MCTS nor promotes a league champion. An independent paired
game benchmark remains necessary.

## Cache-versus-gap gradient diagnostic

`scripts/diagnose_value_gradients.py --source-run TRAIN_RUN
--memorization-run MEMORY_RUN --audit-run AUDIT_RUN` loads the same pinned
pairs and required terminal caches without collecting games or fitting a
new checkpoint. It measures the source run's initial head and, optionally,
the previously audited reconstructed head. The latter must have unchanged
non-value weights, matching audit metrics and matching calibration-cache
hashes.

The diagnostic computes the exact weighted, map-balanced gap and branch-MSE
gradients used in training. Terminal-cache gradients are measured both for
the **expected phase-balanced sampler** and for 32 actual minibatches from
the original sampler and seed. Uniform-cache MSE and its gradient norm are
reported separately: uniform positions and uniform phases need not produce
the same gradient. Only training maps supply gradients. Selection maps are
read solely for outcome measurements.

The report includes raw and weighted gradient norms, cache/gap norm ratios,
pairwise cosines, the fraction of sampled cache gradients opposing the gap,
and the projection of the combined gradient onto the gap gradient. A
negative `gap_cache_cosine` denotes opposing directions. A negative
`gap_dot_combined` predicts ascent of the training gap under an infinitesimal
SGD step. Neither alone proves that an AdamW update damages ranking.

For the first eight sampled batches, four isolated one-step probes compare
gap only, weighted cache only, gap plus cache, and the original full loss.
They use the original learning rate, a fresh AdamW with zero weight decay,
and the trainer's gradient-norm clip of one. Each starts from the same head
and is restored immediately. The report measures actual changes in training
gap loss, ranking, uniform and phase-balanced cache errors, and selection
errors. At the initial head, the first full-loss probe reproduces the first
training update. At update 448, these are fresh-optimizer probes, **not a
reconstruction of the next original update**: its optimizer moments were
not saved. With two heads and eight batches, there are 64 isolated probes.
Parameters and pre-existing gradients are restored; no trained model is
written.

The existing memorization run is read rather than repeated. Its recorded
checkpoint path and architecture must match, and its reported maximum
training accuracy and first perfect update must agree with the saved
per-update history. Its source dataset hashes are compared with the current
sources, and any difference is made explicit. The old test did not record
a parameter fingerprint, so this is historical evidence, not independent
verification of its original weights. In the recorded five-pair experiment,
training ranking first reached 100% at update 29; this does not establish
learnability of a different dataset or generalization to other maps.

`summary.json` stores the complete measurements and historical evidence;
`gradient_batches.jsonl` records sampled indices and gradient geometry;
`adamw_probes.jsonl` records every restored update and its measured changes.
Config, fingerprints, hashes and scalar CSV/JSONL metrics are retained for
reproducibility. Zero gradient norms produce explicit undefined ratios or
cosines rather than NaN. Missing caches or incompatible historical records
stop the diagnostic instead of launching collection or another memorization
training run.

## Projected cache gradients with a transactional AdamW guard

`scripts/compare_value_projection.py --source-run TRAIN_RUN` compares
ordinary summed losses with cache-gradient projection plus an actual-step
guard. It uses the pinned original pairs, split, checkpoint and required
terminal caches. It collects no games. Both variants start from the same
original head, use fresh AdamW optimizers and the same phase-balanced
minibatch draws, source learning rate and batch size. Gap, branch-MSE and
cache coefficients are identical in both arms. The default cache weight is
one; the branch weight is restored from the source run.

For weighted training gap gradient `g` and weighted cache gradient `c`, the
protected arm replaces `c` by `c - dot(g,c)/dot(g,g) * g` when their dot
product is negative. Otherwise it retains the cache gradient. A zero gap
gradient leaves the cache gradient unchanged. The branch-MSE gradient is
not projected. This is an asymmetric use of the projection idea from
[Gradient Surgery for Multi-Task Learning][pcgrad], prioritizing the gap,
rather than the full randomized symmetric PCGrad algorithm. Projection is
calculated in float64 on CPU and converted back to the parameter dtype and
device. Gradient clipping to a norm of one follows assembly.

Projection alone does not control AdamW's preconditioned update or stored
momentum. Before each proposed protected update, the guard snapshots the
head and the complete optimizer state. It accepts the actual step only if
the **full weighted training-pair gap** is finite and does not increase
beyond `1e-10 + 1e-6 * abs(gap_before)`. Selection data never participate in
this decision. Every rejected trial restores parameters, buffers, Adam
moments, step counters, parameter groups and prior gradients before retrying
the same gradient with a smaller LR.

The defaults allow six halvings after the original attempt, with a minimum
LR of `1e-7`. Accepted retries keep their reduced LR for subsequent proposed
updates. Exhausting all trials restores the entire pre-update state,
including its LR, and skips that update. Sixteen consecutively skipped
updates stop the protected arm with `guard_stalled`. Nonfinite trial
parameters or optimizer tensors are rejected, and exceptions during the
acceptance measurement restore the snapshot before propagating.

`--updates 512` is a budget of **proposed** updates per arm. Backtracking
costs extra forward passes, while skips reduce the applied-update count.
The reports therefore record elapsed time, proposed and applied steps,
skipped steps, rejected trials and projected steps. Applied steps and
held-out model admission are distinct: passing the training guard does not
establish calibration, generalization or better gameplay.

The ordinary-sum control runs first and reproduces the original history at
update zero and `--control-update 448`; a mismatch stops comparison before
the protected arm. Both arms use the original held-out calibration admission
rules and ranking-based checkpoint selection. Each arm saves diagnostic
`last-head.pt`, `best-gap-head.pt` and, when admitted, `admitted-head.pt`
snapshots bound to the original network fingerprint. A full `candidate.pt`
is written only if a head is admitted; it may come from either arm and is
not automatically promoted. Independent games remain necessary.

Per-update CSV/JSONL histories remain in each arm's `calibration-1.0`
directory. Its `events.jsonl` records every protected proposal and every
trial LR, gap measurement and rejection reason. The root `summary.json`
contains the full comparison; `gap_pairs.jsonl` separates the original,
zero, last, best-gap and admitted heads on training and selection pairs.
Input model weights and input files remain unchanged.

## Actual AdamW directions and a fresh-moment gap fallback

The protected update now records the **actual parameter displacement** for
each trial. `gap_dot_parameter_delta = dot(gap_gradient, theta_after -
theta_before)` is positive for a first-order gap-ascent step and negative
for descent. `gap_step_cosine` records the corresponding angle; a zero
gradient or zero displacement has no defined cosine. This differs from
the sign convention of `gap_dot_combined`, which describes a gradient that
would be *subtracted* by SGD. These are measured directions rather than a
deduction from the LR or raw projected gradient.

Add `--compare-fallback` to compare three arms: the ordinary source control,
the existing `projected_guarded` method, and `projected_gap_fallback`. All
start from identical source weights and use the same cached observations,
split and minibatch stream. `--projection-reference-run PREVIOUS_RUN`
additionally requires the existing projection to reproduce that run's
counts, final metrics and saved head before the fallback arm starts. Source
configs, guard settings, checkpoint fingerprints and cache hashes must
match. The original five-field guard schema is supported. A mismatch stops
the comparison rather than attributing a changed baseline to the fallback.

In the fallback arm, normal projection/backtracking runs first, unchanged.
If every trial is rejected, the code restores the pre-proposal weights and
full optimizer state, then retries **only the weighted gap gradient** with
fresh AdamW moments. The existing LR, betas, epsilon and other parameter-group
settings are retained. Each fallback backoff starts afresh from that same
snapshot, with fresh moments again, so rejected trials cannot accumulate
momentum. An accepted fallback retains its new optimizer state; this reset
is explicit and counted. The external applied-step count remains cumulative
even though AdamW's internal per-parameter counter restarts at one.

A fallback must lower the full training gap by more than its numerical
tolerance. A zero-motion or zero-gradient attempt cannot count as a reset
or successful update. It must also keep the **uniform full training-cache
MSE** below:

```
cache_before * (1 + fallback_calibration_relative_regression)
    + 1e-10 + relative_tolerance * abs(cache_before)
```

`--fallback-calibration-relative-regression 0` is the default: only
numerical tolerance is allowed. This bound is relative to the state before
the current proposal, not a held-out score. It governs fallback trials;
normal projected trials retain the previous gap-only acceptance rule.
Consequently it prevents an additional local calibration regression from
the fallback, but does not guarantee that calibration never regresses
throughout the whole run. Original held-out calibration and ranking
admission rules remain unchanged for every arm, and selection data never
guide update acceptance or LR retries.

If all fallback trials fail, original weights, gradients, moments, counters
and LR are restored. Exceptions and nonfinite updates receive the same
rollback. The rejection-patience limit still bounds a stalled run. A strict
calibration constraint can reject every fallback when its objectives
conflict; this is reported, not silently bypassed.

`events.jsonl` distinguishes projected and `gap_fallback` trials, with
directions, before/after gap and calibration, LR, limits and all rejection
reasons. CSV and summaries separately count fallback proposals, accepted
fallback steps, retained moment resets, calibration rejections and measured
AdamW ascent trials. Diagnostic head snapshots and the original candidate
admission rules are preserved. Missing caches do not launch new games.
The ordinary sum control does not measure trial directions: its direction
counts are undefined (`null`), rather than reporting unmeasured ascent as
zero. Protected arms report the number of measured displacements alongside
their ascent counts.

## Audit saved successor routing before changing training

`scripts/audit_value_routing.py --source-run TRAIN_RUN --head-run HEAD_RUN`
audits the saved successor pairs with no game collection, search or optimizer
steps. It compares the original head and the selected mode's saved
`last-head.pt` from the verified offline comparison. Source configs,
checkpoint fingerprints, cache hashes and recorded head metrics must match.
Both networks and all input files remain unchanged.

The audit counts physical changes from the root player separately from
model assignments: neither branch changes player, one does, or both do;
and both branches use apprentice, mixed networks, or both use opponent.
Saved native `agent_id` is authoritative for model routing. Any discrepancy
with the root-player convention is reported rather than overwritten.
`--require-agent-id` rejects old packets without that field. Without this
flag, missing assignments are inferred from root/player identity and
explicitly counted as inferred, not verified native routing.

Each opponent-routed state is encoded from its saved public packet using
the **opponent's own encoder and head**, including its observation, legal
actions and visible history. It is not evaluated using apprentice features
fed into the opponent head. The original apprentice-feature verification
is retained. Every successor NPZ receives a hash in the audit provenance.
Source-local pair IDs retain their recorded global offsets when several
collections are combined.

The prediction comparison includes:

- `legacy`: the unscaled apprentice head on both branches, as in the
  existing paired fitting objective;
- `routed`: apprentice on agent zero, frozen opponent on agent one, with
  independently recorded leaf scales (default one for each);
- `routed_zero_apprentice`: the `(0, opponent_scale)` control.

All branch scores and terminal targets are compared in the same root-player
perspective. Original actors, assigned models, turn changes, values, gaps,
preference changes and packet hashes are written to `routing_pairs.jsonl`.

Gradient comparisons use **training pairs and training cache only**. The
legacy and routed losses retain the same confidence weights, map balancing,
normalization and loss coefficients. Frozen opponent values are detached;
only apprentice-routed predictions pass gradients to its head. The expected
phase-balanced cache gradient and the same 32 sampled minibatches are
compared against both gap and branch-MSE gradients. Reports separate all
pairs, apprentice-only, mixed and opponent-only groups. Group contributions
retain the original full-pair denominator and weights, without renormalizing
small groups. Changed scalar scores and changed gradient flow are separate
measurements: routing can change the gradient even when both networks
currently predict identical values.

`no_apprentice_route` denotes no gradient path to this head at the saved
successors, including a zero apprentice leaf scale. `zero_aggregate_gradient`
instead denotes a zero weighted group gradient, which can result from
cancellation despite individual paths. Undefined cosines and ratios are
reported as null; neither status is evidence that a conflict was solved.
Selection positions provide prediction errors and rankings only.

This audits neural values at **real saved post-action observations**, not
the complete MCTS action Q-values or the observations of detached simulated
leaves. An opponent-only pair can still lead to apprentice decisions deeper
in search. A routing mismatch is a methodological finding to quantify; this
audit does not establish its effect on playing strength or alter the existing
training objective. Full measurements are retained in `summary.json`, with
config, source hashes and CSV/JSONL scalar metrics.

## Train policy-consistent values with the native model router

`scripts/train_routed_value.py` is the replacement experiment after the
routing audit. It changes neither architecture nor policy, belief, encoder,
flat training defaults or MCTS routing. It optimizes only `value_head`.
Its input `--source-run` is a historical **legacy** ranked-value run: the
script verifies its initial checkpoints, pair hashes, whole-map split and
both required terminal caches before collecting anything. Those caches must
come from fixed MCTS@32 `(0,1)` against the greedy raw opponent. A missing
cache fails immediately. `--model-out` must be a new, distinct path.

Fresh successor pairs go into a distinct resumable `--pairs-dir`. The source
games now use the same frozen MCTS `(0,1)` policy as the calibration games;
both forced-action branches continue with that policy and a greedy raw
opponent. In this experiment `temperature=0` disables opponent sampling.
This removes the previous mix of greedy-opponent calibration and
sampled-opponent branch targets. Old journals keep their original policy
identity and cannot be silently resumed with changed labels. Historical
sampled-raw pairs are reservations and a preflight control, not new training
labels. The collection retains phase coverage, swapped seats, uncertainty
weights, incomplete-rollout exclusions, bounded time/retries and whole-map
splits. Its source-game, candidate and time budgets bound collection; the
requested number of pairs or informative labels is not guaranteed.

The default screen gives each usable candidate at least four complete
paired repeats; subsequent repeats use the existing bounded refinement
schedule. All usable weighted pairs, including equal observed means, remain
in the dataset. Equal sample means are not proof that the moves are equally
good. Fitting requires at least 24 nonzero-mean training pairs and 12
selection pairs by default. Insufficient evidence is reported explicitly,
without an expensive arena or a misleading accepted checkpoint.

For each saved branch, native `agent_id` selects the apprentice head or the
frozen opponent. Opponent states use that network's own encoder. Both scores
are converted to the original player's perspective for the difference loss.
The loss is the confidence/map-weighted branch MSE plus difference loss and
phase-sampled terminal-cache MSE. The cache coefficients default to
`0.01,0.05,0.2,1.0`. Opponent values are detached constants. Only branches
actually assigned to the apprentice can update it; cache and pair selection
positions never supply training gradients. No gradient projection, guard
fallback or encoder unfreezing is enabled by this experiment.

All variants start from the same source head and use identical cached
minibatch draws. Existing strict ranking and calibration admission remains:
ranking accuracy and difference error must improve under the routed
objective without violating calibration RMSE, zero-baseline or bias checks.
Reports and CSV/JSONL histories separate apprentice-only, mixed and
opponent-only ranking errors. Diagnostic `last-head.pt` snapshots record the
update, routed metrics and initial network fingerprint. These are diagnostic
artifacts, not automatically deployed models. Legacy reconstruction scripts
explicitly reject the new routed objective rather than replaying another
loss under the same name.

An offline-admitted head is saved as `diagnostic-candidate.pt` and evaluated
against both the original head and apprentice `value=0`. All three MCTS@32
arms keep opponent value scale one, use the same fresh maps with swapped
seats, and share the original policy/encoder/belief. The validation seed is
drawn afresh on each invocation unless specified; its resolved value is
recorded in `args.json`. Planned and replacement source/calibration maps and
historical arena reservations are excluded from collection/evaluation as
appropriate. An interrupted collection can resume from the same pair
directory; `--collect-pairs 0` skips collection for an existing compatible
dataset, but still performs the arena if a candidate is admitted.

If a source MCTS batch is invalid, the collector retries each of its original
maps once as a separate swapped-seat pair, while time remains. Seeds are not
replaced and callbacks from the invalid batch are discarded. Persistently
invalid pairs are skipped and remain reserved; successful isolated pairs
retain their own prefixes and root fingerprints. `source-evaluations.jsonl`
records validity, map seeds, the actual invalid reason and available belief
diagnostics (including moves). Retry-game and failed-map counters are separate
from the budget of original source games. Unexpected exceptions and a valid
evaluation returning the wrong map schedule still stop collection.

`--resume-rollouts-only` labels the candidates already saved in `--pairs-dir`
without source games or new probes. It preserves the collection identity,
saved particles and per-pair repeat numbering, so it can resume a scouting
run that stopped before its first rollout. The collection time budget applies
to this invocation. Once saved candidates are exhausted, the trainer can fit
even below the original collection target, provided the unchanged minimum
pair and informative-label requirements are met. This flag does not mean
zero new games: the forced-action continuations and the eventual arena are
still played. It does not relax any quality gate.

Collection progress reports `informative` from the labels already obtained,
even before the minimum fitting budget is reached. `collected_pair_counts`
records usable labels, nonzero observed differences, covered maps and the
remaining minimum budget. Train/selection `informative_pairs` is `null`
until the whole-map split is actually computed; an early collection exit
must not present unmeasured split counts as zero. `confirmed=0` is the
separate sign-test result, not a count of all usable or nonzero-mean pairs.

`--model-out` is written only after the **entire planned arena** completes
and shows evidence against both controls. Each map's two seats form one
observation. Admission requires a positive mean score delta and a one-sided
paired sign-test p-value at most 0.025 against each control (two comparisons),
with at least 16 complete maps by default. Missing maps, tied results and
small tests do not establish improvement. Failed candidates remain in the
run directory with their metrics; the requested output is left unwritten.
`ready_for_mcts` records this experimental gameplay decision, not a league
promotion or a new belief-calibration marker.

This experiment still trains on **real successor observations**, not on
simulated particles' leaf observations. It fixes objective routing and
reference-policy consistency, but does not establish that the frozen
encoder represents every useful value feature or eliminate all possible
gradient conflicts. A retained model must demonstrate its effect in the
fresh gameplay comparison; lower offline RMSE alone is insufficient.

## If turn-aligned terminal targets are too noisy

Build a small second dataset from the positions where the proposed search
actually queries value. For each selected position, continue the same
simulated state with frozen, observation-conditioned policies and average
terminal outcomes across matched continuations. Use those averages as
additional MSE targets alongside the real-game data. Save the state
provenance, policy versions, number of completed continuations and how the
hidden worlds were sampled. Start with a small pilot and measure the fraction
of competing branches with distinguishable returns before collecting more.
An outcome from the real game cannot label an unrelated hypothetical leaf.
The current regression diagnostics can replay real positions from the seed
and action prefix. Simulated-leaf training also needs a verified way to
restore or continue detached simulated states in PolyEnv.

Another experiment is a separate full-state teacher for offline
supervision. Condition it on the information needed to evaluate the fixed
continuation policies and test it against terminal returns on held-out maps
before distillation. A student that only sees the player view can learn the
conditional average of teacher targets over plausible worlds. Keep the
teacher out of deployed search. Baisero and Amato explain why a state-only
critic can bias policy-gradient estimates for history-dependent policies. Comparisons of
AlphaZero targets also show that target choice depends on which policy
collects the games and which policy makes the final decisions. [Asymmetric critics][asymmetric] · [Off-policy AlphaZero targets][off-policy]

## Sources

- Yu et al. (2020), *Gradient Surgery for Multi-Task Learning*. Gradient
  projection for conflicting task objectives. The asymmetric projection
  and transactional training-gap guard above are a separate experimental
  adaptation, without a claim of improved MCTS strength. [Paper][pcgrad]
- Silver et al. (2017), AlphaZero preprint. Terminal targets for learned
  values and the policy–value split in search. [Paper][alphazero]
- Pérez-Liébana et al. (2020), *Tribes: A New Turn-Based Strategy Game for AI
  Research*. Polytopia-like turn structure and planning benchmarks. [Paper][tribes]
- Zhang (2026), *Building a Superhuman Polytopia Bot Using Self-Play RL*.
  First-hand account of fitting a fogged critic and evaluating complete
  opponent replies. [Engineering report][zhang] · [Code repository][zhang-code]
- Cowling, Powley and Whitehouse (2012), *Information Set Monte Carlo Tree
  Search*. Search with hidden information. [Paper][ismcts]
- Brown et al. (2020), ReBeL, and Schmid et al. (2023), Student of Games.
  Value and search conditioned on public beliefs. [ReBeL][rebel] · [Student of Games][sog]
- Willemsen, Baier and Kaisers (2022), *Value Targets in Off-Policy
  AlphaZero*. Comparing targets for different continuation policies.
  [Paper][off-policy]
- Baisero and Amato (2022), *Unbiased Asymmetric Reinforcement Learning
  under Partial Observability*. Limits of full-state critics under
  partial observability. [Paper][asymmetric]

[alphazero]: https://arxiv.org/abs/1712.01815
[pcgrad]: https://arxiv.org/abs/2001.06782
[tribes]: https://gaigresearch.github.io/afm/papers/TribesAIIDE20.pdf
[zhang]: https://thekevinz.com/blog/polytopia-bot
[zhang-code]: https://github.com/kevinz917/superhuman-polytopia-bot
[ismcts]: https://eprints.whiterose.ac.uk/id/eprint/75048/
[rebel]: https://arxiv.org/abs/2007.13544
[sog]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10651118/
[off-policy]: https://research.tue.nl/files/217685150/s00521_021_05928_5.pdf
[asymmetric]: https://arxiv.org/abs/2105.11674
