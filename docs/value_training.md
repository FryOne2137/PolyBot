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
[tribes]: https://gaigresearch.github.io/afm/papers/TribesAIIDE20.pdf
[zhang]: https://thekevinz.com/blog/polytopia-bot
[zhang-code]: https://github.com/kevinz917/superhuman-polytopia-bot
[ismcts]: https://eprints.whiterose.ac.uk/id/eprint/75048/
[rebel]: https://arxiv.org/abs/2007.13544
[sog]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10651118/
[off-policy]: https://research.tue.nl/files/217685150/s00521_021_05928_5.pdf
[asymmetric]: https://arxiv.org/abs/2105.11674
