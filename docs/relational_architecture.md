# Relational policy/value architecture

The explicit variant `ModelConfig(policy_architecture="relational")` is one
network with a shared map/history encoder. The dedicated initializer defaults
to it; generic `ModelConfig` still defaults to `factorized` for compatibility.

![Model data flow](assets/architecture.png)

## Joint action scoring

The older factorized scorer sums separate type, source, target, city and option
scores. With a fixed map and equal options, its preference between two targets
does not directly depend on which source is selected.

The relational scorer combines type, semantic options, economic features,
global context, source/target/city states and relative geometry. Elementwise
source–target and query–endpoint interactions feed a shared nonlinear scorer.
It can represent source-dependent destination preferences and action-type
dependent use of a tile. A regression test fits a preference pattern that an
independent additive source/target sum cannot fit.

Action features use a narrow interaction space, default width 64 for a
192-wide encoder. Tile projections are computed once. Public city and unit IDs
are lookup keys resolved by sorted pointers, not learned numeric features.
Unit-only actions can recover their source tile through a public unit pointer.

There is no action-by-map attention matrix or attention between candidates.
At fixed board size the per-action computation scales linearly with action
count; pointer lookup adds a logarithmic board-size factor.

## Spatial state readout

Mean, maximum and learned attention pooling over revealed tile states supplement
the global token before policy/value prediction. Empty visibility yields zero
pooled features with finite gradients. The extra head-only critic fit still
detaches this representation.

The relational map encoder omits runtime unit, city and origin-ID embeddings.
History unit IDs become presence indicators. The legacy scaled pending-city
ID is omitted from global input; city-linked legal actions can name the
relevant public tile. Positions, types, ownership and other semantic features
remain available. Tests verify policy/value invariance under consistent ID
renaming.

Flat inference skips hidden-map reconstruction. The existing structured belief
branch remains available for its dedicated training stage and optional MCTS.
This is not a learned latent dynamics model.

## New experiments and old checkpoints

```bash
python scripts/init_model.py --architecture relational --seed 2137 --d-model 192 --layers 8 --heads 8 --action-interaction-dim 64 --model-out models/demo/initial.pt --project runs/demo --name initialization
```

This creates recorded random weights; it does not train. It refuses to overwrite
an existing model. Follow the [training guide](training.md) for bootstrap,
flat self-play, calibration and optional search.

Loading an existing attention/factorized checkpoint preserves its saved
architecture. There is no silent reinterpretation of old weights. Use separate
checkpoint directories and compatible league populations for the new variant.
The standalone evaluator can compare differently configured models.

## Throughput benchmark

```bash
python scripts/benchmark_architecture.py --device cuda --batch-size 8 --map-size 11 --actions 32,256,1024 --history 32 --repeats 30 --warmup 5 --training --output runs/demo/architecture-benchmark.json
```

This measures synthetic forward and optional forward/backward/AdamW workloads,
with identical tensor sizes and encoder dimensions. It excludes engine,
rollout and validation time.

A local CPU run used width 192, eight layers, rank 64, 11×11 maps, history 32,
batch four, one thread and 12 repeats after three warmups:

| Actions | Factorized forward | Relational forward | Factorized training step | Relational training step |
| --- | ---: | ---: | ---: | ---: |
| 32 | 19.84 ms | 21.33 ms | 64.86 ms | 63.46 ms |
| 256 | 23.10 ms | 24.14 ms | 65.41 ms | 69.62 ms |
| 1024 | 32.61 ms | 24.83 ms | 80.96 ms | 74.38 ms |

Raw metadata/results: [relational_cpu.json](benchmarks/relational_cpu.json).
This short measurement does not predict CUDA performance or league wall time.

At these dimensions the relational model has 7,462,561 parameters versus
7,804,786 for the factorized model: removing runtime-ID embeddings offsets the
additional layers. Interaction width 32 or 128 provides separate capacity/cost
experiments. Greater expressiveness is not evidence of stronger play without
multi-seed matches.
