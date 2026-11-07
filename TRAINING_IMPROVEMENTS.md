# Training documentation

This file is retained as a compatibility pointer. Historical recovery notes and
commands have been replaced by guides for the current entry points.

- [Training modes and commands](docs/training.md)
- [Flat PPO/PPG improvements](docs/flat_improvements.md)
- [Fresh rollouts and terminal supervision](docs/flat_rollouts.md)
- [Relational architecture](docs/relational_architecture.md)
- [Reproducibility and limitations](docs/reproducibility.md)

The dedicated belief trainer uses league-style arguments such as
`--champion-in`; old `--positions ... --model-out ...` examples do not apply
to `scripts/train_belief.py`.
