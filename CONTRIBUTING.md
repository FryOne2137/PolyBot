# Contributing

Use the [installation guide](docs/installation.md) and install
`requirements-dev.txt`. Scripts run from the repository; tests resolve `src/`
through `pyproject.toml`.

```bash
python scripts/check_environment.py --smoke
python -m pytest
python scripts/check_docs.py
python scripts/audit_public.py
```

Keep flat, belief and MCTS stage semantics separate. Add regression coverage for
changes to sample age, masking, hidden-state boundaries, checkpoint compatibility,
replay provenance and native buffer contracts. Avoid tests that require private
checkpoints, large datasets or a GPU. CUDA-specific tests may skip on CPU.

Performance claims need a stated workload and target hardware. Playing-strength
claims need opponents, seeds, completed/truncated game counts, move limits and
raw results. A lower training loss is not sufficient evidence.

## Documentation and figures

Update the relevant guide when CLI options, training modes or architecture change.
Keep examples relative to the repo and make their initialization requirements
explicit. Do not add personal paths or private checkpoint names.

Edit `scripts/render_diagrams.py`, then regenerate the documentation figures:

```bash
python scripts/render_diagrams.py --output-dir docs/assets
python scripts/check_docs.py
```

SVG is the scalable source output; PNG is the portable README preview. Generation
validates text containment and all figures should also be reviewed visually.
Renderer versions can change output bytes; cross-platform CI checks structure
and links rather than assuming pixel-identical antialiasing.

The detailed model diagram has its own generator, `scripts/render_network.py`.
Both generators write to `docs/assets/`; keep generated images out of the
repository root.

## Pull requests

Describe the concrete problem, changed behavior, tests and any compatibility
limits. Do not include models, runs, replay/database files, IDE state or
credentials. Keep sensitive security reports out of public issues; see
[SECURITY.md](SECURITY.md).
