# Generated figures

Definitions and layout validation live in
[render_diagrams.py](../../scripts/render_diagrams.py).

- [Architecture](architecture.svg): public inputs, the shared model and separate
  flat/search decision paths.
- [Training](training.svg): independent flat, belief and MCTS jobs.
- [Belief-MCTS](belief_mcts.svg): detached hypotheses and the supervision boundary.

Each figure has a PNG preview and a scalable SVG. `diagrams.json` records the
generator fingerprint. Root-level historical figure names are compatibility
copies; do not edit them independently.

Regenerate from the repository root:

```bash
python scripts/render_diagrams.py --output-dir docs/assets --compatibility-copies
python scripts/check_docs.py
```

The figures use repository-owned shapes and text, not game artwork. Different
renderer versions may change antialiasing; source structure and legibility are
the validation criteria.
