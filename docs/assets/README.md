# Generated figures

Definitions and layout validation live in
[render_diagrams.py](../../scripts/render_diagrams.py).

- [Architecture](architecture.svg): public inputs, the shared model and separate
  flat/search decision paths.
- [Training](training.svg): independent flat, belief and MCTS jobs.
- [Belief-MCTS](belief_mcts.svg): detached hypotheses and the supervision boundary.

Each figure has a PNG preview and a scalable SVG. `diagrams.json` records the
generator fingerprint. Figures are kept here rather than copied to the repository
root.

Regenerate from the repository root:

```bash
python scripts/render_diagrams.py --output-dir docs/assets
python scripts/check_docs.py
```

The figures use repository-owned shapes and text, not game artwork. Different
renderer versions may change antialiasing; source structure and legibility are
the validation criteria.

## Detailed model diagram

[model-network.pdf](model-network.pdf) shows the default network dimensions.
The [PNG](model-network.png), [SVG](model-network.svg) and
[parameter manifest](model-network.json) come from
[render_network.py](../../scripts/render_network.py):

```bash
python scripts/render_network.py
```

## Gameplay preview

[self-play-demo.webp](self-play-demo.webp) is the animated README preview of a
gameplay replay in the PolyEnv viewer. It preserves the full two-minute recording
at 8 frames per second, with the outer margins cropped and the local file path
hidden. It loops automatically and is maintained separately from the generated
architecture figures.
