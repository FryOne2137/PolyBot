"""Render the public architecture figures from one editable, validated source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
ALIASES = {
    "bot_architecture_diagram.svg": "architecture.svg",
    "polybot_architecture.png": "architecture.png",
    "polybot_model_architecture.svg": "architecture.svg",
    "polybot_model_architecture.png": "architecture.png",
    "polybot_architecture_paper.svg": "belief_mcts.svg",
    "polybot_architecture_paper.png": "belief_mcts.png",
}
COLORS = {
    "blue": ("#edf4ff", "#2457a7"), "green": ("#edf8f2", "#18724b"),
    "orange": ("#fff5e8", "#a85b0e"), "purple": ("#f5efff", "#7341a6"),
    "gray": ("#ffffff", "#526479"), "red": ("#fff0f0", "#a43131"),
}


def canvas(title, subtitle):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(18, 10), dpi=150)
    fig.patch.set_facecolor("#f7f9fc")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, 18), ylim=(0, 10))
    ax.axis("off")
    ax.text(0.5, 9.45, title, fontsize=24, color="#172f45", weight="bold", va="center")
    ax.text(0.5, 8.93, subtitle, fontsize=11.5, color="#526479", va="center")
    return fig, ax, []


def card(ax, bounds, x, y, width, height, title, lines, color="gray", body_size=11):
    from matplotlib.patches import FancyBboxPatch
    fill, edge = COLORS[color]
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                              boxstyle="round,pad=0.025,rounding_size=0.12",
                              linewidth=1.3, edgecolor=edge, facecolor=fill))
    title_artist = ax.text(x + 0.21, y + height - 0.27, title, fontsize=11.5,
                           color=edge, weight="bold", va="center")
    bounds.append((title_artist, (x + 0.10, y + 0.08, x + width - 0.10, y + height - 0.08)))
    text = "\n".join(lines)
    body = ax.text(x + 0.21, y + height - 0.63, text, fontsize=body_size,
                   color="#24394d", va="top", linespacing=1.55)
    bounds.append((body, (x + 0.10, y + 0.08, x + width - 0.10, y + height - 0.43)))


def arrow(ax, points, label=None, at=None, color="#66788a", dashed=False):
    from matplotlib.path import Path as DrawingPath
    from matplotlib.patches import FancyArrowPatch
    path = DrawingPath(points, [DrawingPath.MOVETO, *([DrawingPath.LINETO] * (len(points) - 1))])
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=14,
                                linewidth=1.5, color=color, linestyle="--" if dashed else "-"))
    if label:
        ax.text(*at, label, fontsize=9.5, color=color, ha="center", va="center",
                bbox={"facecolor": "#f7f9fc", "edgecolor": "none", "pad": 2})


def footer(ax, text):
    ax.text(0.5, 0.48, text, fontsize=10.5, color="#526479", va="center")


def architecture():
    fig, ax, bounds = canvas(
        "PolyBot · one shared network, two decision paths",
        "Public observations and legal actions in; policy, value and optional belief out. Architecture is selected by the checkpoint.",
    )
    card(ax, bounds, .5, 4.2, 3.3, 3.5, "PUBLIC INPUT", [
        "Player-view map: T × 23",
        "Global features + event summary",
        "Visible history: H × 12",
        "Current legal candidates only",
        "Action semantics, costs, pointers",
    ], "blue")
    card(ax, bounds, 4.5, 4.2, 4, 3.5, "SHARED MAP / HISTORY ENCODER", [
        "Semantic embeddings + 2-D positions",
        "History Transformer → global context",
        "L-layer map Transformer",
        "Contextual tile states + global token",
        "Relational variant: spatial readout",
        "Mean / max / learned visible pooling",
    ], "purple")
    card(ax, bounds, 9.5, 6.3, 3.8, 1.9, "POLICY", [
        "Type + options + global context",
        "Source–target / query–tile interactions",
        "One masked logit per legal action",
    ], "green", 10.7)
    card(ax, bounds, 9.5, 4.2, 3.8, 1.6, "VALUE", [
        "Spatial state → scalar in [−1, 1]",
        "Game outcomes supervise the critic",
    ], "orange")
    card(ax, bounds, 4.5, 1.4, 4, 2.0, "OPTIONAL BELIEF DECODER", [
        "Public legal-action evidence",
        "Private spatial refinement",
        "Structured hidden-world hypotheses",
    ], "purple")
    card(ax, bounds, 14.3, 6.3, 3.2, 1.9, "FLAT ACTION SELECTION", [
        "Policy → real legal action",
        "Native VectorGameEnv step",
        "No particles or tree required",
    ], "green", 10.7)
    card(ax, bounds, 14.3, 2.2, 3.2, 3.2, "OPTIONAL BELIEF-MCTS", [
        "Validate detached worlds",
        "Policy/value guide C++ PUCT",
        "Native SelfPlayPool trees",
        "Visits become policy labels",
        "Separate search/readiness gates",
    ], "orange", 10.7)
    arrow(ax, [(3.8, 6.0), (4.5, 6.0)])
    arrow(ax, [(8.5, 7.1), (9.5, 7.1)])
    arrow(ax, [(8.5, 5.0), (9.5, 5.0)])
    arrow(ax, [(6.5, 4.2), (6.5, 3.4)], "tile states", (7.2, 3.8))
    arrow(ax, [(3.8, 7.55), (4.05, 7.55), (4.05, 8.5), (11.4, 8.5), (11.4, 8.2)],
          "legal action features", (8.9, 8.5), COLORS["blue"][1])
    arrow(ax, [(2.15, 4.2), (2.15, 2.4), (4.5, 2.4)], "legal-action evidence", (2.6, 3.55), COLORS["blue"][1])
    arrow(ax, [(13.3, 7.2), (14.3, 7.2)])
    arrow(ax, [(13.3, 6.7), (13.8, 6.7), (13.8, 5.1), (14.3, 5.1)])
    arrow(ax, [(13.3, 4.7), (14.3, 4.7)])
    arrow(ax, [(8.5, 2.5), (14.3, 2.5)], "detached, action-consistent worlds", (11.3, 2.75))
    footer(ax, "Hidden ground truth is training supervision only; it is not an actor/value observation or an oracle world for search.")
    return fig, ax, bounds


def training():
    fig, ax, bounds = canvas(
        "Three explicit training jobs",
        "Shared checkpoint format; separate objectives, collectors and readiness checks. A completed run does not imply a promotion.",
    )
    columns = [
        (.5, "01  FLAT · POLICY / VALUE", "green", [
            ("PUBLIC SELF-PLAY", ["Freeze behavior weights per fragment", "Native VectorGameEnv collection"]),
            ("OPTIMIZE", ["Fresh PPO/GAE; scheduled PPG", "Optional terminal head-only value fit"]),
            ("PROTECT WORKING / BEST", ["Fixed development benchmark", "Paired regression confirmation"]),
            ("MEASURE AND CONTINUE", ["Independent promotion games", "Main / archive / PFSP for later collection"]),
        ]),
        (6.5, "02  BELIEF · HIDDEN MAP", "purple", [
            ("COLLECT SUPERVISION", ["Public observations and action evidence", "Full-map semantics are training labels"]),
            ("FREEZE THE ACTOR", ["Keep shared encoder and policy fixed", "Train belief-specific decoder layers"]),
            ("CHECK WORLD PROPOSALS", ["Compare public legal action sets", "Use discrepancies to focus map losses"]),
            ("VALIDATE", ["Held-out mismatch and semantic metrics", "Save ready apprentice or roll back"]),
        ]),
        (12.5, "03  MCTS · POLICY / VALUE", "orange", [
            ("REQUIRE A USEFUL TEACHER", ["Validated belief + raw/search preflight", "Same maps and frozen opponent"]),
            ("FROZEN SEARCH COLLECTION", ["Detached worlds; batched neural leaves", "C++ SelfPlayPool select / simulate / backup"]),
            ("LEARN SEARCH TARGETS", ["Visits + outcomes / configured bootstraps", "Bounded replay and optimization blocks"]),
            ("GATE", ["Search quality and raw-policy absorption", "Independent promotion or rollback"]),
        ]),
    ]
    for x, title, color, cards in columns:
        ax.text(x, 8.25, title, fontsize=12.2, color=COLORS[color][1], weight="bold")
        for index, (heading, lines) in enumerate(cards):
            y = 6.7 - 1.6 * index
            card(ax, bounds, x, y, 5, 1.25, heading, lines, color, 11)
            if index < 3:
                arrow(ax, [(x + 2.5, y), (x + 2.5, y - .35)])
    footer(ax, "Keep evaluation maps out of training replay. Invalid measurement, weak search and failed model promotion are different outcomes.")
    return fig, ax, bounds


def belief_mcts():
    fig, ax, bounds = canvas(
        "Belief-guided search and the supervision boundary",
        "The engine owns the hidden state. Search receives accepted detached hypotheses; PolyBot owns all neural losses.",
    )
    card(ax, bounds, .5, 6.2, 3.7, 1.6, "PUBLIC PLAYER VIEW", [
        "Visible map + history + global features",
        "Current public legal action list",
    ], "blue", 10.7)
    card(ax, bounds, 5, 6.2, 4.5, 1.6, "NEURAL MODEL", [
        "Policy priors and value estimates",
        "Belief distributions → sampled worlds",
    ], "purple")
    card(ax, bounds, 5, 4, 4.5, 1.6, "VALIDATE AND MATERIALIZE", [
        "Preserve observed tiles and public actions",
        "Reject / retry inconsistent hypotheses",
    ], "purple", 10.7)
    card(ax, bounds, 10.3, 4, 3.1, 1.6, "NATIVE SEARCH", [
        "Detached-world C++ PUCT",
        "Batched policy/value leaves",
    ], "orange", 10.7)
    card(ax, bounds, 14.3, 4, 3.2, 1.6, "ROOT DECISION", [
        "Accepted visit distribution",
        "Select a real legal action",
    ], "green", 10.7)
    card(ax, bounds, 14.3, 6.2, 3.2, 1.6, "REAL GAME STEP", [
        "Execute in the actual engine",
        "Observe public view / result",
    ], "green", 10.7)
    card(ax, bounds, .5, 1.2, 3.7, 2.0, "HIDDEN GROUND TRUTH", [
        "Training-only semantic labels",
        "Never an inference observation",
        "Never the oracle search world",
    ], "red", 10.7)
    card(ax, bounds, 5, 1.2, 8.4, 2.0, "POLYBOT OPTIMIZATION", [
        "Policy: accepted MCTS visits; value: outcomes / configured bootstraps",
        "Belief: hidden-map labels plus discrepancy-guided supervision masks",
        "No gradients or neural loss are computed inside PolyEnv",
    ], "gray", 11)
    arrow(ax, [(4.2, 7), (5, 7)])
    arrow(ax, [(7.2, 6.2), (7.2, 5.6)])
    arrow(ax, [(4.2, 6.5), (4.6, 6.5), (4.6, 4.8), (5, 4.8)], "public constraints", (3.65, 5.2), COLORS["blue"][1])
    arrow(ax, [(9.5, 4.8), (10.3, 4.8)])
    arrow(ax, [(9.5, 7), (11.85, 7), (11.85, 5.6)], "policy / value", (11.3, 7.25))
    arrow(ax, [(13.4, 4.8), (14.3, 4.8)])
    arrow(ax, [(15.9, 5.6), (15.9, 6.2)])
    arrow(ax, [(15.9, 7.8), (15.9, 8.4), (2.3, 8.4), (2.3, 7.8)], "next public observation", (9, 8.4))
    arrow(ax, [(11.85, 4), (11.85, 3.2)], "visits", (12.4, 3.6))
    arrow(ax, [(4.2, 2.2), (5, 2.2)], color=COLORS["red"][1], dashed=True)
    arrow(ax, [(17.5, 6.8), (17.8, 6.8), (17.8, 2.2), (13.4, 2.2)],
          "game outcomes", (15.7, 2.45))
    footer(ax, "Flat policy/value inference uses the direct path and skips particles and search. Belief readiness alone does not establish playing strength.")
    return fig, ax, bounds


def validate_text(fig, ax, bounds):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = ax.transData.inverted()
    for artist, (left, bottom, right, top) in bounds:
        box = artist.get_window_extent(renderer).transformed(inverse)
        if box.x0 < left or box.x1 > right or box.y0 < bottom or box.y1 > top:
            raise ValueError(f"diagram text does not fit its card: {artist.get_text()!r}")


def render(output_dir: Path, compatibility_copies: bool):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                                 "svg.hashsalt": "polybot-public-diagrams-v1"})
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "generator": "scripts/render_diagrams.py",
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "figures": {}, "compatibility_copies": ALIASES if compatibility_copies else {}}
    for name, builder in (("architecture", architecture), ("training", training), ("belief_mcts", belief_mcts)):
        fig, ax, bounds = builder()
        validate_text(fig, ax, bounds)
        title = ax.texts[0].get_text()
        svg_path = output_dir / (name + ".svg")
        fig.savefig(svg_path, metadata={"Date": None, "Creator": "PolyBot diagram generator", "Title": title})
        # Matplotlib's path formatter leaves trailing spaces. Normalize the
        # generated text so figure refreshes pass the repository diff checks.
        svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
        fig.savefig(output_dir / (name + ".png"), metadata={"Software": "PolyBot diagram generator"})
        plt.close(fig)
        manifest["figures"][name] = {"svg": name + ".svg", "png": name + ".png", "title": title}
    if compatibility_copies:
        for destination, source in ALIASES.items():
            shutil.copyfile(output_dir / source, ROOT / destination)
    (output_dir / "diagrams.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/assets")
    parser.add_argument("--compatibility-copies", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    compatibility = args.compatibility_copies
    if compatibility is None:
        compatibility = args.output_dir.resolve() == (ROOT / "docs/assets").resolve()
    manifest = render(args.output_dir, compatibility)
    print(json.dumps({"figures": list(manifest["figures"]), "compatibility_copies": bool(compatibility)}))


if __name__ == "__main__":
    main()
