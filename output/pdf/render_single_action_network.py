"""Render the complete-action shared encoder as one connected neural diagram."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from polybot.config import ModelConfig
from polybot.models.policy_value_belief import PolicyValueBeliefNet
from polybot.models.policy_value_belief import BELIEF_ACTION_FEATURES
from polybot.data.token_schema import MAP_FEATURE_MAX

OUT = Path(__file__).resolve().parent
STEM = "polybot-one-network-single-action-policy-v2-spaced"
CFG = ModelConfig(policy_architecture="relational")
MODEL = PolicyValueBeliefNet(CFG)
COUNTS = {name: sum(p.numel() for p in module.parameters())
          for name, module in MODEL.named_children()}
UNUSED = 0
TOTAL = sum(p.numel() for p in MODEL.parameters())
FLAT = sum(count for name, count in COUNTS.items()
           if not name.startswith("belief_") and name not in {"action_embeddings", "action_continuous"})
BELIEF = TOTAL - FLAT - UNUSED
TILE = COUNTS["tile_encoder"]
FUSION = sum(p.numel() for p in MODEL.tile_encoder.fusion.parameters())
ACTION = COUNTS["action_encoder"]
assert (TOTAL, FLAT, BELIEF, UNUSED, ACTION) == (6945185, 4980674, 1964511, 0, 163648)

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                     "svg.fonttype": "none", "svg.hashsalt": STEM})
BG = "#fbfcfe"
INK = "#26374a"
BLUE = "#456d95"
GREEN = "#317b5b"
GOLD = "#ae762e"
GREY = "#8b929c"
PURPLE = "#82679c"
TEAL = "#127d84"
FAINT = "#778394"
fig, ax = plt.subplots(figsize=(44, 20), dpi=150)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 44)
ax.set_ylim(0, 20)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
rng = np.random.default_rng(2137)
TEXT = []


def text(x, y, value, size=13, color=INK, weight="normal", ha="center", va="center"):
    artist = ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
                     ha=ha, va=va, linespacing=1.35, zorder=5,
                     bbox={"facecolor": BG, "edgecolor": "none", "pad": 4})
    TEXT.append(artist)
    return artist


def column(x, y, n, spacing=.46, color=BLUE, radius=.145):
    ys = y + (np.arange(n)[::-1] - (n-1)/2) * spacing
    points = np.column_stack([np.full(n, x), ys])
    fills = ["#ffffff", "#dedfe1", "#afb4b9", "#737a80", "#424a51"]
    for px, py in points:
        ax.add_patch(Circle((px, py), radius, facecolor=fills[int(rng.integers(5))],
                            edgecolor=color, linewidth=1.2, zorder=4))
    return points


def connect(a, b, color=BLUE, alpha=.15, lw=.48):
    lines = [(u, v) for u in a for v in b]
    ax.add_collection(LineCollection(lines, colors=color, linewidths=lw,
                                     alpha=alpha, zorder=1))


def curved_connect(a, b, controls, color=TEAL, alpha=.2):
    """Keep long map-to-head fans outside the shared action-token columns."""
    for start in a:
        for end in b:
            path = MplPath([start, controls[0], controls[1], end],
                           [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
            ax.add_patch(PathPatch(path, fill=False, edgecolor=color,
                                   linewidth=.65, alpha=alpha, zorder=1))


def annotate_col(points, label, dim, color=BLUE, top=.52, bottom=.6,
                 label_size=14, dim_size=13):
    x = points[0, 0]
    text(x, max(points[:, 1])+top, label, label_size, color, "bold", va="bottom")
    text(x, min(points[:, 1])-bottom, dim, dim_size, color, "bold", va="top")


text(.72, 19.35, "PolyBot · jedna wspólna sieć", 29, INK, "bold", ha="left")
text(.74, 18.72, "Relational · wspólny encoder kafelków i akcji · pełne legalne akcje · mapa 11 × 11 · historia 32",
     15, FAINT, ha="left")
text(43.25, 19.38, f"{TOTAL/1e6:.2f} mln parametrów".replace(".", ","), 25, INK, "bold", ha="right")
text(43.25, 18.76, f"{FLAT/1e6:.2f} mln w ścieżce flat · bez obcinania legalnych akcji".replace(".", ","),
     15, GREEN, ha="right")

# One connected trunk; the symbols stand for tensors, not individual neurons.
main_y = 11.0
board = column(1.2, main_y, 10)
embedded = column(3.6, main_y, 10)
joined = column(6.0, main_y, 10)
layers = [column(7.7 + 1.66*i, main_y, 10) for i in range(8)]
connect(board, embedded)
connect(embedded, joined)
for a, b in zip([joined, *layers[:-1]], layers):
    connect(a, b)
annotate_col(board, "Mapa\nobserwatora", "121 × 24", INK)
annotate_col(embedded, "Encoder kafelka\n+ pozycje X/Y", "121 × 192", INK)
annotate_col(joined, "+ token\nglobalny", "122 × 192", INK)
for i, layer in enumerate(layers):
    annotate_col(layer, f"T{i+1}", "122 × 192", BLUE, dim_size=11.8)
text(13.55, 15.3, "TRANSFORMER MAPY · 8 WARSTW", 20, BLUE, "bold")
text(13.55, 14.75, "Każda: 8 głowic × 24 = 192 · FFN 192 → 768 → 192", 14, BLUE)
text(13.15, 7.82, "T = liczba kafelków. Każda warstwa zachowuje (T + 1) × 192.", 12, FAINT)
text(13.15, 7.45, "Po T8: LayerNorm oraz token globalny i T kontekstowych kafelków.", 12, FAINT)

# History/base context and the new masked public/observer context both add
# to the single global token, rather than creating independent networks.
history = column(1.2, 6.1, 4, .35, PURPLE)
history_enc = column(3.6, 6.1, 4, .35, PURPLE)
connect(history, history_enc, PURPLE, .22, .65)
connect(history_enc, joined[-1:], PURPLE, .35, .8)
annotate_col(history, "Historia + baza\nglobalna / zdarzenia", "32 × 12 + 21 cech", PURPLE,
             top=.35, bottom=.4, label_size=12.3, dim_size=12)
annotate_col(history_enc, "Encoder\nhistorii i bazy", "192", PURPLE, top=.35, bottom=.4, label_size=12.5)

context = column(1.2, 3.5, 3, .31, TEAL)
context_enc = column(3.6, 3.5, 3, .31, TEAL)
connect(context, context_enc, TEAL, .28, .7)
connect(context_enc, joined[-1:], TEAL, .35, .8)
annotate_col(context, "Techy, plemię, gracze\ni typ mapy", "126 cech (w tym maski)", TEAL,
             top=.28, bottom=.28, label_size=12.3, dim_size=12)
annotate_col(context_enc, "MLP + GELU", "126 → 192 → 192", TEAL,
             top=.28, bottom=.28, label_size=13, dim_size=12)

# Shared complete action set: A tokens -> fixed latent set -> A contextual tokens.
readout = column(21.4, 11.0, 6, .34, BLUE)
actions = column(22.3, 15.6, 4, .34, TEAL)
action_tokens = column(25.3, 11.0, 6, .32, TEAL)
latent = column(28.9, 11.0, 4, .32, TEAL)
contextual = column(32.5, 11.0, 6, .32, TEAL)
connect(layers[-1], readout, BLUE, .14)
curved_connect(layers[-1], action_tokens, ((20.5, 7.3), (24.1, 8.6)), TEAL, .10)
connect(readout, action_tokens, TEAL, .14)
connect(actions, action_tokens, TEAL, .27)
connect(action_tokens, latent, TEAL, .25)
connect(latent, contextual, TEAL, .25)
connect(action_tokens, contextual, TEAL, .07)
annotate_col(readout, "Readout pozycji\nmean / max / attention", "192", INK, top=.4, bottom=.4, label_size=13)
annotate_col(actions, "Wszystkie legalne\nakcje silnika", "A × (12 + 4)", TEAL, top=.4, bottom=.45, label_size=14.5, dim_size=14)
annotate_col(action_tokens, "Akcje + mapa\nwspólny MLP", "A × 64", TEAL, top=.4, bottom=.4, label_size=13)
annotate_col(latent, "Odczyt zbioru", "16 × 64", TEAL, top=.4, bottom=.4, label_size=13)
annotate_col(contextual, "Powrót do akcji", "A × 64", TEAL, top=.4, bottom=.4, label_size=13)
text(28.9, 14.4, "WSPÓLNY ENCODER AKCJI", 19, TEAL, "bold")
text(28.9, 13.85, "4 głowice · 16 tokenów pośrednich · uwaga O(A × 16)", 12.7, TEAL)

# Policy reads only shared contextual tokens. No raw-action bypass exists.
policy_norm = column(36.0, 16.0, 5, .34, GREEN)
policy_mlp = column(39.3, 16.0, 5, .34, GREEN)
policy_out = column(42.8, 16.0, 5, .34, GREEN)
connect(contextual, policy_norm, GREEN, .25)
connect(policy_norm, policy_mlp, GREEN, .23)
connect(policy_mlp, policy_out, GREEN, .23)
annotate_col(policy_norm, "LayerNorm", "A × 64", GREEN, top=.45, bottom=.45, label_size=13)
annotate_col(policy_mlp, "Jeden wspólny MLP", "64 → 128 → 1", GREEN, top=.45, bottom=.45, label_size=13)
annotate_col(policy_out, "Logity + softmax", "1 na akcję → A", GREEN, top=.45, bottom=.45)
text(39.3, 17.6, "POLICY", 19, GREEN, "bold")

value_readout = column(36.0, 10.3, 4, .34, GOLD)
value_mlp = column(39.3, 10.3, 5, .34, GOLD)
value_out = column(42.8, 10.3, 1, color=GOLD, radius=.20)
curved_connect(readout, value_readout, ((24.1, 7.0), (33.0, 7.0)), GOLD, .17)
connect(contextual, value_readout, GOLD, .25)
connect(value_readout, value_mlp, GOLD, .27)
connect(value_mlp, value_out, GOLD, .45, .9)
annotate_col(value_readout, "Uwaga po A akcjach\n+ stan mapy", "1 × 64 → 192", GOLD, top=.4, bottom=.45, label_size=13)
annotate_col(value_mlp, "MLP + GELU", "192 → 192 → 1", GOLD, top=.45, bottom=.45)
annotate_col(value_out, "Tanh", "Ocena ∈ [−1, 1]", GOLD, top=.45, bottom=.7, label_size=13, dim_size=12)
text(39.3, 12.15, "VALUE", 19, GOLD, "bold")

belief_attention = column(35.6, 4.6, 6, .32, GREY)
belief_1 = column(37.4, 4.6, 6, .32, GREY)
belief_2 = column(39.2, 4.6, 6, .32, GREY)
belief_heads = column(41.0, 4.6, 6, .32, GREY)
belief_out = column(42.8, 4.6, 6, .32, GREY)
curved_connect(layers[-1], belief_attention, ((20.5, 2.4), (32.0, 2.8)), GREY, .11)
connect(contextual, belief_attention, GREY, .24)
for left, right in zip([belief_attention, belief_1, belief_2, belief_heads], [belief_1, belief_2, belief_heads, belief_out]):
    connect(left, right, GREY, .24)
annotate_col(belief_attention, "Akcje → kafelki\n8 głowic", "64 → 192", GREY, top=.35, bottom=.4, label_size=12, dim_size=12)
annotate_col(belief_1, "B1", "T × 192", GREY, top=.35, bottom=.4, label_size=13, dim_size=12)
annotate_col(belief_2, "B2", "T × 192", GREY, top=.35, bottom=.4, label_size=13, dim_size=12)
annotate_col(belief_heads, "23 głowice", "192 → C[k]", GREY, top=.35, bottom=.4, label_size=12, dim_size=12)
annotate_col(belief_out, "Cechy mapy", "T × C[k]", GREY, top=.35, bottom=.4, label_size=12, dim_size=12)
text(39.3, 6.65, "BELIEF · pomijany w flat", 18, GREY, "bold")
text(39.3, 2.8, "B1/B2: 8 głowic; FFN 192 → 768 → 192.\nWspólne akcje + lokalne maski dowodów; 23 rozkłady semantycznych pól mapy.", 11.8, FAINT)

# Explanatory labels occupy the whitespace under the shared trunk. The
# network itself stays continuous, with no detached process boxes.
text(7.55, 6.15, "PEŁNA LISTA AKCJI DLA POLICY, VALUE I BELIEF", 14, TEAL, "bold", ha="left")
text(7.55, 5.55, "Każda legalna akcja dostaje wektor; padding nie jest akcją i jest maskowany.", 12.5, INK, ha="left")
text(7.55, 5.07, "Typ, opcje, koszty i kafelki źródła / celu / miasta → wspólne tokeny 64-D.", 12.5, INK, ha="left")
text(7.55, 4.59, "A × 64 → uwaga przez 16 tokenów → A × 64, bez macierzy A × A.", 12.5, INK, ha="left")
text(7.55, 4.11, "Policy ocenia każdy token; value i belief odczytują cały zbiór akcji.", 12.5, INK, ha="left")
text(7.55, 3.63, "Bufory obejmują pełną przestrzeń silnika; sieć dostaje tylko rzeczywiste ruchy.", 12.5, INK, ha="left")
text(7.55, 2.65, "Kafelek: 24 pola → embeddingi + populacja i maska → MLP 194 → 192 → 192.\nKontekst 126: własne techy / plemię, gracze, publiczne punkty, statusy / maski, typ mapy.\nRelational: runtime ID są odnośnikami. Lista akcji nie ma embeddingów kolejności.", 11.8, FAINT, ha="left")
text(7.55, 1.48, "Wersja akcji v2: trening od zera. Belief-only zamraża wspólny encoder.\nPolicy: tylko wspólne tokeny → MLP. Bez starego scorera i sumowania logitów.", 11.8, FAINT, ha="left")
text(.8, 1.8, "Historia: embeddingi kategorii i X/Y → MLP 386 → 192 → 192.\nTransformer historii ×1 (8 głowic, FFN 384), gate.\nBaza: 9 globalnych + 12 zdarzeń → 192 → 192.\nT + 1: mapa 13 × 13 → 170 tokenów; mapa 16 × 16 → 257.", 10.7, FAINT, ha="left")

text(.8, .32, "Kółka i połączenia są symboliczne; podpisy podają rzeczywiste rozmiary. To jeden model ze wspólnym encoderem i trzema głowicami.",
     11.3, FAINT, ha="left")
text(43.2, .32, f"{TOTAL:,} = {FLAT:,} flat + {BELIEF:,} belief · encoder akcji: {ACTION:,}".replace(",", " "),
     10.7, FAINT, ha="right")

# Guard against cropped labels before writing any deliverable.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
cropped = []
for artist in TEXT:
    bounds = artist.get_window_extent(renderer)
    if bounds.x0 < 2 or bounds.y0 < 2 or bounds.x1 > fig.bbox.width-2 or bounds.y1 > fig.bbox.height-2:
        cropped.append(artist.get_text())
if cropped:
    raise ValueError(f"cropped diagram labels: {cropped!r}")

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / f"{STEM}.png", dpi=150, facecolor=BG,
            metadata={"Software": "PolyBot architecture renderer"})
fig.savefig(OUT / f"{STEM}.pdf", facecolor=BG,
            metadata={"Title": "PolyBot - one network, one shared action encoder, single policy scorer", "Author": "PolyBot", "CreationDate": None})
fig.savefig(OUT / f"{STEM}.svg", facecolor=BG,
            metadata={"Title": "PolyBot - one network, one shared action encoder, single policy scorer", "Date": None})
# Matplotlib emits trailing spaces inside SVG path attributes. Normalize the
# generated text so a fresh export also passes Git's whitespace checks.
svg_path = OUT / f"{STEM}.svg"
svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")

manifest = {"model_config": vars(CFG), "map_tiles": 121, "history_length": 32,
            "parameters": {"total": TOTAL, "flat": FLAT, "belief": BELIEF,
                           "unused_compatibility": UNUSED, "joint_tile_encoder": TILE, "joint_fusion": FUSION, "shared_action_encoder": ACTION},
            "layout_revision": "semantic history v2; expanded 44x20 layout; action encoder centered in its own whitespace; legal input above; long map readouts routed below the action encoder",
            "module_parameters": COUNTS, "belief_head_class_counts": [x+2 for x in MAP_FEATURE_MAX]}
(OUT / f"{STEM}.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
print(json.dumps({"stem": str(OUT / STEM), "parameters": manifest["parameters"]}))
