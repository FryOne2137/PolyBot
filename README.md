# PolyBot

PolyBot is a fog-of-war self-play agent for [PolyEnv](https://polyenv.readthedocs.io/).  It combines:

- a Transformer over the player-view `map_tokens` (`[tiles, 23]`),
- a policy that scores only `model_request_numpy()["actions"]`, i.e. actions legal now,
- a value head used by PUCT MCTS,
- a belief decoder that samples complete map hypotheses,
- `env.make_belief_env(...)` for detached MCTS particles with no access to the real hidden state,
- self-play training that keeps games, replay data and metrics in RAM only.

`full_map_numpy()` is used only as a **supervised training label** for the belief decoder. It is never sent to the policy, value head or MCTS search.

## Install

PolyEnv requires Python 3.10+ and a C++20 compiler when it is built from source.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PolyBot nie jest instalowany jako pakiet ani biblioteka PIP. Uruchamiaj go bezpośrednio z katalogu repo, dodając `src` do `PYTHONPATH`.

On Apple Silicon, install the current macOS PyTorch build. The training command selects `mps` automatically when it is available. It also enables PyTorch's MPS CPU fallback for an operation not yet implemented on Metal.

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Training pipeline

Nie uruchamiaj pełnego MCTS na losowym modelu. Pipeline ma dwa etapy:

1. `train_belief.py` — losowe legalne akcje, uczenie ukrytej mapy z `full_map_numpy()` jako etykiety. Liczy wyłącznie pozycje, w których istnieje fog-of-war.
2. `train_teacher.py` — behavior-cloning z wbudowanego, sieciowo-niezależnego heuristic teacher; generuje i uczy dane wyłącznie w RAM.
3. `train_self_play.py` — self-play z MCTS na już rozgrzanym modelu.

Cały pipeline uruchamiany jedną komendą zapisuje **wyłącznie końcowy plik modelu**:

```bash
python scripts/train_pipeline.py \
  --belief-positions 2_000 \
  --self-play-games 10 \
  --updates-per-game 2 \
  --simulations 16 \
  --particles 2 \
  --max-moves 300 \
  --device auto \
  --model-out models/polybot.pt
```

Możesz uruchomić etapy osobno:

```bash
python scripts/train_belief.py \
  --positions 20_000 --model-out models/belief.pt

python scripts/train_teacher.py \
  --model-in models/belief.pt --games 100 --model-out models/teacher.pt

python scripts/train_self_play.py \
  --model-in models/teacher.pt --model-out models/self_play.pt \
  --max-moves 300
```

## Evaluation

Evaluation uses fixed seeds, greedy MCTS and writes no games or metrics files.
The checkpoint alternates player slots, so the result is not biased toward the
first player. Compare first against random play:

```bash
python scripts/evaluate.py \
  --model-a models/self_play.pt \
  --games 10 --simulations 32 --particles 2 --max-moves 300 --device mps
```

To compare a newer checkpoint with an older one, pass `--model-b`.

Postęp i składniki reward są wypisywane wyłącznie do terminala. Self-play pokazuje wybrane urządzenie (`mps`, `cuda` albo `cpu`), numer gry/ruchu oraz płynny pasek bieżącego MCTS. `--max-moves` ogranicza długość jednej gry (domyślnie 300); po przekroczeniu limitu wynik jest traktowany jako remis. Replay buffer, gry, pełne mapy, eventy i stan optymalizera nie są zapisywane na dysk.

MCTS zbiera domyślnie `8` liści i wysyła je do GPU w jednym forwardzie (`--inference-batch-size 8`). To zwiększa wykorzystanie CUDA bez zmiany liczby symulacji ani informacji dostępnej botowi. Na GPU z większym zapasem pamięci można próbować `16` albo `32`; przy błędzie CUDA out-of-memory wróć do `8` lub `4`.

Na CUDA można też uruchomić kilka niezależnych actorów (`--actor-workers 4`). Każdy zachowuje własne środowisko i drzewo MCTS na CPU, a wspólny worker GPU łączy ich żądania w batch do `--inference-max-batch-size 64`. Zacznij od czterech actorów; zwiększaj tylko po krótkim teście stabilności. Ten tryb nadal zapisuje wyłącznie checkpoint modelu.

W pierwszych rundach self-play `--exploration-rate 0.20` wykonuje część ruchów z bezpiecznego curriculum legalnych akcji (miasta, walka, eksploracja), aby uniknąć danych z samych remisów. MCTS nadal tworzy target policy dla każdej pozycji.

Gdy dwa identyczne checkpointy produkują same remisy, użyj `--opponent heuristic` albo `--opponent mixed --random-opponent-rate 0.25 --mixed-opponent heuristic`. W grach przeciw zewnętrznemu przeciwnikowi model zbiera przykłady wyłącznie ze swoich ruchów; przeciwnik nadal wpływa na jego target value. Domyślne `--teacher-replay-fraction 0.25` dodaje do każdego batcha w RAM przykłady heuristic teacher. To chroni dobry bootstrap przed utratą ofensywnego zachowania wskutek szumnych targetów wczesnego MCTS; nie zapisuje partii ani replay na dysk.

## Project layout

```text
src/polybot/
├── config.py
├── models/
│   └── policy_value_belief.py    Map Transformer + policy/value/belief heads
├── planning/
│   └── belief_mcts.py            PUCT over detached belief particles
├── training/
│   ├── self_play.py              games, replay buffer and full-map labels
│   ├── replay_buffer.py           stored self-play examples
│   ├── optimisation.py            policy/value/belief loss functions
│   └── rewards.py                 transparent reward shaping
├── data/
│   ├── token_schema.py            authoritative order/ranges of 23 token fields
│   └── polyenv_features.py        conversion from PolyEnv packets to tensors
└── infrastructure/
    ├── device.py                 CPU/CUDA/Apple Metal MPS selection
    └── model_io.py               read/write model weights only

scripts/
├── train_belief.py               stage 1: belief warm-up
├── train_self_play.py            stage 2: MCTS self-play
└── train_pipeline.py             both stages, one final model write
```

## Information-flow rule

```text
model_request_numpy() ──► policy/value/MCTS inputs
visible_events_numpy() ─► optional belief-model history input
full_map_numpy() ───────► belief-loss label only
make_belief_env() ──────► rollout worlds built from predicted complete maps
```

Never use `env.clone()` to branch directly from the real game under fog. `clone()` is safe only after `make_belief_env()` has created a detached hypothetical particle.

## Training objective

```text
loss = policy cross-entropy(MCTS visits)
     + value Huber loss(final win/loss)
     + 0.35 × belief cross-entropy(hidden map tokens)
```

The terminal objective is win `+1` / loss `-1`. Small logged shaping terms make early exploration less sparse; they do not replace the terminal objective.
