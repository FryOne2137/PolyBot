# PolyBot

PolyBot is a fog-of-war self-play agent for [PolyEnv](https://polyenv.readthedocs.io/).  It combines:

- a Transformer over the player-view `map_tokens` (`[tiles, 23]`),
- a factorized policy that scores only `model_request_numpy()["actions"]`, i.e. actions legal now,
- a value head used by PUCT MCTS,
- a belief decoder that samples complete map hypotheses,
- `env.make_belief_env(...)` for detached MCTS particles with no access to the real hidden state,
- self-play training with RAM-only replay and optional durable league history.

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

Nie uruchamiaj pełnego MCTS na losowym modelu. Pipeline ma sześć etapów:

1. `train_belief.py` — losowe legalne akcje, uczenie ukrytej mapy z `full_map_numpy()` jako etykiety. Liczy wyłącznie pozycje, w których istnieje fog-of-war.
2. `train_teacher.py` — behavior-cloning z wbudowanego, sieciowo-niezależnego heuristic teacher; generuje i uczy dane wyłącznie w RAM.
3. `train_vector_teacher.py` — szybkie, natywne C++ VectorGameEnv behavior-cloning policy na setkach envów równolegle. Ten etap nie używa `full_map_numpy()`, więc nie trenuje belief decoder.
4. `train_flat_actor_critic.py` — szybki, równoległy actor-critic na
   `VectorGameEnv`; teacher-KL utrzymuje podstawy strategii, gdy reward jest
   jeszcze rzadki.
5. `train_self_play.py` — globalny self-play z natywnym PolyEnv `MctsPool`:
   wszystkie aktywne gry i particles są jednym pool-em, więc selekcja,
   branchowanie i backup PUCT działają w C++, a PyTorch liczy duże batch'e
   liści. Na starej wersji PolyEnv skrypt bezpiecznie wraca do wolniejszego
   MCTS w Pythonie.
6. druga, mniej eksploracyjna runda MCTS. Jej target value używa TD(lambda):
   root value kolejnej decyzji miesza się z późniejszym zwrotem z partii.

Cały pipeline uruchamiany jedną komendą zapisuje końcowy plik modelu oraz
lekkie artefakty diagnostyczne w `runs/pipeline/train/`:

```bash
python scripts/train_pipeline.py \
  --belief-positions 20_000 \
  --teacher-games 100 \
  --vector-teacher-steps 10_000 \
  --flat-steps 10_000 \
  --self-play-games 200 \
  --refinement-games 100 \
  --simulations 32 --particles 4 --workers 32 --mcts-threads 1 \
  --max-moves 300 --device cuda \
  --model-out models/polybot.pt
```

Po pierwszym dobrym checkpointcie nie trenuj kandydata wyłącznie przeciw
heurystyce. Podaj obecnego championa jako zamrożony model przeciwnika. Pierwszy
`--champion-model` jest punktem odniesienia dla bramki promocji, a kolejne
tworzą pulę, z której przeciwnik jest losowany. Jeżeli kandydat nie uzyska na
osobnych, sparowanych mapach wyniku `win + 0.5 * draw >= --promotion-score`
oraz dolnej granicy ufności `>= --promotion-lower-bound`, `--model-out`
pozostanie championem — słabszy kandydat nigdy go nie nadpisze.
Liga obraca zestaw testowych map przez `--evaluation-seed-stride`, aby kolejne
promocje nie przeuczyły się do jednego, stale powtarzanego benchmarku.

```powershell
python scripts/train_pipeline.py `
  --model-in models\teacher-v1.pt `
  --belief-positions 0 --teacher-games 0 `
  --vector-teacher-steps 0 --flat-steps 0 `
  --self-play-games 600 --refinement-games 300 `
  --updates-per-game 8 --refinement-updates-per-game 8 `
  --simulations 32 --particles 4 --workers 32 --mcts-threads 1 `
  --mcts-max-actions 1024 --batch-size 64 --learning-rate 1e-5 `
  --replay-capacity 100000 --belief-weight 0.10 --td-lambda 0.95 `
  --max-moves 400 --device cuda `
  --champion-model models\teacher-v1.pt `
  --promotion-games 100 --promotion-score 0.55 `
  --promotion-lower-bound 0.50 --evaluation-seed 1500000007 `
  --model-out models\champion-v2.pt
```

Możesz uruchomić etapy osobno:

```bash
python scripts/train_belief.py \
  --positions 20_000 --model-out models/belief.pt

python scripts/train_teacher.py \
  --model-in models/belief.pt --games 100 --model-out models/teacher.pt

python scripts/train_vector_teacher.py \
  --model-in models/teacher.pt --num-envs 128 --steps 10_000 \
  --num-threads 8 --updates-per-step 2 --device cuda --model-out models/vector_teacher.pt

python scripts/train_self_play.py \
  --model-in models/vector_teacher.pt --model-out models/self_play.pt \
  --max-moves 300
```

## Population league and SQLite history

`scripts/train_league.py` zaczyna od szybkiego curriculum bez MCTS w natywnym
`VectorGameEnv`. Nauczyciel visible-only pokazuje przejmowanie wiosek, ruch do
widocznych celów, ekonomię i walkę. Po nim działa etap `objectives`: zbiera on
krótkie, rzeczywiste rollouty z kilku kolejnych decyzji i liczy GAE, dzięki
czemu sygnał z wioski/miasta dociera także do ruchu, który do niego prowadzi.
Nie tworzy sztucznych map ani nie czyta ukrytych pól. Dopiero potem actor-critic
uczy się z własnych akcji. Krótkie, sparowane bramki cofają każdy blok, który
pogorszył grę względem zamrożonego championa. Direct-play trwa co najmniej
`--direct-actor-steps`, a w razie potrzeby rośnie do
`--direct-actor-max-steps`. Baseline chroni wejściowy checkpoint, ale nie jest
liczony jako wytrenowany sukces. Gotowość wymaga
`--direct-readiness-patience` zaakceptowanych bramek po aktualizacji. Rollback
złej próby nie kasuje najlepszego stanu, a po
`--direct-regression-patience` kolejnych odrzuconych blokach etap kończy się
wcześniej zamiast marnować resztę budżetu.

Następnie działa drugi szybki etap bez MCTS: kandydat gra surową polityką
przeciw championowi i historycznym checkpointom dobranym przez PFSP. Tury
przeciwników wykonują zamrożone modele, a kandydat uczy się własnych akcji,
wartości pozycji i destyluje zachowanie populacji. Pierwszy model może jeszcze
korzystać z warm-upu teachera i actor-critic, lecz po pierwszej promocji liga
automatycznie przechodzi do `champion_population`: teacher i actor-critic są
pomijane, ale poprzedza ją mały, silnie zakotwiczony refresh `objectives`
(`--post-bootstrap-objective-steps`). Bieżący champion otrzymuje 65% losowań,
a populacja używa budżetu
`--post-bootstrap-population-steps` do
`--post-bootstrap-population-max-steps`.

Brama populacji traktuje wygrany wynik przeciw populacji jako kryterium główne.
Najgorszy wynik historyczny pozostaje twardym floor, a przejęcia miast i
completion są tie-breakerami zamiast czterech niezależnych warunków blokujących
każdą zmianę polityki. Gry treningowe w natywnym `VectorGameEnv` nie mają
sztucznego limitu ruchów, a bramki direct używają `--direct-gate-max-moves`.
W population play tylko `--exploration-rate` tur kandydata jest próbkowana;
pozostałe wykonują ruch greedy, tak jak w końcowej bramce. Dodatni TD advantage
z tych tur nadal wzmacnia wybraną akcję, więc stabilność nie oznacza zatrzymania
uczenia.

Domyślny `--champion-mode flat` zaczyna od promocji całkowicie niezależnej od
MCTS. Końcowa bramka używa surowej polityki przeciw championowi i całej
historycznej populacji. Tylko po bezpiecznym, realnym awansie uruchamiany jest krótki,
beztreningowy probe: `--mcts-probe-games 10` przy
`--mcts-probe-simulations 32`. Raw policy i search@32 grają na tych samych
sparowanych mapach. Probe przechodzi tylko wtedy, gdy search@32 osiąga co
najmniej `--mcts-probe-score` i jest ściśle lepszy od raw policy o więcej niż
`--mcts-probe-min-improvement`. Nie wykonuje aktualizacji ani nie zmienia Elo.

Po `--mcts-probe-patience` kolejnych sukcesach SQLite oznacza lineage jako
`mcts_ready`. Od następnej rundy MCTS@32 staje się dodatkową bramką awansu:
kandydat musi najpierw wygrać płasko, a następnie ponownie wykazać poprawę z
search@32. Jeżeli search nie pomaga, dotychczasowy champion pozostaje bez zmian,
a poprawny płaski checkpoint kandydata zostaje zachowany jako challenger.
Status bramki jest dziedziczony przez kolejne modele, więc pojedyncza porażka
probe nie wyłącza automatycznie sprawdzania MCTS.

Ciężka historyczna drabina pozostaje dostępna przez `--champion-mode mcts`, ale
zalecana komenda początkowa ustawia `--population-games 0`. Dzięki temu jedynym
kosztem MCTS w rundzie jest krótki probe, a wszystkie właściwe aktualizacje
odbywają się w szybkim `VectorGameEnv`.

Probe i opcjonalne gry MCTS przeciw populacji korzystają z natywnego
`SelfPlayPool`. `agent_id` wybiera właściwy checkpoint osobno dla każdego
gracza, a batche belief i liści są grupowane według modelu i składane z powrotem
w kolejności oczekiwanej przez silnik. Nie ma tu krótkotrwałych `MctsPool`.
Aktualny champion dostaje stałą część losowań, a
pozostała część jest dzielona metodą PFSP: częściej wybierane są historyczne
modele, z którymi champion miał słabszy wynik albo których jeszcze nie
zmierzono. Teacher, bieżący champion i najlepsi wcześniejsi championi tworzą
ograniczoną przez `--population-size` pulę.

We wszystkich fazach uczących zamrożony champion kotwiczy politykę i ocenę
pozycji (`--champion-anchor-weight` oraz
`--champion-value-anchor-weight`). MCTS nadal wnosi nowe ruchy, ale pojedyncza
runda z płytkim lub zaszumionym wyszukiwaniem nie może już tak łatwo nadpisać
dobrej polityki wejściowej ani zniszczyć głowicy value. Kotwica jest używana
wyłącznie jako target trust-region i nigdy nie trafia do optymalizatora.

Kandydat jest promowany tylko wtedy, gdy jednocześnie pokona aktualnego
championa, przejdzie dolną granicę ufności, osiągnie wymagany średni wynik
przeciw populacji i nie spadnie poniżej progu regresji przeciw żadnemu
zachowanemu stylowi. Kandydat z bezpiecznym wynikiem co najmniej `0.50`, który
nie osiągnął jeszcze progu promocji, jest zachowywany jako `challenger` i dalej
trenowany w następnej rundzie. Słaby kandydat jest odrzucany, a champion nigdy
nie jest nadpisywany. Przerwany run można kontynuować przez
`--challenger-in <checkpoint-challengera>` przy zachowaniu tego samego
`--champion-in`.

`--league-db models/league/league.sqlite3` zapisuje konfiguracje runów, rundy,
lineage i sumy SHA-256 checkpointów, dobór przeciwników, wszystkie gry etapu
populacyjnego i bramek, wyniki, metryki treningowe oraz pomocniczy ranking Elo.
Tabela `model_capabilities` przechowuje również wynik raw/search, serię probe'ów
i trwały status `mcts_ready`.
Ciężkie tensory modeli pozostają w atomowo zapisywanych plikach `.pt`; baza
przechowuje ich ścieżki i sumy kontrolne, dzięki czemu nie rośnie o setki MB na
każdy checkpoint. Ranking Elo służy do diagnostyki, nie decyduje o promocji.

Pełna zalecana komenda znajduje się w `command.txt`. Po runie ranking i ostatnie
bramki można odczytać bez uruchamiania PyTorch:

```powershell
python scripts/league_report.py --league-db models\league\league.sqlite3
```

## Training runs and plots

Każdy skrypt treningowy tworzy katalog w stylu Ultralytics, np.
`runs/belief/train`, `runs/teacher/train` i `runs/self_play_pool/train`.
Kolejne uruchomienia nie nadpisują wyników: powstają `train2`, `train3` itd.
Każdy zakończony run zawiera:

- `args.json` i `config.json` — argumenty oraz konfigurację modelu/treningu;
- `metrics.jsonl` — metryki dopisywane na bieżąco, również przy długim runie;
- `metrics.csv` i `summary.json` — tabela przebiegu i podsumowanie;
- `results.png`, `losses.png`, `metrics.png` — surowe i wygładzone krzywe.

Ścieżkę można zmienić przez `--project`, nazwę przez `--name`, a świadome
nadpisanie konkretnego katalogu włączyć przez `--exist-ok`. Pipeline trzyma
wykres każdego etapu w podfolderach jednego runu, natomiast liga tworzy
`round_001`, `round_002` itd. Replay buffer, pełne mapy, eventy i stan
optymalizera nadal pozostają w RAM. Liga dodatkowo zapisuje do SQLite zwarte
rekordy rozegranych partii i ich wyniki, bez tensorów obserwacji.

## Fast native MCTS: `SelfPlayPool`

Nowszy PolyEnv udostępnia `SelfPlayPool`: C++ utrzymuje jednocześnie żywe gry,
ich hipotetyczne belief-worlds, drzewa MCTS i workery. Python nie tworzy w
pętli `GameEnv` ani krótkotrwałego `MctsPool`; dostaje tylko stałe, gęste batche
do GPU. PolyBot używa też buforów `*_into()` (na CUDA przypiętych w RAM) oraz
`max_pending_leaves_per_tree`, więc jeden forward GPU może ocenić do
`num_envs × pending_leaves` różnych liści.

To jest szybki etap **pełnego ISMCTS**. Dla każdej żywej gry PolyEnv przyjmuje
oddzielne particle worlds dla każdego gracza (`[env, player, particle, tile,
23]`) i przy zmianie aktora re-determinizuje drzewo z jego własnej obserwacji.
`--particles` określa więc liczbę tych hipotez. Bounded
`visible_action_history` jest przekazywana zarówno do belief requestów, jak i
liści MCTS. Po zakończeniu MCTS PolyEnv udostępnia wyłącznie dla lossu belief
etykiety `belief_targets_numpy()`: nie są one przekazywane do `submit_beliefs`,
policy ani MCTS, więc nie mogą wpłynąć na zagrany ruch.

Optymalizacja jest przeplatana z rolloutami: gdy kończy się partia, jej wynik
jest propagowany przez decyzje z bootstrapem MCTS (`TD(0.95)`), model dostaje
zaplanowaną porcję aktualizacji, a kolejne ruchy generuje nowszy kandydat.
Partia przerwana przez `--max-moves` nadal uczy policy i belief, ale nie udaje
już remisu w lossie value. Zamrożony `--anchor-model` zachowuje część policy
poprzednika w targetach. Ręczne priory wyszukiwania są wygaszane do zera, a
ewaluacja porównuje same checkpointy bez wspólnej heurystyki.

Checkpoint tworzony wyłącznie przez belief warm-up nie ma jeszcze kompetentnej
policy/value. Przed pierwszym SelfPlayPool trzeba wykonać bootstrap
`train_teacher.py`; self-play zaczyna wtedy od zachowania „idź do celu”, po
czym wygaszanie curriculum pozwala zastąpić je strategią wynikającą z wygranych.

Domyślne `--visible-action-history 0` utrzymuje identyczne wejścia policy/value
w szybkim treningu i w ewaluacji `GameEnv`. Historię należy włączyć dopiero,
gdy używany evaluator przekazuje ten sam tensor historii co `SelfPlayPool`.

### Eksperymentalne top-K tylko w PolyBot

`--mcts-top-actions K` włącza pierwszy, jakościowy etap ograniczania policy
MCTS. W priorach zachowywane jest top-K sieci, `--mcts-random-actions N`
losowych legalnych ruchów oraz wszystkie ataki, end-turn, budowa mostu, ruiny,
eksploracja, zakładanie i zdobywanie miast. Pełny `action_mask` nadal trafia do
walidacji belief i PolyEnv, więc eksperyment nie ukrywa legalnych ruchów przed
silnikiem. PolyBot ponownie projektuje zwróconą policy korzenia na zapamiętany
zbiór kandydatów, dlatego wykonany ruch i target treningowy naprawdę respektują
filtr. Z tego powodu etap mierzy przede wszystkim jakość; rzeczywiste
usunięcie krawędzi i pełny zysk wydajności wymagają późniejszego API PolyEnv.

Logi, wykresy i SQLite zawierają `search_mean_legal_actions`,
`search_mean_selected_actions`, `search_action_reduction` oraz
`search_prior_mass_retained`. Dodatkowe metryki zgłaszają wizyty PolyEnv poza
kandydatami i brak zapamiętanego zbioru dla korzenia. W `command.txt` test
`4a` porównuje ten sam checkpoint na sparowanych mapach: filtrowany model A
przeciw pełnemu modelowi B. Wartość `0` jest dokładnym przełącznikiem
wyłączającym stare zachowanie.

Szybki samodzielny test na CUDA:

```powershell
python scripts/train_self_play_pool.py `
  --model-in models\champion-v3.pt `
  --anchor-model models\champion-v3.pt --anchor-policy-weight 0.20 `
  --model-out models\pool-smoke.pt `
  --num-envs 128 --steps 1000 --updates 500 `
  --num-threads 16 --max-pending-leaves 4 `
  --simulations 32 --mcts-max-actions 1024 `
  --batch-size 64 --learning-rate 3e-5 --device cuda
```

Do pełnego pipeline dodaj np. `--self-play-pool-envs 128
--self-play-pool-steps 4000 --self-play-pool-updates 2000
--self-play-pool-threads 16 --self-play-pool-pending-leaves 4` przed
wolniejszymi etapami `--self-play-games` i `--refinement-games`.

## Evaluation

Samodzielny evaluator używa fixed seeds i greedy MCTS oraz nie zapisuje plików.
Dla `--model-b` wykonuje wszystkie gry równolegle w wielomodelowym
`SelfPlayPool`; bramka ligi korzysta z dokładnie tego samego protokołu, a wynik
każdej gry trafia dodatkowo do SQLite. Tylko diagnostyczny przeciwnik losowy
pozostaje w skalarnej ścieżce, ponieważ nie jest checkpointem routowanym przez
`agent_id`.
Każda mapa jest rozgrywana dwa razy z zamianą slotów graczy, więc wynik nie
jest obciążony przewagą pierwszego gracza. Zwracane są też `model_a_score` i
`score_lower_95`. Compare first against random play:

```bash
python scripts/evaluate.py \
  --model-a models/self_play.pt \
  --games 10 --simulations 32 --particles 2 --max-moves 300 --device mps
```

To compare a newer checkpoint with an older one, pass `--model-b`.

Postęp nadal jest widoczny w terminalu, a skalarne metryki i wykresy trafiają
do katalogu runu. Self-play pokazuje wybrane urządzenie (`mps`, `cuda` albo
`cpu`), numer gry/ruchu oraz płynny pasek bieżącego MCTS. `--max-moves`
ogranicza długość jednej gry (domyślnie 300); po przekroczeniu limitu wynik
jest remisem w ewaluacji, natomiast w treningu nie uczestniczy w lossie value.
Replay buffer, gry, pełne mapy, eventy i stan optymalizera nie są zapisywane
na dysk. Pipeline nadpisuje wskazany plik modelu
po bezpiecznych granicach etapów, dzięki czemu po awarii natywnego MCTS
pozostaje poprawny model do wznowienia.

W globalnym MCTS zacznij od `--mcts-threads 1`. Każda realna decyzja tworzy nowy natywny `MctsPool`; jeden wątek C++ zapobiega tworzeniu wielkiej liczby krótkotrwałych workerów. GPU nadal dostaje jeden wspólny batch ze wszystkich aktywnych gier i particle worlds. Po stabilnym runie można ostrożnie sprawdzić `2` albo `4`.

MCTS zbiera domyślnie `8` liści i wysyła je do GPU w jednym forwardzie (`--inference-batch-size 8`). To zwiększa wykorzystanie CUDA bez zmiany liczby symulacji ani informacji dostępnej botowi. Na GPU z większym zapasem pamięci można próbować `16` albo `32`; przy błędzie CUDA out-of-memory wróć do `8` lub `4`.

W pierwszych rundach self-play `--exploration-rate 0.20` wykonuje część ruchów z bezpiecznego curriculum legalnych akcji (miasta, walka, eksploracja), aby uniknąć danych z samych remisów. MCTS nadal tworzy target policy dla każdej pozycji.

Gdy dwa identyczne checkpointy produkują same remisy, użyj `--opponent heuristic` albo `--opponent mixed --random-opponent-rate 0.25 --mixed-opponent heuristic`. Po uzyskaniu championa użyj zamiast tego `--opponent model --opponent-model models/champion.pt` albo puli przez wielokrotne `--opponent-model`. W grach przeciw zewnętrznemu przeciwnikowi model zbiera przykłady wyłącznie ze swoich ruchów; przeciwnik nadal wpływa na jego target value. Domyślne `--teacher-replay-fraction 0.25` dodaje do każdego batcha w RAM przykłady heuristic teacher. To chroni dobry bootstrap przed utratą ofensywnego zachowania wskutek szumnych targetów wczesnego MCTS; nie zapisuje partii ani replay na dysk.

## Project layout

```text
src/polybot/
├── config.py
├── models/
│   └── policy_value_belief.py    Map Transformer + factorized policy/value/belief heads
├── planning/
│   └── belief_mcts.py            PUCT over detached belief particles
├── training/
│   ├── self_play.py              games, replay buffer and full-map labels
│   ├── league.py                 PFSP population training and promotion gates
│   ├── league_db.py              SQLite games, metrics, lineage and ratings
│   ├── vector_population.py      raw-policy population play without MCTS
│   ├── vector_objectives.py       multi-step visible-objective rollouts (GAE)
│   ├── routed_self_play_pool.py  multi-model native training and paired gates
│   ├── replay_buffer.py           stored self-play examples
│   ├── optimisation.py            policy/value/belief loss functions
│   ├── run_artifacts.py            incremented runs, scalar tables and plots
│   └── rewards.py                 transparent reward shaping
├── data/
│   ├── token_schema.py            authoritative order/ranges of 23 token fields
│   └── polyenv_features.py        conversion from PolyEnv packets to tensors
└── infrastructure/
    ├── device.py                 CPU/CUDA/Apple Metal MPS selection
    └── model_io.py               read/write model weights only

scripts/
├── train_belief.py               stage 1: belief warm-up
├── train_teacher.py              stage 2: scalar heuristic teacher
├── train_vector_teacher.py       stage 3: native C++ vector curriculum
├── train_self_play.py            stage 4: scalar MCTS self-play
├── train_league.py               population league
├── league_report.py              SQLite standings and recent gates
└── train_pipeline.py             all curriculum stages, one final model write
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
     + value Huber loss(TD(lambda) return / final win-loss)
     + 0.35 × belief cross-entropy(hidden semantic map fields)
```

The belief objective excludes per-game runtime city IDs and decoder-owned
visibility fields. The terminal objective is win `+1` / loss `-1`; with the
default Monte Carlo target it is not discounted by game length. Small logged
shaping terms make early exploration less sparse; they do not replace the
terminal objective.

## Factorized legal-action policy

The default new model does not run action-to-map attention for every legal
action. It computes a score from four components: action type, encoded source
tile, encoded target tile and remaining semantic options/costs. Raw unit/city
runtime IDs are deliberately excluded from this fast score, so it is less able
to memorise identifiers that change from one game to another. Existing
checkpoints keep their old attention policy automatically; start a new
pipeline without `--model-in` to use the factorized model.
