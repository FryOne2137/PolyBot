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

`scripts/train_league.py` uruchamia od razu self-play MCTS. Kandydat uczy się
policy targetu z odwiedzin MCTS i value targetu z wyniku partii, a champion
zmienia się wyłącznie po niezależnej ewaluacji. Teacher, objective rollouts,
flat actor-critic oraz raw-population PPO pozostają jako osobne skrypty dla
modelu od zera, ale liga ich nie wywołuje: nie mogą zablokować MCTS ani
nadpisać istniejącego checkpointu lokalnym rewardem.
PPO jest history-aware: `--visible-action-history H` włącza ostatnie `H`
zdarzeń bezpiecznych względem mgły wojny (domyślnie 32). `VectorGameEnv`
zwraca pełniejszy format 26 pól, który PolyBot zamienia na ten sam zwarty token
12 pól używany przez `SelfPlayPool`. Kandydat, zamrożeni przeciwnicy, champion
anchor, objective replay i wszystkie płaskie bramki dostają identyczne okno.
Domyślnie aktywny jest też single-network PPG
(`--direct-population-ppg`). Faza policy zachowuje PPO/GAE, ale value loss
aktualizuje wtedy tylko głowicę krytyka i nie nadpisuje wspólnego encodera.
Co `--direct-population-ppg-interval` rolloutów faza auxiliary losuje
zróżnicowane pozycje z ograniczonego bufora, zamraża na nich bieżący pełny
rozkład policy, a następnie uczy wspólny encoder z value targetów. KL względem
zamrożonego rozkładu chroni jakość ruchów podczas tej dodatkowej nauki.
Domyślny budżet to 512 pozycji z rolloutu, maksymalnie 4096 pozycji na fazę,
2 epoki co 8 rolloutów; `--no-direct-population-ppg` wykonuje test kontrolny
czystego PPO.
Niedokończona trajektoria jest bootstrapowana wyłącznie na końcu bloku albo po
przekroczeniu bezpiecznego limitu RAM
`--direct-population-ppo-max-pending-examples`; nigdy nie jest oznaczana jako
remis. PPO ma również bounded objective replay: po widocznym sukcesie
(wejście do wioski, założenie lub przejęcie miasta albo wygrana) zapisuje
publiczne decyzje prowadzące do zdarzenia i okresowo dodaje ważoną stratę
self-imitation. `--direct-population-objective-replay-max-epochs` ogranicza
średnią liczbę ponownych użyć, więc rzadki sukces nie może zdominować PPO.
Bieżący champion otrzymuje 65% losowań, a populacja używa budżetu
`--post-bootstrap-population-steps` do
`--post-bootstrap-population-max-steps`.

Brama populacji traktuje wygrany wynik przeciw populacji jako kryterium główne.
Najgorszy wynik historyczny pozostaje twardym floor, a przejęcia miast i
completion są tie-breakerami zamiast czterech niezależnych warunków blokujących
każdą zmianę polityki. Po każdym bloku PPO działa szybka, sparowana diagnostyka
`--direct-population-diagnostic-gate-games` (domyślnie 20 gier). Dopiero jeden
najlepszy checkpoint otrzymuje pełną bramkę
`--direct-population-gate-games` (domyślnie 100 gier na przeciwnika), więc czas
jest przeznaczany na dane treningowe, a nie powtarzane kosztowne ewaluacje. Gry
treningowe w natywnym `VectorGameEnv` nie mają sztucznego limitu ruchów, a
bramki direct również używają natywnego `VectorGameEnv`: dwie równoległe
partie każdej mapy zamieniają strony pretendenta, a bufory wejściowe są
ponownie używane i na CUDA przypinane w pamięci hosta. Limit bramki ustawia
`--direct-gate-max-moves`. PPO próbuje pełną politykę o
temperaturze `--direct-population-ppo-rollout-temperature`; promocja nadal
ocenia deterministyczną raw policy.

Domyślny `--champion-mode flat` pozostawia promocję całkowicie zależną od
surowej polityki. Opcjonalne `--flat-mcts-distill-games N` uruchamia Expert
Iteration tylko po obowiązkowym probe (`--mcts-probe-games >= 2`): raw policy
i search@N grają na tych samych sparowanych mapach, a search musi osiągnąć co
najmniej `--mcts-probe-score` oraz być ściśle lepszy od raw policy o więcej niż
`--mcts-probe-min-improvement`. Gdy probe nie przejdzie, destylacja jest
pomijana — nie wolno uczyć surowej policy z zaszumionego drzewa. Po
destylacji działa ta sama sparowana bramka `--flat-mcts-distill-gate-games`;
jeśli raw policy nie poprawi wyniku przeciw championowi, wszystkie aktualizacje
MCTS są cofane. Końcowa bramka nadal używa wyłącznie surowej polityki przeciw
championowi i całej historycznej populacji.

Po `--mcts-probe-patience` kolejnych sukcesach SQLite oznacza lineage jako
`mcts_ready`. W trybie `--champion-mode auto` kolejna runda automatycznie
przechodzi z flat recovery do zamrożonej generacji MCTS: najpierw zbierany jest
cały replay przez niezmieniany model, dopiero potem uczy się challenger. Musi
on przejść zarówno bramkę search, jak i raw-policy absorption; przy porażce
waga wraca do ostatniego bezpiecznego stanu. Stan jest zapisany przy
checkpointcie, więc restart ligi nie cofa go do trybu flat. Jeżeli search nie
pomaga przed pierwszym przejściem, auto pozostaje w recovery i nie używa go do
treningu. Po przejściu nieudany rung wraca natomiast do poprzedniego
MCTS-ready championa.

W zwykłej generacji MCTS domyślnie działa selective MCTS. PolyBot najpierw
wykonuje tani, publiczny forward surowej policy, a następnie kieruje do
natywnego `SelfPlayPool.select_leaves_masked_into()` tylko najbardziej
niepewne, strategiczne i losowo audytowane sloty. Pozostałe sloty wykonują
legalną direct policy. Wszystkie ruchy nadal uczą belief i value, lecz policy
otrzymuje target wyłącznie z odwiedzonych rootów MCTS — direct policy nie jest
destylowana sama do siebie. Połowa budżetu MCTS jest dodatkowo losowana z
publicznych decyzji strategicznych (walka z obrażeniami, miasto i konstrukcja),
więc pewna siebie, lecz błędna policy nie może uniknąć searcha. Ustawienia to
`--population-selective-mcts-fraction` (domyślnie 0.50),
`--population-selective-mcts-minimum-envs` i
`--population-selective-mcts-audit-fraction` (domyślnie 0.10). Domyślnie replay
zachowuje zwykłe i krytyczne targety MCTS; `--population-mcts-critical-only`
jest trybem późnej kompresji replay, nie bootstrapu. Logi zawierają m.in.
`selective_mcts_strategic_coverage_rate`, `mcts_raw_policy_agreement` i
`mcts_raw_policy_mean_l1`, dzięki czemu można odróżnić brak danych MCTS od
braku ich wpływu na policy. Strict belief calibration oraz sparowane bramki
nadal używają pełnego MCTS.

Każdy batch raportuje też `value_mae` i `value_bias` oraz osobne metryki
ukrytej obecności miasta i jednostki: `belief_hidden_city_precision/recall`
i `belief_hidden_unit_precision/recall`. Dzięki temu teren poprawnie
odgadnięty przy błędnie rozmieszczonych miastach lub jednostkach nie wygląda
fałszywie jak dobry belief.

Przed każdą generacją treningową rung wykonuje dodatkowy asymetryczny preflight
na tych samych parach seedów: candidate direct i candidate MCTS grają osobno
przeciw temu samemu zamrożonemu championowi **bez** MCTS. Jeżeli
`mcts_preflight_search_score` nie jest wyższy od
`mcts_preflight_raw_score` i nie osiąga progu `--mcts-probe-score`, rung nie
zbiera replay ani nie wykonuje optymalizacji. Chroni to model przed uczeniem z
MCTS, który jest słabszy od własnej policy. Domyślna drabina zaczyna się od
`32,64`, a nie od 16 symulacji. Gdy słabszy rung nie przejdzie preflight, liga
nie trenuje na nim, lecz może sprawdzić następny, głębszy rung z tej samej
drabiny.

Przed tym preflightem liga domyślnie wykonuje `--value-bootstrap-games 64`:
obie strony grają wyłącznie surową policy, a policy loss każdego zebranego
ruchu wynosi zero. Uczą się wyłącznie głowice `value` i `belief` z wyniku gry
i pełnej mapy dostępnej tylko jako etykieta treningowa. Dzięki temu słaby value
nie może od razu zatruć targetów MCTS; `--value-bootstrap-games 0` wyłącza ten
etap wyłącznie dla świadomych eksperymentów.

`root_shield` nie oznacza, że niezgodna particle może wejść do drzewa. Najpierw
każda particle musi zachować publiczny zestaw legalnych akcji dla aktywnego
gracza; w przeciwnym razie jest próbkowana ponownie, a po wyczerpaniu retry
pozycja jest odrzucana. Osłona rootu pozostaje dodatkową ochroną mapowania
akcji, ale nie może naprawić wizyt wykonanych w niemożliwym świecie. Dla
MCTS należy używać `--belief-action-equivalence-target 0.02` lub niżej.

Decoder belief ma też focal loss dla obecności ukrytych miast/jednostek
(`--belief-presence-focal-gamma`, domyślnie `1.5`) oraz
`belief_entity_consistency_loss` (`--belief-entity-consistency-weight`,
domyślnie `0.35`). Ten drugi karze m.in. miasto na przewidywanej wodzie i
atrybuty miasta/jednostki na tile'u, na którym decoder przewiduje brak encji.
Particle są próbkowane sekwencyjnie: statyczny teren, obecność encji, następnie
atrybuty tylko dla faktycznie wylosowanych encji. Log kalibracji zawiera odtąd
osobno wynik `collection` i stały `held-out mismatch` po aktualizacji;
parametr `--belief-calibration-eval-games` (domyślnie 8) kontroluje koszt tej
oceny bez gradientu.

Kalibracja aktualizuje prywatny, dwuwarstwowy transformer `belief_spatial_refiner`.
Widoczna mapa, legalne akcje i ich endpointy mogą więc wpływać na sąsiednie
ukryte tile'e, ale policy/value nie przechodzą przez ten moduł i nie zmieniają
się od samego treningu beliefu. Stare checkpointy są zgodne: nowa ścieżka ma
bramkę wyzerowaną przy starcie. Po pierwszym pomiarze held-out kolejny run nie
nadpisze apprentice'a, jeśli niezależny mismatch pogorszy się o więcej niż
0,5 punktu procentowego; w logu pojawi się wtedy
`regression: restored prior apprentice`.

Wymuszony tryb `--champion-mode mcts` pozostaje dostępny dla eksperymentów.
`auto` wymaga dodatniego `--population-games` i co najmniej dwóch probe games;
nie pozwala przypadkiem wejść w generowanie danych MCTS bez bramki jakości.

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

### Kalibracja belief przed destylacją MCTS

Jeżeli PolyEnv udostępnia `ismcts_belief_action_differences`, PolyBot sprawdza
przed `submit_beliefs` pełny publiczny zbiór legalnych akcji dla każdego
particle. Hipoteza z brakującą albo dodatkową akcją jest losowana ponownie;
po wyczerpaniu retry slot jest resetowany, a nie przekazywany do MCTS. Chroni
to drzewo przed wymyśloną drogą, przejściem lub budową.

Liga prowadzi ponadto osobny checkpoint `*-belief-apprentice.pt`. Dopóki jego
`belief_action_equivalence_rejection_rate` nie spadnie poniżej domyślnego 10%
na co najmniej 512 particle, aktualizowane są wyłącznie moduły belief — policy,
value oraz champion pozostają zamrożone. Checkpoint jest zachowywany pomiędzy
rundami, także gdy kandydat nie byłby gotowy do promocji. Dopiero po tej
kalibracji MCTS może być targetem dla policy/value. Ścieżkę i próg regulują
`--belief-apprentice-path`, `--belief-action-equivalence-target` oraz
`--belief-action-equivalence-min-particles`.

Po kalibracji liga może użyć `--population-belief-search-mode root_shield`.
Tryb zachowuje strict checker dla kalibracji i ewaluacji, lecz podczas zwykłego
generowania policy/value nie losuje ponownie particle'a tylko dlatego, że jego
zbiór akcji różni się od publicznej obserwacji. Po MCTS PolyBot przepisuje
policy korzenia na dokładną, publiczną kolejność `action_id` z PolyEnv; ruch
wykonany oraz target replay są więc zawsze legalne. Mismatch nadal zasila
`engine_action_constraint_loss` i metryki `root_shield_*`. Błędy strukturalne
particle'a nadal przechodzą przez strict recovery — root shield nie wyłącza
walidacji mapy.

Optymalizacja jest przeplatana z rolloutami: gdy kończy się partia, jej wynik
jest propagowany przez decyzje z bootstrapem MCTS (`TD(0.95)`), model dostaje
zaplanowaną porcję aktualizacji, a kolejne ruchy generuje nowszy kandydat.
Partia przerwana przez `--max-moves` nadal uczy policy i belief, ale nie udaje
już remisu w lossie value. Zamrożony `--anchor-model` zachowuje część policy
poprzednika w targetach. Ręczne priory wyszukiwania są wygaszane do zera, a
ewaluacja porównuje same checkpointy bez wspólnej heurystyki.

### Eksploracja AlphaZero

Podczas zbierania self-play PolyBot miesza prior **tylko w korzeniu** z
szumem Dirichleta (`--root-dirichlet-epsilon`, domyślnie `0.25`).
`--root-dirichlet-concentration` jest całkowitą koncentracją rozdzielaną na
liczbę legalnych akcji, dzięki czemu eksploracja nie znika na pozycjach z
setkami ruchów. Pierwsze `--temperature-drop-move` decyzji (domyślnie 20)
jest próbkowanych z wizyt MCTS o temperaturze `--temperature`; później
`--late-temperature 0` wybiera najczęściej odwiedzaną gałąź. Bramka,
ewaluator i interaktywne rozgrywki nie włączają tego szumu.

Ręczny `--exploration-rate` pozostał wyłącznie jako niewielki bootstrap
(domyślnie 0.05), ale Dirichlet jest głównym źródłem późniejszej eksploracji:
wpływa na samo drzewo i jego target wizyt, zamiast zastępować ruch po searchu.

Checkpoint tworzony wyłącznie przez belief warm-up nie ma jeszcze kompetentnej
policy/value. Przed pierwszym SelfPlayPool trzeba wykonać bootstrap
`train_teacher.py`; self-play zaczyna wtedy od zachowania „idź do celu”, po
czym wygaszanie curriculum pozwala zastąpić je strategią wynikającą z wygranych.

Domyślne `--visible-action-history 32` jest używane spójnie przez szybkie PPO,
objective replay, płaską ewaluację `VectorGameEnv` i `SelfPlayPool`. Wartość `0`
wyłącza pamięć na potrzeby testu porównawczego; wartości większe niż 32
zwiększają koszt jednowarstwowego attention mniej więcej kwadratowo względem
długości okna.

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

## Saved self-play games

`scripts/play_self.py` uruchamia jeden checkpoint przeciw niemu samemu na
mapach treningowych (domyślnie `lakes` 11x11, Bardur kontra Imperius). Tryb
`flat` gra bezpośrednio policy modelu, a `mcts` używa belief-MCTS. Każda partia
jest od razu zapisywana przez `GameEnv.save()` do natywnego pliku
`game_XXX.polygame`. Następnie skrypt wczytuje go przez `GameEnv.load()` i
weryfikuje liczbę ruchów, wynik oraz stan terminalny. `manifest.json` pozostaje
lekkim indeksem zawierającym konfigurację, sumę SHA-256 modelu i zbiorcze
wyniki.

```bash
python scripts/play_self.py \
  --model models/champion.pt \
  --mode flat --games 10 --map-size 11 --moves 300 \
  --tribes Bardur Imperius

python scripts/play_self.py \
  --model models/champion.pt \
  --mode mcts --games 10 --map-size 14 \
  --simulations 32 --particles 2
```

Wyniki trafiają domyślnie do `runs/self_play_games/NAZWA_MODE`; istniejący
katalog nie jest nadpisywany, tylko dostaje kolejny numer. Własne miejsce
można wskazać przez `--save-path KATALOG` (starsza nazwa `--output-dir` również
działa). `--moves N` ogranicza każdą grę do N ruchów; domyślna wartość to 300.
Starsza nazwa `--max-moves N` działa tak samo.
Plemiona obu graczy wybiera się przez `--tribes TRIBE_1 TRIBE_2`; domyślnie
są to `Bardur Imperius`.

Podczas gry `tqdm` pokazuje postęp całej serii i ruchy bieżącej partii, gracza,
ostatnią akcję, a w trybie MCTS również postęp symulacji i liczbę particles.
Wybrane urządzenie jest wypisywane przed startem. Domyślne `--device auto`
wybiera kolejno CUDA, MPS, a jeśli żadne nie jest dostępne — CPU.

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

Every belief update also logs terrain-only diagnostics, evaluated exclusively
on tiles hidden in the input: `belief_hidden_terrain_accuracy`,
`belief_hidden_terrain_nll`, per-class accuracy/support, and
`belief_bridge_geometry_accuracy`/`nll`. The parallel
`belief_action_constraint_terrain_*` metrics cover only hidden tiles selected
because a sampled particle changed the public legal-action set. These are
evaluation metrics, not extra losses; NLL is the primary measure for particle
quality because ISMCTS samples from the predicted distribution.

## Factorized legal-action policy

The default new model does not run action-to-map attention for every legal
action. It computes a score from four components: action type, encoded source
tile, encoded target tile and remaining semantic options/costs. Raw unit/city
runtime IDs are deliberately excluded from this fast score, so it is less able
to memorise identifiers that change from one game to another. Existing
checkpoints keep their old attention policy automatically; start a new
pipeline without `--model-in` to use the factorized model.
