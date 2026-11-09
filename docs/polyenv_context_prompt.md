# Prompt do Codex w repozytorium PolyEnv

Zaimplementuj w PolyEnv eksport tylko czterech nowych informacji dla modelu: własne technologie obserwatora, własne plemię obserwatora, początkową liczbę uczestników i publiczną tablicę aktualnych wyników punktowych uczestników.

Pracuj wyłącznie w PolyEnv. Nie zmieniaj PolyBot, nie licz lossów, nie dodawaj PyTorch. Nie zmieniaj mechaniki ani punktacji gry. Zachowaj dotychczasowe pola obserwacji, format 11 kolumn `state`, legalne akcje, historię, typ/rozmiar mapy oraz populację miasta w kolumnie 23. Dodawaj osobne tablice.

## Kontrakt dla batcha bieżącego obserwatora

- `own_techs`: int32[B,25], bity technologii ID 0–24, 0 nie posiada, 1 posiada. ID 25 (`Count`) nie jest technologią.
- `own_tribe`: int32[B], rzeczywiste plemię obserwatora, niezależne od plemienia terenu.
- `initial_player_count`: int32[B], początkowa liczba graczy, stała również po eliminacjach.
- `player_scores`: int32[B,16], aktualne publiczne punkty w kolejności globalnych PlayerId, także dla pozostałych graczy.
- `player_scores_mask`: uint8[B,16], 1 dla pierwotnych uczestników, 0 dla nieużywanych slotów. To wyłącznie maska paddingu, nie status aktywności.

Obsłuż 2–16 graczy i powtarzające się plemiona. Sloty ponad `initial_player_count` mają score 0 i maskę 0; rzeczywisty gracz może mieć score 0 i maskę 1. Nie usuwaj slotów ani nie zmieniaj maski po eliminacji. Wykorzystaj istniejący system naliczania score; nie przybliżaj punktów z widocznych kafelków.

W skalarnych `observation()` i `model_request_numpy()["obs"]` zachowaj `techs` i dodaj odpowiedniki wymienionych pól bez wymiaru B. Korzystaj z istniejącej tożsamości perspektywy; nie dodawaj drugiej, potencjalnie sprzecznej identyfikacji gracza.

Nie dodawaj `known_player_techs`, `known_player_tribes`, `player_status`, prywatnej ekonomii ani liczby ukrytych jednostek/miast przeciwników. Pozostałą wiedzę model ma wyciągać ze swojej mapy i widocznej historii. Publiczne punkty są celowo dozwolonym wyjątkiem. Eksportuj sumę score bez ujawniania jej prywatnych składowych. Jeżeli nie ma dotąd publicznej tablicy punktów, dodaj dokładnie taki kontrakt obserwacji.

## Aktualność, perspektywa i izolacja

Własne technologie muszą być pełne i aktualne po zakupie, nagrodzie z ruiny, zmianie gracza i resecie. Każda pozycja opisuje dane własne swojego obserwatora, nigdy cudze technologie w jego wierszu.

Dla liści MCTS korzystaj z perspektywy liścia. Po zmianie gracza nie kopiuj technologii ani plemienia gracza korzenia. Dla odłączonego świata belief korzystaj wyłącznie z tego świata i dozwolonej wiedzy, bez doczytywania rzeczywistego ukrytego stanu źródłowej gry. Score liścia musi dotyczyć jego stanu, a nie skopiowanej punktacji z korzenia.

## Podłączenie API

Obsłuż GameEnv, VectorGameEnv, MctsPool i SelfPlayPool: reset, step, zwykłe pakiety, per-player belief, maskowane liście, wybrane sloty i wszystkie warianty `*_into`.

Dla pakietów per-player użyj wymiarów [B,P,25], [B,P], [B,P], [B,P,16] i [B,P,16]. Publiczne score może być takie samo między perspektywami, własne technologie i plemię zależą od perspektywy.

Zaktualizuj batch_spec, slot_batch_spec, specyfikacje liści i belief, walidację kształtów/dtype/alignment/aliasingu i bufory wielokrotnego użytku. Wypełniaj wszystkie pola po resecie; nie zostawiaj danych poprzedniej gry. Zachowaj valid-prefix liści i semantykę state_id/episode_id. Eksport realizuj w C++, bez Pythonowej pętli po środowiskach.

## Testy

1. Technologie początkowe, zakup, ruina, zmiana perspektywy i automatyczny reset.
2. Zgodność scalar/vector/MCTS dla identycznego stanu i perspektywy.
3. Dwa, trzy i cztery gracze, duplikaty plemion, właściwe indeksy score.
4. Stała początkowa liczba i maska slotów po eliminacji.
5. Score zgodny z punktacją silnika po zmianach punktów i resecie.
6. Zmiana ukrytych technologii przeciwnika bez zmiany publicznego score/widoku nie ujawnia tych technologii w eksporcie obserwatora.
7. Zgodność zwykłych API i `*_into`, slotów i maskowania; brak starych danych po resecie.
8. Poprawne własne technologie, plemię i score liści po zmianie gracza oraz izolacja hipotetycznych światów.

Zaktualizuj dokumentację i typy, uruchom odpowiednie testy i opisz dokładny kontrakt oraz wyniki testów. Nie modyfikuj modelu ani treningu PolyBot.
