"""Small terminal-only progress display; it never writes a log file."""

from __future__ import annotations

import sys


class ConsoleProgress:
    """Render one replaceable progress line for MCTS and self-play."""

    def __init__(self, enabled: bool = True, width: int = 24) -> None:
        self.enabled = enabled
        self.width = width
        self._interactive = sys.stdout.isatty()
        self._last_line = ""

    def stage(self, text: str) -> None:
        if self.enabled:
            self._write(text, replace=False)

    def mcts(
        self,
        game: int,
        games: int,
        move: int,
        completed: int,
        total: int,
        valid_particles: int,
    ) -> None:
        total = max(total, 1)
        fraction = min(completed / total, 1.0)
        filled = int(fraction * self.width)
        bar = "#" * filled + "-" * (self.width - filled)
        line = (
            f"game {game}/{games} | move {move} | "
            f"MCTS [{bar}] {completed}/{total} | particles {valid_particles}"
        )
        self._write(line, replace=True)

    def game_complete(self, game: int, games: int, moves: int, examples: int) -> None:
        self._write(
            f"game {game}/{games} complete | moves {moves} | examples {examples}",
            replace=False,
        )

    def _write(self, line: str, replace: bool) -> None:
        if not self.enabled:
            return
        if replace and self._interactive:
            padding = " " * max(0, len(self._last_line) - len(line))
            print("\r" + line + padding, end="", flush=True)
            self._last_line = line
            return
        if replace and not self._interactive and line == self._last_line:
            return
        print(line, flush=True)
        self._last_line = line
