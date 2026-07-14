"""Spawn-safe CPU actors that own PolyEnv and communicate with one GPU parent."""

from __future__ import annotations

import traceback
import multiprocessing as mp
from typing import Any, Iterable, Iterator, List, Tuple

from ..config import MCTSConfig, TrainingConfig
from ..planning.belief_mcts import BeliefMCTS
from ..planning.process_inference import ProcessInferenceClient
from .self_play import SelfPlayRunner


Job = Tuple[int, bool, str | None, int | None]


def actor_loop(
    actor_id: int,
    tasks: Any,
    results: Any,
    requests: Any,
    responses: Any,
    training: TrainingConfig,
    mcts: MCTSConfig,
) -> None:
    """Top-level function required by Windows multiprocessing spawn."""
    client = ProcessInferenceClient(actor_id, requests, responses)
    planner = BeliefMCTS(None, None, mcts, inference=client)  # type: ignore[arg-type]
    runner = SelfPlayRunner(planner, training, logger=None)
    while True:
        job = tasks.get()
        if job is None:
            return
        game_index, use_anchor, opponent_agent, train_player = job
        try:
            examples, metrics = runner.play_game(
                training.seed + game_index,
                iteration=0,
                game_index=game_index,
                train_player=train_player,
                opponent_agent=opponent_agent,
                on_search_progress=None,
            )
            results.put((actor_id, "ok", game_index, use_anchor, examples, metrics))
        except BaseException:
            results.put((actor_id, "error", game_index, traceback.format_exc()))


class ProcessActorPool:
    """Work-conserving Windows-safe pool: a free actor gets the next env."""

    def __init__(self, workers: int, training: TrainingConfig, mcts: MCTSConfig) -> None:
        self.context = mp.get_context("spawn")
        self.workers = workers
        self.requests = self.context.Queue()
        self.results = self.context.Queue()
        self.tasks = [self.context.Queue() for _ in range(workers)]
        self.responses = [self.context.Queue() for _ in range(workers)]
        self.processes = [
            self.context.Process(
                target=actor_loop,
                args=(
                    actor_id,
                    self.tasks[actor_id],
                    self.results,
                    self.requests,
                    self.responses[actor_id],
                    training,
                    mcts,
                ),
                daemon=True,
            )
            for actor_id in range(workers)
        ]
        for process in self.processes:
            process.start()

    def run(self, jobs: Iterable[Job]) -> Iterator[tuple]:
        pending = iter(jobs)
        active = 0
        for actor_id in range(self.workers):
            try:
                self.tasks[actor_id].put(next(pending))
                active += 1
            except StopIteration:
                break
        while active:
            result = self.results.get()
            actor_id = int(result[0])
            active -= 1
            if result[1] != "ok":
                raise RuntimeError("actor %d failed in game %s:\n%s" % (actor_id, result[2], result[3]))
            try:
                self.tasks[actor_id].put(next(pending))
                active += 1
            except StopIteration:
                pass
            # Submit first, then expose the completed result.  The actor can
            # start its next environment while the parent updates tqdm and
            # copies training examples into replay.
            yield result

    def close(self) -> None:
        for task in self.tasks:
            task.put(None)
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
