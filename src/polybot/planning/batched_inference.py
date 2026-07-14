"""One GPU inference worker shared by several CPU self-play actors."""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Event, RLock, Thread
from time import monotonic
from typing import Callable, Dict, List, Optional, Tuple


Evaluation = Tuple[Dict[int, float], float]
EvaluateFn = Callable[[List[Dict[str, object]], Optional[List[Dict[str, object] | None]]], List[Evaluation]]


@dataclass
class _Request:
    packets: List[Dict[str, object]]
    events: Optional[List[Dict[str, object] | None]]
    ready: Event = field(default_factory=Event)
    result: Optional[List[Evaluation]] = None
    error: Optional[BaseException] = None


class BatchedInference:
    """Collect leaf requests briefly, then evaluate them in one model call.

    Game environments and MCTS trees remain owned by their CPU actor threads.
    Only model access is centralised, which avoids CUDA races and lets leaves
    from independent games fill a useful GPU batch.
    """

    def __init__(self, evaluate: EvaluateFn, max_batch_size: int = 64, wait_ms: float = 2.0) -> None:
        self._evaluate = evaluate
        self.max_batch_size = max(1, int(max_batch_size))
        self.wait_seconds = max(0.0, float(wait_ms)) / 1_000.0
        self._requests: Queue[_Request | None] = Queue()
        self._model_lock = RLock()
        self._closed = False
        self._worker = Thread(target=self._run, name="polybot-gpu-inference", daemon=True)
        self._worker.start()

    def evaluate(
        self,
        packets: List[Dict[str, object]],
        events: Optional[List[Dict[str, object] | None]] = None,
    ) -> List[Evaluation]:
        if self._closed:
            raise RuntimeError("batched inference service is closed")
        request = _Request(packets, events)
        self._requests.put(request)
        request.ready.wait()
        if request.error is not None:
            raise request.error
        if request.result is None:
            raise RuntimeError("inference worker returned no result")
        return request.result

    def exclusive(self, operation: Callable[[], object]) -> object:
        """Run rare non-batchable GPU work (belief-particle sampling) safely."""
        with self._model_lock:
            return operation()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._requests.put(None)
            self._worker.join()

    def _run(self) -> None:
        while True:
            first = self._requests.get()
            if first is None:
                return
            requests = [first]
            packet_count = len(first.packets)
            deadline = monotonic() + self.wait_seconds
            while packet_count < self.max_batch_size:
                timeout = deadline - monotonic()
                if timeout <= 0:
                    break
                try:
                    request = self._requests.get(timeout=timeout)
                except Empty:
                    break
                if request is None:
                    # ``close`` is only called after actors finish, but keep
                    # this branch safe if shutdown races with a final batch.
                    self._closed = True
                    break
                requests.append(request)
                packet_count += len(request.packets)
            packets = [packet for request in requests for packet in request.packets]
            events = [
                event
                for request in requests
                for event in (request.events if request.events is not None else [None] * len(request.packets))
            ]
            try:
                with self._model_lock:
                    evaluations = self._evaluate(packets, events)
                offset = 0
                for request in requests:
                    size = len(request.packets)
                    request.result = evaluations[offset:offset + size]
                    offset += size
                    request.ready.set()
            except BaseException as error:
                for request in requests:
                    request.error = error
                    request.ready.set()
            if self._closed:
                return
