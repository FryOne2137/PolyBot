"""CUDA inference RPC for CPU self-play processes.

Only the parent process owns the PyTorch CUDA model.  Each actor process owns
its PolyEnv instances and MCTS trees, then requests policy/value or particle
inference through multiprocessing queues.  This avoids the GIL bottleneck
without duplicating CUDA model memory per actor.
"""

from __future__ import annotations

from collections import deque
from queue import Empty
from threading import Thread
from time import monotonic
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple


Evaluation = Tuple[Dict[int, float], float]


class ProcessInferenceClient:
    """Blocking inference client used inside exactly one actor process."""

    def __init__(self, actor_id: int, requests: Any, responses: Any) -> None:
        self.actor_id = actor_id
        self._requests = requests
        self._responses = responses

    def evaluate(
        self,
        packets: List[Dict[str, object]],
        events: Optional[List[Dict[str, object] | None]] = None,
    ) -> List[Evaluation]:
        self._requests.put(("evaluate", self.actor_id, packets, events))
        status, payload = self._responses.get()
        if status != "ok":
            raise RuntimeError("GPU inference failed in parent process: " + str(payload))
        return payload

    def sample_particles(
        self,
        packet: Dict[str, object],
        events: Optional[Dict[str, object]],
        particles: int,
        temperature: float,
    ) -> List[Any]:
        self._requests.put(("particles", self.actor_id, packet, events, particles, temperature))
        status, payload = self._responses.get()
        if status != "ok":
            raise RuntimeError("GPU particle sampling failed in parent process: " + str(payload))
        return payload


class ProcessInferenceServer:
    """Parent-side inference loop batching requests from actor processes."""

    def __init__(
        self,
        requests: Any,
        responses: List[Any],
        evaluate: Callable[[List[Dict[str, object]], Optional[List[Dict[str, object] | None]]], List[Evaluation]],
        sample_particles: Callable[[Dict[str, object], Optional[Dict[str, object]], int, float], List[Any]],
        max_batch_size: int = 128,
        wait_ms: float = 0.25,
    ) -> None:
        self._requests = requests
        self._responses = responses
        self._evaluate = evaluate
        self._sample_particles = sample_particles
        self.max_batch_size = max(1, int(max_batch_size))
        self.wait_seconds = max(0.0, float(wait_ms)) / 1_000.0
        self._pending: Deque[tuple] = deque()
        self._batches = 0
        self._packets = 0
        self._thread = Thread(target=self._run, name="polybot-process-gpu-inference", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._requests.put(("stop",))
        self._thread.join()

    def stats(self) -> Dict[str, float]:
        return {
            "inference_batches": float(self._batches),
            "inference_packets": float(self._packets),
            "mean_inference_batch_size": self._packets / max(1, self._batches),
        }

    def _next(self, timeout: Optional[float] = None) -> tuple:
        if self._pending:
            return self._pending.popleft()
        if timeout is None:
            return self._requests.get()
        return self._requests.get(timeout=timeout)

    def _reply(self, actor_id: int, status: str, payload: Any) -> None:
        self._responses[actor_id].put((status, payload))

    def _run(self) -> None:
        while True:
            request = self._next()
            if request[0] == "stop":
                return
            if request[0] == "particles":
                _, actor_id, packet, events, count, temperature = request
                try:
                    self._reply(actor_id, "ok", self._sample_particles(packet, events, count, temperature))
                except BaseException as error:
                    self._reply(actor_id, "error", repr(error))
                continue

            requests = [request]
            packet_count = len(request[2])
            deadline = monotonic() + self.wait_seconds
            while packet_count < self.max_batch_size:
                timeout = deadline - monotonic()
                if timeout <= 0:
                    break
                try:
                    candidate = self._next(timeout)
                except Empty:
                    break
                if candidate[0] != "evaluate":
                    self._pending.append(candidate)
                    break
                requests.append(candidate)
                packet_count += len(candidate[2])
            packets = [packet for request in requests for packet in request[2]]
            events = [
                event
                for request in requests
                for event in (request[3] if request[3] is not None else [None] * len(request[2]))
            ]
            try:
                evaluations = self._evaluate(packets, events)
                self._batches += 1
                self._packets += len(packets)
                offset = 0
                for request in requests:
                    actor_id, size = request[1], len(request[2])
                    self._reply(actor_id, "ok", evaluations[offset:offset + size])
                    offset += size
            except BaseException as error:
                for request in requests:
                    self._reply(request[1], "error", repr(error))
