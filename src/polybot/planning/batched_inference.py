"""One GPU inference worker shared by several CPU self-play actors."""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Event, Lock, RLock, Thread
from time import monotonic
from typing import Callable, Dict, List, Optional, Tuple


Evaluation = Tuple[Dict[int, float], float]
EvaluateFn = Callable[[List[Dict[str, object]], Optional[List[Dict[str, object] | None]]], List[Evaluation]]
NativeEvaluateFn = Callable[[Dict[str, object]], Tuple[object, object]]


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

    def __init__(self, evaluate: EvaluateFn, max_batch_size: int = 64, wait_ms: float = 0.25) -> None:
        self._evaluate = evaluate
        self.max_batch_size = max(1, int(max_batch_size))
        self.wait_seconds = max(0.0, float(wait_ms)) / 1_000.0
        self._requests: Queue[_Request | None] = Queue()
        self._model_lock = RLock()
        self._stats_lock = Lock()
        self._batches = 0
        self._packets = 0
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

    def stats(self) -> Dict[str, float]:
        """Return compact in-memory throughput diagnostics for terminal output."""
        with self._stats_lock:
            return {
                "inference_batches": float(self._batches),
                "inference_packets": float(self._packets),
                "mean_inference_batch_size": self._packets / max(1, self._batches),
            }

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
                with self._stats_lock:
                    self._batches += 1
                    self._packets += len(packets)
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


@dataclass
class _NativeRequest:
    leaves: Dict[str, object]
    ready: Event = field(default_factory=Event)
    result: Optional[Tuple[object, object]] = None
    error: Optional[BaseException] = None


class NativeLeafInference:
    """Batch dense leaves emitted by several native ``MctsPool`` instances.

    MctsPool performs tree work in C++, but emits at most one pending leaf per
    tree. This service combines those small per-game batches before the model
    call, restoring a large GPU batch without moving tree control into Python.
    """

    def __init__(self, evaluate: NativeEvaluateFn, max_batch_size: int = 256, wait_ms: float = 1.0) -> None:
        self._evaluate = evaluate
        self.max_batch_size = max(1, int(max_batch_size))
        self.wait_seconds = max(0.0, float(wait_ms)) / 1_000.0
        self._requests: Queue[_NativeRequest | None] = Queue()
        self._stats_lock = Lock()
        self._batches = 0
        self._leaves = 0
        self._closed = False
        self._worker = Thread(target=self._run, name="polybot-native-leaf-inference", daemon=True)
        self._worker.start()

    def evaluate(self, leaves: Dict[str, object]) -> Tuple[object, object]:
        if self._closed:
            raise RuntimeError("native leaf inference service is closed")
        request = _NativeRequest(leaves)
        self._requests.put(request)
        request.ready.wait()
        if request.error is not None:
            raise request.error
        if request.result is None:
            raise RuntimeError("native leaf inference service returned no result")
        return request.result

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._requests.put(None)
            self._worker.join()

    def stats(self) -> Dict[str, float]:
        with self._stats_lock:
            return {
                "native_inference_batches": float(self._batches),
                "native_inference_leaves": float(self._leaves),
                "native_mean_inference_batch_size": self._leaves / max(1, self._batches),
            }

    def _run(self) -> None:
        while True:
            first = self._requests.get()
            if first is None:
                return
            requests = [first]
            leaf_count = len(first.leaves["leaf_id"])
            deadline = monotonic() + self.wait_seconds
            while leaf_count < self.max_batch_size:
                timeout = deadline - monotonic()
                if timeout <= 0:
                    break
                try:
                    request = self._requests.get(timeout=timeout)
                except Empty:
                    break
                if request is None:
                    self._closed = True
                    break
                requests.append(request)
                leaf_count += len(request.leaves["leaf_id"])
            try:
                # Independent pools can have different complete action counts.
                # Pad their action axes, retaining each request's exact order.
                from ..data.full_actions import merge_action_batches
                merged = merge_action_batches([request.leaves for request in requests])
                logits, values = self._evaluate(merged)
                with self._stats_lock:
                    self._batches += 1
                    self._leaves += leaf_count
                offset = 0
                for request in requests:
                    size = len(request.leaves["leaf_id"])
                    width = request.leaves["action_mask"].shape[1]
                    request.result = (logits[offset:offset + size, :width], values[offset:offset + size])
                    offset += size
                    request.ready.set()
            except BaseException as error:
                for request in requests:
                    request.error = error
                    request.ready.set()
            if self._closed:
                return
