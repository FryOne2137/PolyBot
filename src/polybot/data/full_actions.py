"""Exact native action packets: reserve the ActionSpace, trim only padding.

PolyEnv's vector/pool capacities are immutable. Reserving its stable complete
ActionSpace is the only lossless bound supported without rebuilding live games.
The internal reusable buffers keep that bound; external/model packets contain
only the largest actually present action row. No state is replayed or reset.
"""
from __future__ import annotations

from functools import lru_cache
from collections import OrderedDict
from typing import Any, Mapping

import numpy as np

ACTION_ARRAYS = frozenset(("action_features", "action_mask", "action_id", "policy", "visit_count"))


def require_complete_actions(batch: Mapping[str, np.ndarray], rows: int | None = None) -> None:
    """Refuse partial input, including before any optimization or evaluation."""
    if "action_mask" not in batch:
        return
    mask = np.asarray(batch["action_mask"])
    count = mask.shape[0] if rows is None else rows
    if "action_truncated" in batch and np.any(np.asarray(batch["action_truncated"])[:count]):
        raise RuntimeError("incomplete legal action packet: action_truncated is set")
    if "total_legal_action_count" in batch:
        total = np.asarray(batch["total_legal_action_count"])[:count]
        supplied = mask[:count].astype(bool).sum(axis=-1)
        if np.any(total != supplied):
            raise RuntimeError("incomplete legal action packet: supplied and total action counts differ")


def compact_actions(batch: Mapping[str, np.ndarray], *, rows: int | None = None,
                    copy: bool = False) -> dict[str, np.ndarray]:
    """Remove trailing padding, retaining action order and every valid entry."""
    require_complete_actions(batch, rows)
    width = None
    if "action_mask" in batch:
        valid = np.asarray(batch["action_mask"], dtype=bool)
        valid = valid if rows is None else valid[:rows]
        columns = np.flatnonzero(valid.any(axis=0))
        width = max(1, int(columns[-1]) + 1) if columns.size else 1
    result = {}
    for name, value in batch.items():
        array = np.asarray(value)
        if name in ACTION_ARRAYS and width is not None and array.ndim >= 2:
            array = array[:, :width]
        result[name] = array.copy() if copy else array
    return result


@lru_cache(maxsize=32)
def action_space_capacity(map_size: int) -> int:
    from PolyEnv import GameEnv
    # Stable IDs are determined by board size, not the actual hidden map.
    return int(GameEnv(seed=2137, map_size=int(map_size)).action_space_size())


def _allocate(spec):
    return {name: np.empty(tuple(field["shape"]), dtype=np.dtype(field["dtype"]))
            for name, field in spec.items()}


class ExactActionNative:
    """Reuse native full-capacity buffers behind compact, complete packets."""

    def __init__(self, native: Any):
        self.native = native
        self._buffers: OrderedDict[tuple, dict[str, np.ndarray]] = OrderedDict()
        self.peak_legal_actions = 0
        self.complete_action_packets = 0

    def __getattr__(self, name):
        return getattr(self.native, name)

    def _raw(self, method, *args):
        key = (method, *args)
        if method == "leaf_batch_spec" and key not in self._buffers:
            # A full-space leaf buffer is large. Forest activity changes the
            # requested row capacity; caching every historical size would grow
            # host RAM over a long run. Two LRU sizes cover root/leaf waves.
            leaf_keys = [item for item in self._buffers if item[0] == method]
            if len(leaf_keys) >= 2:
                del self._buffers[leaf_keys[0]]
        if key not in self._buffers:
            self._buffers[key] = _allocate(getattr(self.native, method)(*args))
        self._buffers.move_to_end(key)
        return self._buffers[key]

    @staticmethod
    def _spec(spec):
        # External buffers are allowed to grow on observe/step. Native calls
        # validate their own full-sized internal packet instead of these views.
        result = {}
        for name, field in spec.items():
            field = dict(field)
            if name in ACTION_ARRAYS and len(field["shape"]) >= 2:
                field["shape"] = (*field["shape"][:1], 1, *field["shape"][2:])
            result[name] = field
        return result

    def _write(self, destination, source, rows=None):
        compact = compact_actions(source, rows=rows)
        if "action_mask" in compact:
            mask = compact["action_mask"] if rows is None else compact["action_mask"][:rows]
            self.peak_legal_actions = max(self.peak_legal_actions, int(mask.sum(axis=-1).max(initial=0)))
            self.complete_action_packets += 1
        for name, array in compact.items():
            target = destination.get(name)
            if target is not None and target.shape == array.shape and target.dtype == array.dtype:
                np.copyto(target, array)
            else:
                destination[name] = array.copy()
        return destination

    def batch_spec(self):
        return self._spec(self.native.batch_spec())

    def belief_batch_spec(self):
        return self._spec(self.native.belief_batch_spec())

    def leaf_batch_spec(self, max_leaves=0):
        return self._spec(self.native.leaf_batch_spec(max_leaves))

    def root_policy_spec(self):
        return self._spec(self.native.root_policy_spec())

    def reset_into(self, buffers, **kwargs):
        if not hasattr(self.native, "reset_into"):
            self._write(buffers, self.native.reset(**kwargs))
            return
        spec = "belief_batch_spec" if hasattr(self.native, "belief_batch_spec") else "batch_spec"
        raw = self._raw(spec)
        self.native.reset_into(raw, **kwargs)
        self._write(buffers, raw)

    def reset(self, **kwargs):
        result = {}
        self.reset_into(result, **kwargs)
        return result

    def step_into(self, action_ids, buffers):
        if not hasattr(self.native, "step_into"):
            self._write(buffers, self.native.step(action_ids))
            return
        spec = "belief_batch_spec" if hasattr(self.native, "belief_batch_spec") else "batch_spec"
        raw = self._raw(spec)
        self.native.step_into(action_ids, raw)
        self._write(buffers, raw)

    def step(self, action_ids):
        result = {}
        self.step_into(action_ids, result)
        return result

    def step_into_checked(self, action_ids, state_ids, buffers):
        spec = "belief_batch_spec" if hasattr(self.native, "belief_batch_spec") else "batch_spec"
        raw = self._raw(spec)
        self.native.step_into_checked(action_ids, state_ids, raw)
        self._write(buffers, raw)

    def step_checked(self, action_ids, state_ids):
        result = {}
        self.step_into_checked(action_ids, state_ids, result)
        return result

    def belief_requests_into(self, buffers):
        raw = self._raw("belief_batch_spec")
        self.native.belief_requests_into(raw)
        self._write(buffers, raw)

    def belief_requests(self):
        result = {}
        self.belief_requests_into(result)
        return result

    def reset_slots_into(self, env_ids, seeds, agent_ids, buffers):
        raw = self._raw("belief_batch_spec")
        self.native.reset_slots_into(env_ids, seeds, agent_ids, raw)
        self._write(buffers, raw)

    def select_leaves_into(self, buffers, max_leaves=0):
        raw = self._raw("leaf_batch_spec", max_leaves)
        count = self.native.select_leaves_into(raw, max_leaves=max_leaves)
        self._write(buffers, raw, rows=count)
        return count

    def select_leaves(self, max_leaves=0):
        result = {}
        count = self.select_leaves_into(result, max_leaves)
        return {name: value[:count] for name, value in result.items()}

    def select_leaves_masked_into(self, buffers, search_mask, max_leaves=None):
        limit = 0 if max_leaves is None else max_leaves
        raw = self._raw("leaf_batch_spec", limit)
        count = self.native.select_leaves_masked_into(raw, search_mask, max_leaves=max_leaves)
        self._write(buffers, raw, rows=count)
        return count

    def select_leaves_masked(self, search_mask, max_leaves=None):
        result = {}
        count = self.select_leaves_masked_into(result, search_mask, max_leaves)
        return {name: value[:count] for name, value in result.items()}

    def expand_and_backup(self, leaf_ids, policy_logits, values):
        logits = np.asarray(policy_logits, dtype=np.float32)
        if logits.shape[1] != self.native.max_actions:
            padded = np.full((len(leaf_ids), self.native.max_actions), -np.inf, dtype=np.float32)
            padded[:, :logits.shape[1]] = logits
            logits = padded
        self.native.expand_and_backup(leaf_ids, logits, values)

    def root_policy_into(self, buffers, temperature=1.0):
        raw = self._raw("root_policy_spec")
        self.native.root_policy_into(raw, temperature=temperature)
        self._write(buffers, raw)

    def root_policy(self, temperature=1.0):
        result = {}
        self.root_policy_into(result, temperature)
        return result


def exact_vector_env(native_type, **kwargs):
    kwargs["max_actions"] = max(int(kwargs.get("max_actions", 1)),
                                action_space_capacity(int(kwargs.get("map_size", 11))))
    return ExactActionNative(native_type(**kwargs))


def exact_self_play_pool(native_type, **kwargs):
    kwargs["max_actions"] = 0  # documented full stable ActionSpace mode
    kwargs["require_all_actions"] = True
    return ExactActionNative(native_type(**kwargs))


def exact_mcts_pool(native_type, roots, **kwargs):
    root = roots[0] if isinstance(roots, (list, tuple)) else roots
    kwargs["max_actions"] = max(int(kwargs.get("max_actions", 1)), int(root.action_space_size()))
    return ExactActionNative(native_type(roots, **kwargs))


def merge_action_batches(batches):
    """Pad only the action axis when independent native pools are batched."""
    width = max(batch["action_mask"].shape[1] for batch in batches)
    result = {}
    for name in batches[0]:
        arrays = []
        for batch in batches:
            value = np.asarray(batch[name])
            if name in ACTION_ARRAYS and value.ndim >= 2 and value.shape[1] != width:
                padding = [(0, 0), (0, width-value.shape[1]), *[(0, 0)]*(value.ndim-2)]
                value = np.pad(value, padding, constant_values=0 if name == "action_mask" else -1)
            arrays.append(value)
        result[name] = np.concatenate(arrays, axis=0)
    return result
