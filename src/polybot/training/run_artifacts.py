"""Ultralytics-style, incremented training-run artifacts and plots."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence


_AXIS_FIELDS = {
    "iteration", "step", "update", "game", "round", "position",
    "environment_step", "env_steps",
}


def _jsonable(value: Any) -> Any:
    """Convert configs, paths and NumPy-style scalar objects to JSON values."""
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def increment_path(base: Path, exist_ok: bool = False) -> Path:
    """Return ``train``, then ``train2``, ``train3`` without overwriting."""
    if exist_ok or not base.exists():
        return base
    index = 2
    while True:
        candidate = base.with_name(f"{base.name}{index}")
        if not candidate.exists():
            return candidate
        index += 1


MetricSink = Callable[[Path, Mapping[str, Any]], None]


class TrainingRun:
    """Collect lightweight scalar history and render final run artifacts."""

    def __init__(
        self,
        directory: Path,
        arguments: Mapping[str, Any] | None = None,
        metric_sink: MetricSink | None = None,
    ) -> None:
        self.directory = Path(directory)
        self._metric_sink = metric_sink
        self.directory.mkdir(parents=True, exist_ok=True)
        self._rows: list[Dict[str, Any]] = []
        self._jsonl_path = self.directory / "metrics.jsonl"
        self._events_path = self.directory / "events.jsonl"
        self._jsonl_path.write_text("", encoding="utf-8")
        self._events_path.write_text("", encoding="utf-8")
        metadata = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "directory": str(self.directory.resolve()),
            "arguments": _jsonable(dict(arguments or {})),
        }
        self._write_json("args.json", metadata)

    @classmethod
    def create(
        cls,
        project: Path | str,
        stage: str,
        name: str = "train",
        *,
        exist_ok: bool = False,
        arguments: Mapping[str, Any] | None = None,
    ) -> "TrainingRun":
        if not stage or not name:
            raise ValueError("run stage and name must not be empty")
        directory = increment_path(Path(project) / stage / name, exist_ok=exist_ok)
        return cls(directory, arguments=arguments)

    def child(self, stage: str, arguments: Mapping[str, Any] | None = None) -> "TrainingRun":
        """Create one named stage inside a unique parent pipeline/league run."""
        if not stage:
            raise ValueError("child run stage must not be empty")
        return TrainingRun(
            self.directory / stage,
            arguments=arguments,
            metric_sink=self._metric_sink,
        )

    def set_metric_sink(self, sink: MetricSink | None) -> None:
        """Mirror future scalar rows to an external store such as SQLite."""
        self._metric_sink = sink

    def write_config(self, **configs: Any) -> None:
        self._write_json("config.json", configs)

    def log(self, **metrics: Any) -> None:
        """Append one scalar snapshot without retaining games or model inputs."""
        row: Dict[str, Any] = {"iteration": len(self._rows) + 1}
        for name, value in metrics.items():
            number = _numeric(value)
            if number is not None:
                row[str(name)] = number
        self._rows.append(row)
        with self._jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        if self._metric_sink is not None:
            self._metric_sink(self.directory, row)

    def event(self, phase: str, **details: Any) -> None:
        """Record a textual training decision without polluting scalar charts."""
        payload = {
            "iteration": len(self._rows),
            "phase": str(phase),
            **_jsonable(details),
        }
        with self._events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")

    def finish(self, summary: Mapping[str, Any]) -> Path:
        """Write final CSV/JSON and PNG plots; return the absolute run path."""
        final_summary = dict(_jsonable(dict(summary)))
        final_summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        final_summary["run_dir"] = str(self.directory.resolve())
        self._write_json("summary.json", final_summary)
        self._write_csv()
        self._write_plots()
        return self.directory.resolve()

    def _write_json(self, filename: str, payload: Any) -> None:
        (self.directory / filename).write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def _write_csv(self) -> None:
        fields: list[str] = []
        for row in self._rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        if not fields:
            fields = ["iteration"]
        with (self.directory / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self._rows)

    def _write_plots(self) -> None:
        series = self._series_names()
        if not series:
            return
        # Use one writable cache shared by successive runs instead of a user
        # home directory that may be read-only on a training server. Reusing it
        # also avoids rebuilding matplotlib's font cache for every train2/3.
        plot_cache = Path(tempfile.gettempdir()) / "polybot-matplotlib-cache"
        plot_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(plot_cache.resolve()))
        os.environ.setdefault("XDG_CACHE_HOME", str((plot_cache / "xdg").resolve()))
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        plt.style.use("seaborn-v0_8-whitegrid")
        self._plot_grid(plt, series, self.directory / "results.png", "Training results")
        losses = [name for name in series if "loss" in name.lower()]
        metrics = [name for name in series if name not in losses]
        if losses:
            self._plot_grid(plt, losses, self.directory / "losses.png", "Loss curves")
        if metrics:
            self._plot_grid(plt, metrics, self.directory / "metrics.png", "Training metrics")
        plt.close("all")

    def _series_names(self) -> list[str]:
        names: list[str] = []
        for row in self._rows:
            for key, value in row.items():
                if key not in _AXIS_FIELDS and _numeric(value) is not None and key not in names:
                    names.append(key)
        return names

    def _plot_grid(self, plt: Any, names: Sequence[str], path: Path, title: str) -> None:
        columns = min(3, max(1, len(names)))
        rows = int(math.ceil(len(names) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(6.0 * columns, 3.8 * rows), squeeze=False)
        figure.suptitle(title, fontsize=16)
        for axis, name in zip(axes.flat, names):
            points = [
                (int(row["iteration"]), float(row[name]))
                for row in self._rows
                if name in row and _numeric(row[name]) is not None
            ]
            x = [point[0] for point in points]
            y = [point[1] for point in points]
            axis.plot(x, y, color="#9bb9df", linewidth=1.0, alpha=0.75, label="raw")
            window = min(31, max(1, len(y) // 20 * 2 + 1))
            if window >= 3:
                smoothed = _moving_average(y, window)
                axis.plot(x, smoothed, color="#1557a0", linewidth=2.0, label=f"smooth ({window})")
                axis.legend(loc="best", fontsize=8)
            axis.set_title(name.replace("_", " "))
            axis.set_xlabel("logged iteration")
        for axis in list(axes.flat)[len(names):]:
            axis.set_visible(False)
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)


def _moving_average(values: Sequence[float], window: int) -> list[float]:
    """Centered moving average without a NumPy dependency."""
    if window <= 1:
        return list(values)
    radius = window // 2
    result: list[float] = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        result.append(sum(values[start:end]) / (end - start))
    return result
