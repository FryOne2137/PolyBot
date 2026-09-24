from __future__ import annotations

import csv
import json

from polybot.training.run_artifacts import TrainingRun


def test_training_run_increments_and_writes_metrics_and_plots(tmp_path):
    first = TrainingRun.create(tmp_path, "belief", arguments={"positions": 8})
    for step in range(1, 9):
        first.log(step=step, belief_loss=1.0 / step, hidden_tiles=20 - step)
    output = first.finish({"positions": 8, "belief_loss": 0.125})

    second = TrainingRun.create(tmp_path, "belief")

    assert output.name == "train"
    assert second.directory.name == "train2"
    assert (output / "results.png").is_file()
    assert (output / "losses.png").is_file()
    assert (output / "metrics.png").is_file()
    with (output / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert rows[-1]["belief_loss"] == "0.125"
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["positions"] == 8
    assert summary["run_dir"] == str(output.resolve())
