"""Inspect cache-versus-gap gradients offline and reuse an existing memorization result."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from _bootstrap import add_local_source_root

add_local_source_root()

from calibrate_value_scale import pair_arrays, terminal_arrays, verify_audited_head
from reconstruct_ranked_value import load_source_data, read_source
from polybot.infrastructure.device import select_device
from polybot.infrastructure.model_io import load_model
from polybot.training.cli import add_run_options, print_summary, training_run_from_args
from polybot.training.flat_experiments import model_fingerprint
from polybot.training.value_gradient_diagnostics import inspect_cache_gradients


def memorization_evidence(directory, source, model_config):
    """Verify the saved history, without repeating training or overstating its scope."""
    previous = read_source(directory)
    summary, config, arguments = (previous[key] for key in ("summary", "config", "args"))
    evidence = summary.get("memorization")
    if not evidence or not evidence.get("full_batch") or not arguments.get("memorization_test"):
        raise ValueError("--memorization-run must contain a completed full-batch memorization test")
    if config["model"] != json.loads(json.dumps(asdict(model_config))):
        raise ValueError("memorization test used a different network configuration")
    if Path(arguments["model_in"]).resolve() != Path(source["args"]["model_in"]).resolve():
        raise ValueError("memorization test refers to a different initial checkpoint path")
    history_path = directory / "delta" / "metrics.jsonl"
    history = [json.loads(line) for line in history_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    history = [row for row in history if "update" in row and "train_preference_accuracy" in row]
    if not history or len({row["update"] for row in history}) != len(history):
        raise ValueError("memorization history is missing or contains duplicate updates")
    accuracy = [float(row["train_preference_accuracy"]) for row in history]
    if not all(np.isfinite(value) and 0 <= value <= 1 for value in accuracy):
        raise ValueError("invalid memorization accuracy history")
    first = min((int(row["update"]) for row in history if row["train_preference_accuracy"] == 1), default=None)
    fit = summary["loss_comparison"]["delta"]
    if (first != evidence["first_perfect_training_update"]
            or not np.isclose(max(accuracy), evidence["max_training_preference_accuracy"])
            or max(row["update"] for row in history) != fit["updates"]
            or evidence["train_pairs"] != len(evidence["pair_ids"])):
        raise ValueError("memorization summary disagrees with its saved per-update history")
    old_hashes = config["source"]["source_sha256"]
    return {
        "run_dir": str(directory), "history_verified": True, **evidence,
        "train_gap_rmse_before": fit["train_before"]["gap_rmse"],
        "train_gap_rmse_last": fit["train_last"]["gap_rmse"],
        "selection_preference_accuracy_last": fit["selection_last"]["preference_accuracy"],
        "selection_value_rmse_last": fit["selection_last"]["rmse"],
        "same_pair_dataset": any(row["source_sha256"] == old_hashes
                                 for row in source["config"]["ranking_sources"]["sources"]),
        "scope": "historical learnability on these saved pairs; not proof of new-pair learnability or generalization",
        "checkpoint_identity": "recorded input path and architecture match; old test did not store a parameter fingerprint",
        "sha256": {name: sha256((directory / name).read_bytes()).hexdigest()
                   for name in ("args.json", "config.json", "summary.json", "delta/metrics.jsonl")},
    }


def run_diagnostics(args, model, opponent, source, run):
    model.eval().requires_grad_(False)
    opponent.eval().requires_grad_(False)
    data = load_source_data(source, model, opponent, device=args.device, calibration_cache=args.calibration_cache)
    memory = memorization_evidence(args.memorization_run, source, model.config)
    fingerprint = model_fingerprint(model)
    initial = deepcopy(model.value_head.state_dict())
    stages = {"initial": initial}
    if args.audit_run is not None:
        audit = read_source(args.audit_run)
        if (not audit["summary"].get("reconstruction_matches")
                or audit["config"]["source_config"] != source["config"]):
            raise ValueError("audit must verify a reconstructed head from the same source run")
        candidate, _ = load_model(Path(audit["summary"]["saved_checkpoint"]), next(model.parameters()).device)
        candidate.eval().requires_grad_(False)
        verify_audited_head(candidate, model, pair_arrays(candidate, data["ranking"]["selection"]),
                            terminal_arrays(candidate, data["calibration"]["selection"]),
                            audit["summary"], args.verification_tolerance)
        for name, cache in data["cache_reports"].items():
            if cache["sha256"] != audit["config"]["calibration"][name]["sha256"]:
                raise ValueError("cache changed after the reconstructed-head audit")
        stages[f"audited-{audit['summary']['update']}"] = deepcopy(candidate.value_head.state_dict())
        del candidate
    run.write_config(source_run=str(source["directory"]), source_config=source["config"],
        calibration=data["cache_reports"], initial_checkpoint_fingerprint=fingerprint,
        memorization=memory, audit_run=args.audit_run, batches=args.batches, probe_batches=args.probe_batches,
        gap_weight=1, branch_mse_weight=source["config"].get("pair_value_loss_weight", 0),
        cache_weight=args.calibration_weight, gradient_scope="training maps only", diagnostic_only=True)
    report = {"diagnostic_only": True, "new_games_collected": 0, "benchmark_performed": False,
              "memorization": memory, "stages": {}, "saved_checkpoint": None}
    model.value_head.requires_grad_(True)
    print(f"reused memorization | pairs={memory['train_pairs']} | "
          f"max_train_rank={memory['max_training_preference_accuracy']:.3f} | "
          f"first_perfect_update={memory['first_perfect_training_update']} | "
          f"same_pair_dataset={int(memory['same_pair_dataset'])}", flush=True)
    try:
        with (run.directory / "gradient_batches.jsonl").open("w", encoding="utf-8") as gradients, \
             (run.directory / "adamw_probes.jsonl").open("w", encoding="utf-8") as probes:
            for stage, state in stages.items():
                model.value_head.load_state_dict(state)
                measured, rows, updates = inspect_cache_gradients(
                    model, data["ranking"], data["calibration"], seed=data["training"].seed,
                    batch_size=source["args"]["batch_size"], learning_rate=source["args"]["learning_rate"],
                    pair_weight=source["config"].get("pair_value_loss_weight", 0), cache_weight=args.calibration_weight,
                    balance_pair_maps=data["weighted"], batches=args.batches, probe_batches=args.probe_batches)
                report["stages"][stage] = measured
                for stream, records in ((gradients, rows), (probes, updates)):
                    for row in records:
                        stream.write(json.dumps({"stage": stage, **row}, allow_nan=False) + "\n")
                    stream.flush()
                expected = measured["expected_phase_balanced_gradients"]
                print(f"value gradients | stage={stage} | gap_norm={expected['gap_norm']:.6g} | "
                      f"cache_norm={expected['weighted_cache_norm']:.6g} | "
                      f"cache/gap={expected['cache_to_gap_norm_ratio']} | cosine={expected['gap_cache_cosine']} | "
                      f"batch_conflicts={measured['minibatches']['cache_conflict_fraction']}", flush=True)
                run.log(stage_index=list(stages).index(stage), **expected,
                        cache_conflict_fraction=measured["minibatches"]["cache_conflict_fraction"])
    finally:
        model.value_head.load_state_dict(initial)
    report["weights_unchanged"] = model_fingerprint(model) == fingerprint
    if not report["weights_unchanged"]:
        raise RuntimeError("diagnostic failed to restore the input network")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_options(parser)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--memorization-run", type=Path, required=True)
    parser.add_argument("--audit-run", type=Path)
    parser.add_argument("--calibration-cache", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--calibration-weight", type=float, default=1.0)
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--probe-batches", type=int, default=8)
    parser.add_argument("--verification-tolerance", type=float, default=0.0001)
    parser.set_defaults(project=Path("runs/flat-11x11/logs"), name="flat-004-cache-gap-gradients")
    args = parser.parse_args()
    if (not 1 <= args.probe_batches <= args.batches or not np.isfinite(args.calibration_weight)
            or args.calibration_weight < 0 or not 0 < args.verification_tolerance < 1):
        parser.error("invalid gradient batch budget, weight or verification tolerance")
    inputs = {path.resolve() for path in (args.source_run, args.memorization_run, args.audit_run) if path is not None}
    if (args.project / "value" / args.name).resolve() in inputs:
        parser.error("diagnostic logs must not overwrite any input run")
    source = read_source(args.source_run)
    device = select_device(args.device)
    model, _ = load_model(Path(source["args"]["model_in"]), device)
    opponent, _ = load_model(Path(source["args"]["opponent"]), device)
    run = training_run_from_args(args, "value")
    report = run_diagnostics(args, model, opponent, source, run)
    report["run_dir"] = str(run.finish(report, plots=False))
    print_summary({
        "run_dir": report["run_dir"], "new_games_collected": 0, "weights_unchanged": report["weights_unchanged"],
        "memorization": {key: report["memorization"][key] for key in
                         ("train_pairs", "first_perfect_training_update", "max_training_preference_accuracy", "same_pair_dataset")},
        "stages": {name: {"expected_gradients": row["expected_phase_balanced_gradients"],
                          "minibatches": row["minibatches"],
                          "adamw_gap_change": {mode: {"mean_delta": probe["delta"]["train_gap_loss"]["mean"],
                                                      "worsened_fraction": probe["fraction_train_gap_worsened"]}
                                               for mode, probe in row["isolated_adamw_probes"].items()}}
                   for name, row in report["stages"].items()},
    })


if __name__ == "__main__":
    main()
