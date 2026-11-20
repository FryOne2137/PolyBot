"""Compare branch-MSE coefficients offline, keeping gap and terminal-cache losses fixed."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from _bootstrap import add_local_source_root

add_local_source_root()

from reconstruct_ranked_value import _recorded_row, compare_metrics, gap_audit, load_source_data, read_source
from polybot.infrastructure.device import select_device
from polybot.infrastructure.model_io import load_model, save_model
from polybot.training.calibrated_value import calibration_admission, fit_calibrated_heads
from polybot.training.cli import add_run_options, print_summary, training_run_from_args
from polybot.training.flat_experiments import FlatImprovements, model_fingerprint


def _gap_key(metrics):
    """All weights use the same held-out criterion, including the initial head."""
    return metrics["rank"]["gap_rmse"], metrics["calibration"]["rmse"]


def _finite(metrics):
    return all(np.isfinite(value) for value in _gap_key(metrics))


def run_experiment(args, model, opponent, source, run):
    configuration, previous = source["config"], source["args"]
    control = configuration.get("pair_value_loss_weight", 0.0)
    if (control not in args.pair_mse_weights or args.calibration_weight not in configuration["calibration_weights"]
            or not 1 <= args.control_update <= min(args.updates, previous["updates"])):
        raise ValueError("include the source branch-MSE coefficient and a control update from its original run")
    recorded = {step: _recorded_row(source["directory"], args.calibration_weight, step)
                for step in (0, args.control_update)}
    # This loader requires existing caches and the original checkpoint/dataset
    # identities. A missing cache cannot trigger a new collection.
    model.eval().requires_grad_(False)
    opponent.eval().requires_grad_(False)
    data = load_source_data(source, model, opponent, device=args.device, calibration_cache=args.calibration_cache)
    initial = deepcopy(model.value_head.state_dict())
    fingerprint = model_fingerprint(model)
    model.value_head.requires_grad_(True)
    settings = FlatImprovements(
        value_updates=args.updates, value_batch_size=previous["batch_size"],
        value_learning_rate=previous["learning_rate"], value_fit_backoffs=0,
        value_capacity=max(previous["batch_size"], len(data["calibration"]["train"])),
    )
    run.write_config(
        source_run=str(source["directory"]), source_config=configuration, calibration=data["cache_reports"],
        checkpoint_fingerprint=fingerprint, pair_mse_weights=args.pair_mse_weights,
        gap_loss_weight=1.0, calibration_weight=args.calibration_weight, updates=args.updates,
        batch_size=settings.value_batch_size, learning_rate=settings.value_learning_rate,
        seed=data["training"].seed, control_update=args.control_update, control_pair_mse_weight=control,
        diagnostic_only=True, selection_rule="original calibration admission; ranking accuracy, gap RMSE, value RMSE, bias",
    )
    report = {
        "diagnostic_only": True, "benchmark_performed": False, "new_games_collected": 0,
        "source_run": str(source["directory"]), "saved_checkpoint": None, "selected": None,
        "control_matches": False, "variants": {},
        "fixed_loss_weights": {"gap": 1.0, "terminal_cache_mse": args.calibration_weight},
        "selection_note": "reused selection maps; diagnostic checkpoint needs an independent game benchmark",
    }
    heads = {"original": initial, "zero": None}
    winner = winner_state = None
    # Verify the known control before spending updates on the other coefficients.
    order = (control, *(weight for weight in args.pair_mse_weights if weight != control))
    try:
        for coefficient in order:
            model.value_head.load_state_dict(initial)
            child = run.child(f"pair-mse-{coefficient!r}")
            child.write_config(pair_mse_weight=coefficient, gap_loss_weight=1.0,
                               calibration_weight=args.calibration_weight, settings=settings)
            captures, best_gap, last_state, last_step = {}, None, None, 0

            def capture(weight, step, metrics, state):
                nonlocal best_gap, last_state, last_step
                last_step = step
                last_state = deepcopy(state)
                if step in (0, args.control_update, args.updates):
                    captures[step] = {"update": step, **deepcopy(metrics)}
                if _finite(metrics) and (best_gap is None or _gap_key(metrics) < _gap_key(best_gap["metrics"])):
                    best_gap = {"update": step, "metrics": deepcopy(metrics), "state": deepcopy(state)}

            state, fitted = fit_calibrated_heads(
                model, data["ranking"], data["calibration"], data["training"], settings,
                weights=(args.calibration_weight,), pair_value_weight=coefficient, balance_pair_maps=data["weighted"],
                rmse_regression=configuration["rmse_regression"], bias_regression=configuration["bias_regression"],
                run=child, log_interval=args.log_interval, update_callback=capture,
            )
            if coefficient == control:
                checks = {str(step): compare_metrics(recorded[step], captures[step], args.reproduction_tolerance)
                          for step in recorded if step in captures}
                report["control_checks"] = checks
                report["control_matches"] = len(checks) == 2 and all(row["matches"] for row in checks.values())
                if not report["control_matches"]:
                    return {**report, "reason": "control_reproduction_mismatch"}
            reference, selected = fitted["reference"], fitted["selected"]
            last = fitted["variants"][str(args.calibration_weight)]["last"]
            # The callback records every update at the same selection criterion;
            # these checkpoints are diagnostic, not automatic promotions.
            variant = {
                "pair_mse_weight": coefficient, "before": reference, "last": last,
                "attempted_updates": last_step, "selected": selected,
                "rejections": fitted["variants"][str(args.calibration_weight)]["rejections"],
                "best_gap": None,
            }
            heads[f"pair-{coefficient!r}-last"] = last_state
            if best_gap is not None:
                failures = calibration_admission(
                    best_gap["metrics"]["rank"], best_gap["metrics"]["calibration"],
                    reference["rank"], reference["calibration"],
                    rmse_regression=configuration["rmse_regression"], bias_regression=configuration["bias_regression"],
                )
                variant["best_gap"] = {"update": best_gap["update"], **best_gap["metrics"], "gate_failures": failures}
                heads[f"pair-{coefficient!r}-best-gap"] = best_gap["state"]
                torch.save({"value_head": best_gap["state"], "input_checkpoint_fingerprint": fingerprint,
                            "pair_mse_weight": coefficient, "update": best_gap["update"], "diagnostic_only": True},
                           child.directory / "best-gap-head.pt")
            if state is not None:
                heads[f"pair-{coefficient!r}-admitted"] = state
                key = (-selected["rank"]["preference_accuracy"], selected["rank"]["gap_rmse"],
                       selected["calibration"]["rmse"], abs(selected["calibration"]["bias"]))
                if winner is None or key < winner[0]:
                    winner, winner_state = (key, {"pair_mse_weight": coefficient, **deepcopy(selected)}), deepcopy(state)
                torch.save({"value_head": state, "input_checkpoint_fingerprint": fingerprint,
                            "pair_mse_weight": coefficient, "update": selected["update"], "diagnostic_only": True},
                           child.directory / "admitted-head.pt")
            report["variants"][str(coefficient)] = variant
            child.finish(variant, plots=False)
            print(f"pair MSE comparison | weight={coefficient:g} | "
                  f"train_gap={last['train_rank']['gap_rmse']:.4f} | "
                  f"selection_gap={last['rank']['gap_rmse']:.4f} | "
                  f"rank={last['rank']['preference_accuracy']} | "
                  f"value_rmse={last['calibration']['rmse']:.4f} | admitted={int(state is not None)}", flush=True)
            run.log(pair_mse_weight=coefficient, last_train_gap_rmse=last["train_rank"]["gap_rmse"],
                    last_selection_gap_rmse=last["rank"]["gap_rmse"],
                    last_preference_accuracy=last["rank"]["preference_accuracy"],
                    last_value_rmse=last["calibration"]["rmse"], admitted=state is not None)
            run.finish({**report, "reason": "sweep_in_progress"}, plots=False)
        report["gap_diagnostics"] = gap_audit(model, data["ranking"], heads, run.directory / "gap_pairs.jsonl")
        if winner is not None:
            model.value_head.load_state_dict(winner_state)
            output = args.model_out or run.directory / "candidate.pt"
            save_model(output, model, model.config)
            report.update(selected=winner[1], saved_checkpoint=str(output))
        return {**report, "reason": "offline_comparison_complete"}
    finally:
        model.value_head.load_state_dict(initial)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_options(parser)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--calibration-cache", type=Path)
    parser.add_argument("--model-out", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--pair-mse-weights", default="0,0.01,0.05,0.1,1.0")
    parser.add_argument("--calibration-weight", type=float, default=1.0)
    parser.add_argument("--updates", type=int, default=512)
    parser.add_argument("--control-update", type=int, default=448)
    parser.add_argument("--reproduction-tolerance", type=float, default=0.0001)
    parser.add_argument("--log-interval", type=int, default=64)
    parser.set_defaults(project=Path("runs/flat-11x11/logs"), name="flat-004-pair-mse-offline")
    args = parser.parse_args()
    try:
        args.pair_mse_weights = tuple(float(value) for value in args.pair_mse_weights.split(","))
    except ValueError:
        parser.error("--pair-mse-weights requires comma-separated numbers")
    if (not args.pair_mse_weights or len(set(args.pair_mse_weights)) != len(args.pair_mse_weights)
            or any(not np.isfinite(value) or value < 0 for value in args.pair_mse_weights)
            or not np.isfinite(args.calibration_weight) or args.calibration_weight < 0
            or not np.isfinite(args.reproduction_tolerance) or args.reproduction_tolerance <= 0
            or args.updates < 1 or not 1 <= args.control_update <= args.updates or args.log_interval < 1):
        parser.error("invalid loss coefficients, update budget or reproduction tolerance")
    source = read_source(args.source_run)
    model_path, opponent_path = (Path(source["args"][key]) for key in ("model_in", "opponent"))
    if (args.project / "value" / args.name).resolve() == args.source_run.resolve():
        parser.error("comparison logs must not overwrite the source run")
    if args.model_out is not None and (args.model_out.exists() or args.model_out.resolve() in
                                     {model_path.resolve(), opponent_path.resolve()}):
        parser.error("--model-out must be a new path distinct from both original checkpoints")
    device = select_device(args.device)
    model, _ = load_model(model_path, device)
    opponent, _ = load_model(opponent_path, device)
    run = training_run_from_args(args, "value")
    report = run_experiment(args, model, opponent, source, run)
    report["run_dir"] = str(run.finish(report, plots=False))
    # Full per-phase metrics and curves stay in the artifacts. Keep the console
    # comparison small enough to paste without losing its final selection.
    console = {key: report[key] for key in ("reason", "control_matches", "new_games_collected",
                                          "benchmark_performed", "saved_checkpoint", "run_dir")}
    console["variants"] = {
        weight: {
            "attempted_updates": row["attempted_updates"],
            "last_train_gap_rmse": row["last"]["train_rank"]["gap_rmse"],
            "last_selection_gap_rmse": row["last"]["rank"]["gap_rmse"],
            "last_preference_accuracy": row["last"]["rank"]["preference_accuracy"],
            "last_value_rmse": row["last"]["calibration"]["rmse"],
            "best_gap_update": row["best_gap"]["update"] if row["best_gap"] else None,
            "best_gap_rmse": row["best_gap"]["rank"]["gap_rmse"] if row["best_gap"] else None,
            "best_gap_gate_failures": row["best_gap"]["gate_failures"] if row["best_gap"] else None,
            "admitted_update": row["selected"]["update"] if row["selected"] else None,
        } for weight, row in report["variants"].items()
    }
    if report["selected"]:
        console["selected"] = {key: report["selected"][key] for key in ("pair_mse_weight", "update")}
    print_summary(console)
    if not report["control_matches"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
