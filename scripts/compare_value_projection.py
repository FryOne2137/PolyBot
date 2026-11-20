"""Compare ordinary value-loss sums with cache projection and a transactional AdamW guard."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from _bootstrap import add_local_source_root

add_local_source_root()

from reconstruct_ranked_value import _recorded_row, compare_metrics, gap_audit, load_source_data, read_source
from polybot.infrastructure.device import select_device
from polybot.infrastructure.model_io import load_model, save_model
from polybot.training.calibrated_value import fit_calibrated_heads
from polybot.training.cli import add_run_options, print_summary, training_run_from_args
from polybot.training.flat_experiments import FlatImprovements, model_fingerprint
from polybot.training.value_gradient_guard import GapGuardConfig


def projection_reference(directory, source, data, settings, variant, state, tolerance):
    previous = read_source(directory)
    config, summary = previous["config"], previous["summary"]
    if (config["source_config"] != source["config"] or not summary.get("control_matches")
            or config["checkpoint_fingerprint"] != source["config"]["checkpoint_fingerprints"]["model"]):
        raise ValueError("projection reference uses another source run or an unverified control")
    if any(asdict(settings).get(key) != value for key, value in config["guard"].items()):
        raise ValueError("projection reference has different guard settings")
    if any(row["sha256"] != config["calibration"][name]["sha256"] for name, row in data["cache_reports"].items()):
        raise ValueError("projection reference used different cache files")
    prior = summary["variants"]["projected_guarded"]
    counts = {key: prior[key] == variant[key] for key in
              ("applied_steps", "rejected_steps", "rejected_trials", "projected_steps", "attempted_steps", "stop_reason")}
    metrics = {
        "train_gap": (prior["last"]["train_rank"]["gap_rmse"], variant["last"]["train_rank"]["gap_rmse"]),
        "selection_gap": (prior["last"]["rank"]["gap_rmse"], variant["last"]["rank"]["gap_rmse"]),
        "value_rmse": (prior["last"]["calibration"]["rmse"], variant["last"]["calibration"]["rmse"]),
        "value_bias": (prior["last"]["calibration"]["bias"], variant["last"]["calibration"]["bias"]),
    }
    metric_checks = {key: bool(np.isclose(a, b, atol=tolerance, rtol=tolerance)) for key, (a, b) in metrics.items()}
    saved = torch.load(directory / "projected_guarded/last-head.pt", map_location="cpu", weights_only=True)
    head_matches = (saved["input_checkpoint_fingerprint"] == config["checkpoint_fingerprint"]
                    and saved["value_head"].keys() == state.keys()
                    and all(torch.allclose(value.cpu(), saved["value_head"][key], atol=tolerance, rtol=tolerance)
                            for key, value in state.items()))
    return {"matches": all(counts.values()) and all(metric_checks.values()) and head_matches,
            "counts": counts, "metrics": metric_checks, "head_matches": head_matches, "run_dir": str(directory)}


def run_experiment(args, model, opponent, source, run):
    configuration, previous = source["config"], source["args"]
    if (args.calibration_weight not in configuration["calibration_weights"]
            or not 1 <= args.control_update <= min(args.updates, previous["updates"])):
        raise ValueError("the control weight and step must exist in the original source run")
    recorded = {step: _recorded_row(source["directory"], args.calibration_weight, step)
                for step in (0, args.control_update)}
    model.eval().requires_grad_(False)
    opponent.eval().requires_grad_(False)
    data = load_source_data(source, model, opponent, device=args.device, calibration_cache=args.calibration_cache)
    initial = deepcopy(model.value_head.state_dict())
    fingerprint = model_fingerprint(model)
    model.value_head.requires_grad_(True)
    settings = FlatImprovements(value_updates=args.updates, value_batch_size=previous["batch_size"],
        value_learning_rate=previous["learning_rate"], value_fit_backoffs=0,
        value_capacity=max(previous["batch_size"], len(data["calibration"]["train"])))
    modes = [("sum", None), ("projected_guarded", args.guard)]
    if getattr(args, "compare_fallback", False):
        modes.append(("projected_gap_fallback", replace(args.guard, fallback_gap_only=True,
            fallback_calibration_relative_regression=getattr(args, "fallback_calibration_relative_regression", 0.0))))
    run.write_config(source_run=str(source["directory"]), source_config=configuration,
        calibration=data["cache_reports"], checkpoint_fingerprint=fingerprint, guard=args.guard,
        modes={mode: guard for mode, guard in modes},
        projection_reference_run=getattr(args, "projection_reference_run", None),
        gap_weight=1, branch_mse_weight=configuration.get("pair_value_loss_weight", 0),
        cache_weight=args.calibration_weight, settings=settings,
        seed=data["training"].seed, control_update=args.control_update, diagnostic_only=True,
        guard_scope="full weighted training gap; no selection samples used for updates or retries")
    report = {"diagnostic_only": True, "new_games_collected": 0, "benchmark_performed": False,
              "control_matches": False, "projection_reference_matches": None,
              "variants": {}, "selected": None, "saved_checkpoint": None}
    heads = {"original": initial, "zero": None}
    winner = winner_state = None
    try:
        for mode, guard in modes:
            model.value_head.load_state_dict(initial)
            child = run.child(mode)
            child.write_config(mode=mode, guard=guard, settings=settings, source_config=configuration)
            captured, best_gap, last_state, last_step = {}, None, None, 0

            def capture(weight, step, metrics, state):
                nonlocal best_gap, last_state, last_step
                last_state, last_step = deepcopy(state), step
                if step in (0, args.control_update):
                    captured[step] = deepcopy(metrics)
                key = metrics["rank"]["gap_rmse"], metrics["calibration"]["rmse"]
                if np.isfinite(key).all() and (best_gap is None or key < best_gap["key"]):
                    best_gap = {"key": key, "update": step, "metrics": deepcopy(metrics), "state": deepcopy(state)}

            start = perf_counter()
            print(f"value projection start | mode={mode} | proposed_updates={args.updates}", flush=True)
            state, fitted = fit_calibrated_heads(
                model, data["ranking"], data["calibration"], data["training"], settings,
                weights=(args.calibration_weight,), pair_value_weight=configuration.get("pair_value_loss_weight", 0),
                balance_pair_maps=data["weighted"], rmse_regression=configuration["rmse_regression"],
                bias_regression=configuration["bias_regression"], run=child, log_interval=args.log_interval,
                update_callback=capture, gradient_guard=guard)
            if mode == "sum":
                checks = {str(step): compare_metrics(recorded[step], captured[step], args.reproduction_tolerance)
                          for step in recorded if step in captured}
                report["control_checks"] = checks
                report["control_matches"] = len(checks) == 2 and all(row["matches"] for row in checks.values())
                if not report["control_matches"]:
                    return {**report, "reason": "control_reproduction_mismatch"}
            counts = fitted["variants"][str(args.calibration_weight)]
            variant = {"before": fitted["reference"], "last": counts["last"], "selected": fitted["selected"],
                       "attempted_steps": last_step, "elapsed_seconds": perf_counter() - start,
                       **{key: counts[key] for key in ("applied_steps", "rejected_steps", "rejected_trials",
                            "projected_steps", "stop_reason", "rejections", "fallback_proposals", "fallback_applied_steps",
                            "optimizer_moment_resets", "fallback_calibration_rejections", "adamw_gap_ascent_trials", "adamw_measured_trials")}}
            if mode == "projected_guarded" and getattr(args, "projection_reference_run", None) is not None:
                verified = projection_reference(args.projection_reference_run, source, data, args.guard,
                                                 variant, last_state, args.reproduction_tolerance)
                report["projection_reference"] = verified
                report["projection_reference_matches"] = verified["matches"]
                if not verified["matches"]:
                    return {**report, "reason": "projection_reference_mismatch"}
            heads[f"{mode}-last"] = last_state
            torch.save({"value_head": last_state, "update": last_step, "mode": mode,
                        "input_checkpoint_fingerprint": fingerprint, "diagnostic_only": True},
                       child.directory / "last-head.pt")
            if best_gap is not None:
                variant["best_gap"] = {"update": best_gap["update"], **best_gap["metrics"]}
                heads[f"{mode}-best-gap"] = best_gap["state"]
                torch.save({"value_head": best_gap["state"], "update": best_gap["update"], "mode": mode,
                            "input_checkpoint_fingerprint": fingerprint, "diagnostic_only": True},
                           child.directory / "best-gap-head.pt")
            if state is not None:
                selected = fitted["selected"]
                heads[f"{mode}-admitted"] = state
                key = (-selected["rank"]["preference_accuracy"], selected["rank"]["gap_rmse"],
                       selected["calibration"]["rmse"], abs(selected["calibration"]["bias"]))
                if winner is None or key < winner[0]:
                    winner, winner_state = (key, {"mode": mode, **deepcopy(selected)}), deepcopy(state)
                torch.save({"value_head": state, "update": selected["update"], "mode": mode,
                            "input_checkpoint_fingerprint": fingerprint, "diagnostic_only": True},
                           child.directory / "admitted-head.pt")
            report["variants"][mode] = variant
            child.finish(variant, plots=False)
            last = variant["last"]
            print(f"value projection comparison | mode={mode} | applied={variant['applied_steps']}/{last_step} | "
                  f"rejected_trials={variant['rejected_trials']} | train_gap={last['train_rank']['gap_rmse']:.4f} | "
                  f"selection_gap={last['rank']['gap_rmse']:.4f} | rank={last['rank']['preference_accuracy']} | "
                  f"value_rmse={last['calibration']['rmse']:.4f} | fallback={variant['fallback_applied_steps']}/"
                  f"{variant['fallback_proposals']} | stop={variant['stop_reason']}", flush=True)
            run.log(mode_index=len(report["variants"]), applied_steps=variant["applied_steps"],
                    rejected_steps=variant["rejected_steps"], rejected_trials=variant["rejected_trials"],
                    elapsed_seconds=variant["elapsed_seconds"], selection_gap_rmse=last["rank"]["gap_rmse"],
                    selection_value_rmse=last["calibration"]["rmse"])
            run.finish({**report, "reason": "comparison_in_progress"}, plots=False)
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
    parser.add_argument("--calibration-weight", type=float, default=1.0)
    parser.add_argument("--updates", type=int, default=512)
    parser.add_argument("--control-update", type=int, default=448)
    parser.add_argument("--reproduction-tolerance", type=float, default=0.0001)
    parser.add_argument("--log-interval", type=int, default=64)
    parser.add_argument("--guard-backoffs", type=int, default=6)
    parser.add_argument("--guard-backoff-factor", type=float, default=0.5)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-7)
    parser.add_argument("--guard-rejection-patience", type=int, default=16)
    parser.add_argument("--compare-fallback", action="store_true")
    parser.add_argument("--projection-reference-run", type=Path)
    parser.add_argument("--fallback-calibration-relative-regression", type=float, default=0.0,
                        help="Allowed per-fallback relative increase of full training-cache MSE; default zero")
    parser.set_defaults(project=Path("runs/flat-11x11/logs"), name="flat-004-projected-value-offline")
    args = parser.parse_args()
    try:
        args.guard = GapGuardConfig(backoffs=args.guard_backoffs, backoff_factor=args.guard_backoff_factor,
            minimum_learning_rate=args.minimum_learning_rate, rejection_patience=args.guard_rejection_patience)
    except ValueError as error:
        parser.error(str(error))
    if (args.updates < 1 or not 1 <= args.control_update <= args.updates or args.log_interval < 1
            or not np.isfinite(args.calibration_weight) or args.calibration_weight < 0
            or not np.isfinite(args.fallback_calibration_relative_regression) or args.fallback_calibration_relative_regression < 0
            or not np.isfinite(args.reproduction_tolerance) or args.reproduction_tolerance <= 0):
        parser.error("invalid coefficient, update budget or reproduction tolerance")
    source = read_source(args.source_run)
    model_path, opponent_path = (Path(source["args"][key]) for key in ("model_in", "opponent"))
    if args.minimum_learning_rate > source["args"]["learning_rate"]:
        parser.error("minimum learning rate exceeds the source run's initial learning rate")
    inputs = {path.resolve() for path in (args.source_run, args.projection_reference_run) if path is not None}
    if (args.project / "value" / args.name).resolve() in inputs:
        parser.error("comparison logs must not overwrite an input run")
    if args.model_out is not None and (args.model_out.exists() or args.model_out.resolve() in
                                     {model_path.resolve(), opponent_path.resolve()}):
        parser.error("--model-out must be a new checkpoint path")
    device = select_device(args.device)
    model, _ = load_model(model_path, device)
    opponent, _ = load_model(opponent_path, device)
    run = training_run_from_args(args, "value")
    report = run_experiment(args, model, opponent, source, run)
    report["run_dir"] = str(run.finish(report, plots=False))
    console = {key: report[key] for key in ("reason", "control_matches", "projection_reference_matches", "new_games_collected",
                                          "benchmark_performed", "saved_checkpoint", "run_dir")}
    console["variants"] = {mode: {
        **{key: row[key] for key in ("attempted_steps", "applied_steps", "rejected_steps", "rejected_trials",
                "projected_steps", "stop_reason", "elapsed_seconds", "fallback_proposals", "fallback_applied_steps",
                "optimizer_moment_resets", "fallback_calibration_rejections", "adamw_gap_ascent_trials", "adamw_measured_trials")},
        "last_train_gap_rmse": row["last"]["train_rank"]["gap_rmse"],
        "last_selection_gap_rmse": row["last"]["rank"]["gap_rmse"],
        "last_preference_accuracy": row["last"]["rank"]["preference_accuracy"],
        "last_value_rmse": row["last"]["calibration"]["rmse"],
        "best_gap_update": row["best_gap"]["update"] if row.get("best_gap") else None,
        "admitted_update": row["selected"]["update"] if row["selected"] else None,
    } for mode, row in report["variants"].items()}
    if report["selected"]:
        console["selected"] = {key: report["selected"][key] for key in ("mode", "update")}
    print_summary(console)
    if not report["control_matches"] or report["projection_reference_matches"] is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
