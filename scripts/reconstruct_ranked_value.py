"""Replay an exact logged value-head update, audit its gaps, and run a fresh MCTS arena."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from _bootstrap import add_local_source_root

add_local_source_root()

from polybot.config import MCTSConfig, TrainingConfig
from polybot.infrastructure.device import select_device
from polybot.infrastructure.model_io import load_model, save_model
from polybot.training.calibrated_value import (
    cached_terminal_positions, calibration_admission, fit_calibrated_heads, informative_pairs, load_ranking_sources,
)
from polybot.training.cli import add_run_options, print_summary, training_run_from_args
from polybot.training.flat_experiments import FlatImprovements, model_fingerprint
from polybot.training.flat_value import _paired_value_tensors, _value_gap, paired_gap_breakdown
from polybot.training.mcts_value import evaluate_mcts_value_heads
from polybot.training.saved_value_pairs import split_value_pairs


def read_source(directory):
    def read(name):
        return json.loads((directory / name).read_text(encoding="utf-8-sig"))
    return {"directory": directory, "args": read("args.json")["arguments"],
            "config": read("config.json"), "summary": read("summary.json")}


def _recorded_row(directory, weight, update):
    path = directory / f"calibration-{weight!r}" / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    found = [row for row in rows if int(row.get("update", -1)) == update and row.get("calibration_weight") == weight]
    if len(found) != 1:
        raise ValueError(f"source run must contain exactly one logged update {update} for weight {weight}")
    return found[0]


def compare_metrics(recorded, measured, tolerance):
    actual = {
        "selection_gap_rmse": measured["rank"]["gap_rmse"],
        "selection_preference_accuracy": measured["rank"]["preference_accuracy"],
        "selection_value_rmse": measured["calibration"]["rmse"],
        "selection_value_bias": measured["calibration"]["bias"],
        "train_gap_rmse": measured["train_rank"]["gap_rmse"],
        "train_preference_accuracy": measured["train_rank"]["preference_accuracy"],
        "train_calibration_mse": measured["train_calibration_mse"],
        "train_weighted_gap_loss": measured["train_weighted_gap_loss"],
        "train_weighted_pair_mse": measured["train_weighted_pair_mse"],
    }
    required = {name for name in actual if name.startswith("selection_")}
    if not required <= recorded.keys():
        raise ValueError("source history is missing required reconstruction metrics")
    comparisons = {name: {"recorded": recorded[name], "replayed": value,
                          "matches": bool(np.isclose(recorded[name], value, atol=tolerance, rtol=tolerance))}
                   for name, value in actual.items() if name in recorded and value is not None}
    return {"matches": all(row["matches"] for row in comparisons.values()), "metrics": comparisons}


def gap_audit(model, ranking, heads, path):
    original = deepcopy(model.value_head.state_dict())
    device = next(model.parameters()).device
    report, records = {}, []
    try:
        for split, rows in ranking.items():
            features, targets, signs, ordered = _paired_value_tensors(rows, device)
            predictions = {}
            report[split] = {}
            for name, state in heads.items():
                if state is None:
                    predicted = torch.zeros_like(targets)
                else:
                    model.value_head.load_state_dict(state)
                    with torch.no_grad():
                        predicted = model.value_head(features).squeeze(-1).float()
                report[split][name] = paired_gap_breakdown(predicted, targets, signs)
                predictions[name] = _value_gap(predicted, signs).detach().float().cpu().tolist()
            target_gaps = _value_gap(targets, signs).float().cpu().tolist()
            for index, pair in enumerate(ordered):
                records.append({
                    "split": split, "map_seed": pair[0].game_key[0], "root_player": pair[0].game_key[1],
                    "pair_id": pair[0].game_key[-1], "phase": pair[0].phase, "target_gap": target_gaps[index],
                    "group": "tied" if target_gaps[index] == 0 else "informative",
                    "predicted_gaps": {name: values[index] for name, values in predictions.items()},
                    "gap_weight": pair[0].gap_weight, "supervision_weight": pair[0].supervision_weight,
                })
        path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in records), encoding="utf-8")
    finally:
        model.value_head.load_state_dict(original)
    return report


def reconstruct(args, model, opponent, model_config, source, run):
    configuration, previous = source["config"], source["args"]
    if model_fingerprint(model) != configuration["checkpoint_fingerprints"]["model"]:
        raise ValueError("input checkpoint differs from the original run's initial weights")
    if model_fingerprint(opponent) != configuration["checkpoint_fingerprints"]["opponent"]:
        raise ValueError("opponent checkpoint differs from the original run")
    if args.update > previous["updates"] or args.calibration_weight not in configuration["calibration_weights"]:
        raise ValueError("requested weight or update was not present in the source run")
    recorded = {step: _recorded_row(source["directory"], args.calibration_weight, step) for step in (0, args.update)}
    model.eval().requires_grad_(False)
    model.value_head.requires_grad_(True)
    opponent.eval().requires_grad_(False)
    training = replace(TrainingConfig(**configuration["training"]), device=args.device)
    mcts = MCTSConfig(**configuration["mcts"])
    saved_sources = configuration["ranking_sources"]["sources"]
    examples, current_source = load_ranking_sources(model, training, [Path(row["directory"]) for row in saved_sources],
                                                    history=previous["visible_action_history"])
    for before, now in zip(saved_sources, current_source["sources"], strict=True):
        if before["source_sha256"] != now["source_sha256"]:
            raise ValueError("saved pair dataset changed after the original fit; exact reconstruction requires its original snapshot")
    split = configuration["ranking_split"]
    ranking, replayed_split = split_value_pairs(examples, seed=split["seed"], selection_fraction=split["selection_fraction"])
    if replayed_split != split:
        raise ValueError("replayed whole-map split differs from the original run")
    weighted = configuration.get("balance_pair_maps", False)
    if not weighted:
        ranking = {name: informative_pairs(rows) for name, rows in ranking.items()}
    pool = {key: previous[key] for key in ("num_envs", "num_threads", "visible_action_history", "max_pending_leaves")}
    pool["opponent_value_scale"] = 1.0
    reserved = set(current_source["reserved_map_seeds"])
    for seed, games in configuration["schedules"].values():
        reserved.update(range(seed, seed + games // 2))
    arena_maps = set(range(args.validation_seed, args.validation_seed + args.validation_games // 2))
    if reserved & arena_maps:
        raise ValueError("fresh arena maps overlap original source, calibration or arena reservations")
    calibration, cache_reports = {}, {}
    for name in ("train", "selection"):
        seed, games = configuration["schedules"][name]
        own_schedule = set(range(seed, seed + games // 2))
        calibration[name], summary = cached_terminal_positions(
            model, opponent, replace(training, seed=seed), mcts, games,
            cache_directory=args.calibration_cache or Path(previous["calibration_cache"]),
            excluded_map_seeds=(reserved - own_schedule) | arena_maps,
            positions_per_game=previous["positions_per_game"], pool=pool,
            policy=configuration["calibration_policy"], require_existing=True,
        )
        original_summary = source["summary"]["calibration"][name]
        for key in ("positions", "map_seeds", "attempted_map_seeds"):
            if summary[key] != original_summary[key]:
                raise ValueError(f"{name} calibration cache differs from the original collection")
        reserved.update(summary["attempted_map_seeds"])
        cache_reports[name] = {**summary, "sha256": sha256(Path(summary["cache_path"]).read_bytes()).hexdigest()}
        print(f"reconstruction cache | split={name} | positions={len(calibration[name])} | cache_loaded=1", flush=True)
    settings = FlatImprovements(value_updates=args.update, value_batch_size=previous["batch_size"],
                                value_learning_rate=previous["learning_rate"], value_fit_backoffs=0,
                                value_capacity=max(previous["batch_size"], len(calibration["train"])))
    captured = {}

    def capture(weight, step, metrics, state):
        if step in (0, args.update):
            captured[step] = {"state": deepcopy(state), "metrics": deepcopy(metrics)}

    run.write_config(source_run=str(source["directory"]), source_config=configuration,
                     reconstruction_update=args.update, reconstruction_weight=args.calibration_weight,
                     tolerance=args.reproduction_tolerance, calibration=cache_reports,
                     arena_map_seeds=sorted(arena_maps), diagnostic_only=True)
    fit_calibrated_heads(
        model, ranking, calibration, training, settings, weights=(args.calibration_weight,),
        rmse_regression=configuration["rmse_regression"], bias_regression=configuration["bias_regression"],
        pair_value_weight=configuration.get("pair_value_loss_weight", 0.0), balance_pair_maps=weighted,
        run=run, log_interval=args.log_interval, update_callback=capture,
    )
    checks = {str(step): compare_metrics(recorded[step], captured[step]["metrics"], args.reproduction_tolerance)
              for step in (0, args.update) if step in captured}
    matched = len(checks) == 2 and all(row["matches"] for row in checks.values())
    report = {"source_run": str(source["directory"]), "update": args.update, "calibration_weight": args.calibration_weight,
              "diagnostic_only": True, "reconstruction_matches": matched, "metric_checks": checks,
              "saved_checkpoint": None, "benchmark_performed": False, "beats_zero": None}
    if not matched:
        return {**report, "reason": "reconstruction_mismatch"}
    initial, candidate = captured[0], captured[args.update]
    failures = calibration_admission(candidate["metrics"]["rank"], candidate["metrics"]["calibration"],
                                     initial["metrics"]["rank"], initial["metrics"]["calibration"],
                                     rmse_regression=configuration["rmse_regression"], bias_regression=configuration["bias_regression"])
    report.update(before=initial["metrics"], reconstructed=candidate["metrics"], original_gate_failures=failures)
    report["gap_diagnostics"] = gap_audit(model, ranking,
        {"original": initial["state"], "reconstructed": candidate["state"], "zero": None}, run.directory / "gap_pairs.jsonl")
    for name, groups in report["gap_diagnostics"]["selection"].items():
        print(f"gap audit | head={name} | tied={groups['tied']['gap_rmse']} ({groups['tied']['pairs']}) | "
              f"informative={groups['informative']['gap_rmse']} ({groups['informative']['pairs']})", flush=True)
    try:
        model.value_head.load_state_dict(candidate["state"])
        save_model(args.model_out, model, model_config)
    finally:
        model.value_head.load_state_dict(initial["state"])
    report.update(saved_checkpoint=str(args.model_out), reason="arena_pending")
    run.finish(report, plots=False)
    print(f"diagnostic checkpoint saved | weight={args.calibration_weight:g} | update={args.update} | "
          f"reconstruction_verified=1 | original_gate_failures={','.join(failures) or 'none'} | "
          f"path={args.model_out}", flush=True)
    benchmark = evaluate_mcts_value_heads(model, opponent, replace(training, seed=args.validation_seed), mcts,
        args.validation_games, fitted_heads={"reconstructed": candidate["state"], "original": initial["state"]},
        isolate_failed_pairs=True, **pool)
    report.update(mcts_benchmark=benchmark, benchmark_performed=True,
                  reason="arena_complete" if benchmark["valid"] else "arena_invalid")
    if benchmark["valid"]:
        scores = benchmark["paired_scores"]
        report.update(beats_zero=scores["reconstructed"] > scores["zero"],
                      score_delta_vs_zero=scores["reconstructed"] - scores["zero"],
                      score_delta_vs_original=scores["reconstructed"] - scores["original"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_options(parser)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--model-in", type=Path)
    parser.add_argument("--opponent", type=Path)
    parser.add_argument("--calibration-cache", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--update", type=int, default=448)
    parser.add_argument("--calibration-weight", type=float, default=1.0)
    parser.add_argument("--reproduction-tolerance", type=float, default=0.0001)
    parser.add_argument("--validation-games", type=int, default=24)
    parser.add_argument("--validation-seed", type=int, default=2141127001)
    parser.add_argument("--log-interval", type=int, default=64)
    parser.set_defaults(project=Path("runs/flat-11x11/logs"), name="flat-004-step448-value-audit")
    args = parser.parse_args()
    if (args.update < 1 or not np.isfinite(args.calibration_weight) or args.calibration_weight <= 0
            or not np.isfinite(args.reproduction_tolerance) or args.reproduction_tolerance <= 0
            or args.validation_games < 4 or args.validation_games % 2 or args.validation_seed < 1
            or args.validation_seed + args.validation_games // 2 > 2**31 or args.log_interval < 1):
        parser.error("invalid reconstruction, tolerance or arena budget")
    source = read_source(args.source_run)
    model_path = args.model_in or Path(source["args"]["model_in"])
    opponent_path = args.opponent or Path(source["args"]["opponent"])
    if args.model_out.resolve() in {model_path.resolve(), opponent_path.resolve()}:
        parser.error("diagnostic output must differ from both original checkpoints")
    if (args.project / "value" / args.name).resolve() == args.source_run.resolve():
        parser.error("audit logs must use a separate directory from the source run")
    device = select_device(args.device)
    model, model_config = load_model(model_path, device)
    opponent, _ = load_model(opponent_path, device)
    run = training_run_from_args(args, "value")
    report = reconstruct(args, model, opponent, model_config, source, run)
    report["run_dir"] = str(run.finish(report, plots=False))
    print_summary(report)
    if not report["reconstruction_matches"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
