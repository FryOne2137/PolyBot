"""Audit value outliers, cross-validate attenuation on training maps, and test one selected scale."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from _bootstrap import add_local_source_root

add_local_source_root()

from reconstruct_ranked_value import load_source_data, read_source
from polybot.infrastructure.device import select_device
from polybot.infrastructure.model_io import load_model
from polybot.training.cli import add_run_options, print_summary, training_run_from_args
from polybot.training.flat_experiments import model_fingerprint
from polybot.training.flat_value import _paired_value_tensors
from polybot.training.mcts_value import evaluate_mcts_value_heads
from polybot.training.value_scale import cross_validate_scale, rank_errors, scale_metrics


def pair_arrays(model, examples):
    device = next(model.parameters()).device
    features, targets, signs, ordered = _paired_value_tensors(examples, device)
    with torch.no_grad():
        values = model.value_head(features).squeeze(-1).float().cpu().numpy().astype(np.float64)
    return {
        "predictions": values, "targets": targets.cpu().numpy().astype(np.float64), "signs": signs.cpu().numpy(),
        "map_seeds": np.array([pair[0].game_key[0] for pair in ordered]),
        "pair_ids": np.array([pair[0].game_key[-1] for pair in ordered]),
        "players": np.array([pair[0].game_key[1] for pair in ordered]),
        "phases": np.array([pair[0].phase for pair in ordered]),
        "gap_weights": np.array([pair[0].gap_weight for pair in ordered]),
        "value_weights": np.array([pair[0].supervision_weight for pair in ordered]),
    }


def terminal_arrays(model, examples):
    features = torch.as_tensor(np.stack([row.features for row in examples]), device=next(model.parameters()).device)
    with torch.no_grad():
        values = model.value_head(features).reshape(-1).float().cpu().numpy().astype(np.float64)
    return {"predictions": values, "targets": np.array([row.return_target for row in examples]),
            "map_seeds": np.array([row.game_key[0] for row in examples])}


def verify_audited_head(candidate, original, pairs, calibration, audit, tolerance):
    if candidate.config != original.config or candidate.state_dict().keys() != original.state_dict().keys():
        raise ValueError("audited head has an incompatible network configuration")
    for key, value in candidate.state_dict().items():
        if not key.startswith("value_head.") and not torch.equal(value, original.state_dict()[key]):
            raise ValueError(f"audited checkpoint changes the frozen network: {key}")
    gap = np.diff(pairs["predictions"] * pairs["signs"], axis=1).reshape(-1)
    target = np.diff(pairs["targets"] * pairs["signs"], axis=1).reshape(-1)
    informative = target != 0
    observed = {
        "gap_rmse": float(np.sqrt(np.square(gap - target).mean())),
        "preference_accuracy": float(np.mean(gap[informative] * target[informative] > 0)) if informative.any() else None,
        "calibration_rmse": float(np.sqrt(np.square(calibration["predictions"] - calibration["targets"]).mean())),
        "calibration_bias": float(np.mean(calibration["predictions"] - calibration["targets"])),
    }
    expected = audit["reconstructed"]
    reference = {"gap_rmse": expected["rank"]["gap_rmse"], "preference_accuracy": expected["rank"]["preference_accuracy"],
                 "calibration_rmse": expected["calibration"]["rmse"], "calibration_bias": expected["calibration"]["bias"]}
    if any(value is None or not np.isclose(value, reference[key], atol=tolerance, rtol=tolerance)
           for key, value in observed.items()):
        raise ValueError("checkpoint metrics differ from the audited reconstructed head")
    return observed


def run_analysis(args, original, candidate, opponent, source, audit_source, run):
    if not audit_source["summary"].get("reconstruction_matches"):
        raise ValueError("the audit run did not verify a reconstructed checkpoint")
    if audit_source["config"]["source_config"] != source["config"]:
        raise ValueError("audit and training runs describe different source data")
    prior_arena = set(audit_source["config"]["arena_map_seeds"])
    arena = set(range(args.validation_seed, args.validation_seed + args.validation_games // 2))
    if arena & prior_arena:
        raise ValueError("new arena reuses the previous audit's test maps")
    original.eval().requires_grad_(False)
    candidate.eval().requires_grad_(False)
    opponent.eval().requires_grad_(False)
    data = load_source_data(source, original, opponent, device=args.device, calibration_cache=args.calibration_cache,
                            reserved_extra_seeds=arena | prior_arena)
    for name, cache in data["cache_reports"].items():
        if cache["sha256"] != audit_source["config"]["calibration"][name]["sha256"]:
            raise ValueError("calibration cache changed after the reconstructed checkpoint audit")
    pairs = {name: pair_arrays(candidate, rows) for name, rows in data["ranking"].items()}
    calibration = {name: terminal_arrays(candidate, rows) for name, rows in data["calibration"].items()}
    verification = verify_audited_head(candidate, original, pairs["selection"], calibration["selection"],
                                       audit_source["summary"], args.verification_tolerance)
    cv = cross_validate_scale(pairs["train"], calibration["train"], folds=args.folds, seed=args.fold_seed,
                               calibration_weights=args.calibration_weights,
                               pair_value_weight=source["config"].get("pair_value_loss_weight", 1.0),
                               maximum=args.maximum_scale)
    scale = cv["selection"]["scale"]
    outliers, per_pair = {}, []
    for name, rows in data["ranking"].items():
        before = pair_arrays(original, rows)["predictions"]
        diagnostic = rank_errors(pairs[name], original_predictions=before, scale=scale, top=args.top_errors)
        per_pair.extend({"split": name, **row} for row in diagnostic.pop("all_pairs"))
        outliers[name] = diagnostic
    (run.directory / "gap_errors.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in per_pair), encoding="utf-8")
    report = {
        "diagnostic_only": True, "model_updated": False, "benchmark_performed": False, "beats_zero": None,
        "head_verification": verification, "cross_validation": cv, "selected_scale": scale,
        "worst_errors": outliers,
        "offline": {name: {arm: scale_metrics(pairs[name], calibration[name], value)
                            for arm, value in (("unscaled", 1.0), ("selected", scale), ("zero", 0.0))}
                    for name in pairs},
    }
    configuration = {"leaf_value_scale": scale, "checkpoint": str(args.model_in),
                     "checkpoint_fingerprint": model_fingerprint(candidate), "source_run": str(source["directory"]),
                     "audit_run": str(audit_source["directory"]), "selection": cv["selection"], "fold_seed": args.fold_seed,
                     "arena_map_seeds": sorted(arena), "simulations": data["mcts"].simulations,
                     "opponent_value_scale": data["pool"]["opponent_value_scale"],
                     "selection_used": "original training maps only; original selection maps and arena excluded"}
    run.write_config(**configuration, source_config=source["config"], calibration=data["cache_reports"])
    (run.directory / "value_scale.json").write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    for name, row in cv["methods"].items():
        metrics = row["out_of_fold"]
        run.log(method_index=list(cv["methods"]).index(name), scale_median=row["scale_median"],
                scale_std=row["scale_std"], cv_gap_mse=metrics["weighted_gap_mse"],
                cv_value_rmse=metrics["calibration_rmse"], eligible=row["eligible"])
        print(f"scale CV | method={name} | scale={row['scale_median']:.4f} | std={row['scale_std']:.4f} | "
              f"gap_rmse={metrics['gap']['all']['gap_rmse']:.4f} | eligible={int(row['eligible'])}", flush=True)
    print(f"selected value scale | scale={scale:.6f} | method={cv['selection']['method']} | "
          "selection=training_maps_only", flush=True)
    if scale <= 1e-8 or args.offline_only:
        return {**report, "reason": "zero_selected" if scale <= 1e-8 else "offline_only"}
    report["reason"] = "arena_pending"
    run.finish(report, plots=False)
    benchmark = evaluate_mcts_value_heads(candidate, opponent, replace(data["training"], seed=args.validation_seed),
        data["mcts"], args.validation_games, value_scales=(0.0, scale), isolate_failed_pairs=True, **data["pool"])
    report.update(benchmark_performed=True, mcts_benchmark=benchmark,
                  reason="arena_complete" if benchmark["valid"] else "arena_invalid")
    if benchmark["valid"]:
        key = f"{scale:g}"
        delta = benchmark["paired_scores"][key] - benchmark["paired_scores"]["0"]
        report.update(beats_zero=delta > 0, score_delta_vs_zero=delta)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_options(parser)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--audit-run", type=Path, required=True)
    parser.add_argument("--model-in", type=Path)
    parser.add_argument("--calibration-cache", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu", "mps", "auto"), default="cuda")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--fold-seed", type=int, default=1758127001)
    parser.add_argument("--calibration-weights", default="0,0.05,0.2,1.0")
    parser.add_argument("--maximum-scale", type=float, default=1.0)
    parser.add_argument("--verification-tolerance", type=float, default=0.0001)
    parser.add_argument("--top-errors", type=int, default=10)
    parser.add_argument("--validation-games", type=int, default=24)
    parser.add_argument("--validation-seed", type=int, default=2142127001)
    parser.add_argument("--offline-only", action="store_true")
    parser.set_defaults(project=Path("runs/flat-11x11/logs"), name="flat-004-step448-scale-audit")
    args = parser.parse_args()
    try:
        args.calibration_weights = tuple(float(value) for value in args.calibration_weights.split(","))
    except ValueError:
        parser.error("--calibration-weights needs comma-separated numbers")
    if (not args.calibration_weights or len(set(args.calibration_weights)) != len(args.calibration_weights)
            or any(not np.isfinite(value) or value < 0 for value in args.calibration_weights)
            or not 0 < args.maximum_scale <= 1 or not 0 < args.verification_tolerance < 1
            or args.folds < 2 or args.fold_seed < 0 or args.top_errors < 1
            or args.validation_games < 4 or args.validation_games % 2 or args.validation_seed < 1
            or args.validation_seed + args.validation_games // 2 > 2**31):
        parser.error("invalid scale, fold or arena configuration")
    source, audit = read_source(args.source_run), read_source(args.audit_run)
    if args.model_in is None:
        if not audit["summary"].get("saved_checkpoint"):
            parser.error("the audit run did not save a checkpoint")
        args.model_in = Path(audit["summary"]["saved_checkpoint"])
    destination = (args.project / "value" / args.name).resolve()
    if destination in {args.source_run.resolve(), args.audit_run.resolve()}:
        parser.error("scale analysis must not overwrite its input runs")
    device = select_device(args.device)
    original, _ = load_model(Path(source["args"]["model_in"]), device)
    candidate, _ = load_model(args.model_in, device)
    opponent, _ = load_model(Path(source["args"]["opponent"]), device)
    run = training_run_from_args(args, "value")
    report = run_analysis(args, original, candidate, opponent, source, audit, run)
    report["run_dir"] = str(run.finish(report, plots=False))
    print_summary(report)


if __name__ == "__main__":
    main()
