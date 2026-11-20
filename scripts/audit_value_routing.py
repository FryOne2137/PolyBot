"""Audit turn changes, MCTS model routing and cache-gradient conflicts on saved pairs."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from _bootstrap import add_local_source_root

add_local_source_root()

from calibrate_value_scale import pair_arrays, terminal_arrays, verify_audited_head
from reconstruct_ranked_value import load_source_data, read_source
from polybot.infrastructure.device import select_device
from polybot.infrastructure.model_io import load_model
from polybot.training.cli import add_run_options, print_summary, training_run_from_args
from polybot.training.flat_experiments import model_fingerprint
from polybot.training.value_routing_audit import audit_routing, load_routing_inputs


def run_audit(args, model, opponent, source, run):
    model.eval().requires_grad_(False)
    opponent.eval().requires_grad_(False)
    data = load_source_data(source, model, opponent, device=args.device, calibration_cache=args.calibration_cache)
    fingerprints = {"apprentice": model_fingerprint(model), "opponent": model_fingerprint(opponent)}
    initial = deepcopy(model.value_head.state_dict())
    stages, head_provenance = {"initial": initial}, {}
    if args.head_run is not None:
        previous = read_source(args.head_run)
        if (not previous["summary"].get("control_matches")
                or previous["config"]["source_config"] != source["config"]
                or previous["summary"].get("projection_reference_matches") is False):
            raise ValueError("saved head run must have a verified control and match this source run")
        for name, cache in data["cache_reports"].items():
            if cache["sha256"] != previous["config"]["calibration"][name]["sha256"]:
                raise ValueError("saved head used different calibration caches")
        snapshot = torch.load(args.head_run / args.head_mode / "last-head.pt", map_location="cpu", weights_only=True)
        if (snapshot["input_checkpoint_fingerprint"] != fingerprints["apprentice"]
                or snapshot["mode"] != args.head_mode):
            raise ValueError("saved head fingerprint or mode differs from the selected source")
        candidate = deepcopy(model)
        candidate.value_head.load_state_dict(snapshot["value_head"], strict=True)
        measured = previous["summary"]["variants"][args.head_mode]
        verify_audited_head(candidate, model, pair_arrays(candidate, data["ranking"]["selection"]),
            terminal_arrays(candidate, data["calibration"]["selection"]),
            {"reconstructed": measured["last"]}, args.verification_tolerance)
        stages[f"{args.head_mode}-last"] = deepcopy(candidate.value_head.state_dict())
        head_provenance = {"run": str(args.head_run), "mode": args.head_mode,
                           "proposed_steps": measured["attempted_steps"], "applied_steps": measured["applied_steps"]}
        del candidate
    lookup = load_routing_inputs(model, opponent, data["training"], data["ranking"],
        source["config"]["ranking_sources"]["sources"], history=source["args"]["visible_action_history"],
        require_agent_id=args.require_agent_id)
    run.write_config(source_run=str(source["directory"]), source_config=source["config"],
        calibration=data["cache_reports"], checkpoint_fingerprints=fingerprints, head=head_provenance,
        apprentice_value_scale=args.apprentice_value_scale, opponent_value_scale=args.opponent_value_scale,
        gradient_scope="training pairs and training cache only", batches=args.batches,
        successor_files=[branch for pair in lookup.values() for branch in pair], diagnostic_only=True)
    report = {"diagnostic_only": True, "new_games_collected": 0, "model_updated": False,
              "benchmark_performed": False, "head": head_provenance, "stages": {}}
    model.value_head.requires_grad_(True)
    try:
        with (run.directory / "routing_pairs.jsonl").open("w", encoding="utf-8") as stream:
            for stage, state in stages.items():
                model.value_head.load_state_dict(state)
                result, rows = audit_routing(model, data["ranking"], data["calibration"], lookup,
                    seed=data["training"].seed, batch_size=source["args"]["batch_size"], batches=args.batches,
                    pair_weight=source["config"].get("pair_value_loss_weight", 0), cache_weight=args.calibration_weight,
                    balance_pair_maps=data["weighted"], apprentice_scale=args.apprentice_value_scale,
                    opponent_scale=args.opponent_value_scale)
                for row in rows:
                    stream.write(json.dumps({"stage": stage, **row}, allow_nan=False) + "\n")
                stream.flush()
                result["preferences_changed_by_routing"] = sum(row["preference_changed_by_routing"] for row in rows)
                report["stages"][stage] = result
                gradient = result["training_gradients"]["all"]
                print(f"routing audit | stage={stage} | changed_preferences={result['preferences_changed_by_routing']} | "
                      f"legacy_cosine={gradient['legacy']['expected_phase_balanced']['gap_cache_cosine']} | "
                      f"routed_cosine={gradient['routed']['expected_phase_balanced']['gap_cache_cosine']} | "
                      f"routed_status={gradient['routed']['status']}", flush=True)
                for variant in ("legacy", "routed"):
                    run.log(stage_index=list(stages).index(stage), routed=int(variant == "routed"),
                            **gradient[variant]["expected_phase_balanced"])
    finally:
        model.value_head.load_state_dict(initial)
    report["weights_unchanged"] = (model_fingerprint(model) == fingerprints["apprentice"]
                                   and model_fingerprint(opponent) == fingerprints["opponent"])
    if not report["weights_unchanged"]:
        raise RuntimeError("routing audit changed an input network")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_options(parser)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--calibration-cache", type=Path)
    parser.add_argument("--head-run", type=Path)
    parser.add_argument("--head-mode", choices=("sum", "projected_guarded", "projected_gap_fallback"), default="projected_guarded")
    parser.add_argument("--require-agent-id", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--calibration-weight", type=float, default=1.0)
    parser.add_argument("--apprentice-value-scale", type=float, default=1.0)
    parser.add_argument("--opponent-value-scale", type=float, default=1.0)
    parser.add_argument("--verification-tolerance", type=float, default=0.0001)
    parser.set_defaults(project=Path("runs/flat-11x11/logs"), name="flat-004-value-routing-audit")
    args = parser.parse_args()
    if (args.batches < 1 or not 0 <= args.calibration_weight < float("inf")
            or not 0 <= args.apprentice_value_scale <= 1 or not 0 <= args.opponent_value_scale <= 1
            or not 0 < args.verification_tolerance < 1):
        parser.error("invalid batch count, coefficient, scale or verification tolerance")
    inputs = {path.resolve() for path in (args.source_run, args.head_run) if path is not None}
    if (args.project / "value" / args.name).resolve() in inputs:
        parser.error("audit logs must not overwrite input runs")
    source = read_source(args.source_run)
    device = select_device(args.device)
    model, _ = load_model(Path(source["args"]["model_in"]), device)
    opponent, _ = load_model(Path(source["args"]["opponent"]), device)
    run = training_run_from_args(args, "value")
    report = run_audit(args, model, opponent, source, run)
    report["run_dir"] = str(run.finish(report, plots=False))
    console = {key: report[key] for key in ("run_dir", "new_games_collected", "weights_unchanged", "head")}
    console["counts"] = {split: value["counts"] for split, value in report["stages"]["initial"]["splits"].items()}
    console["stages"] = {}
    for stage, result in report["stages"].items():
        console["stages"][stage] = {
            "preferences_changed_by_routing": result["preferences_changed_by_routing"],
            "prediction_metrics": {split: {variant: {
                "gap_rmse": metrics["all"]["gap_rmse"], "tied_gap_rmse": metrics["tied"]["gap_rmse"],
                "informative_pairs": metrics["informative"]["pairs"],
                "preference_accuracy": metrics["informative"]["preference_accuracy"]}
                for variant, metrics in values["groups"]["all"].items() if variant != "pairs"}
                for split, values in result["splits"].items()},
            "training_gradients": {group: {
                "gap_gradient_change_norm": metrics["gap_gradient_change_norm"],
                "gap_gradient_cosine_legacy_vs_routed": metrics["gap_gradient_cosine_legacy_vs_routed"],
                **{
                variant: {"status": metrics[variant]["status"],
                    "gap_norm": metrics[variant]["expected_phase_balanced"]["gap_norm"],
                    "cache_to_gap_norm_ratio": metrics[variant]["expected_phase_balanced"]["cache_to_gap_norm_ratio"],
                    "gap_cache_cosine": metrics[variant]["expected_phase_balanced"]["gap_cache_cosine"],
                    "sampled_cache_conflict_fraction": metrics[variant]["sampled_cache_conflict_fraction"]}
                for variant in ("legacy", "routed")}}
                for group, metrics in result["training_gradients"].items() if metrics["legacy"]["pairs"]},
        }
    print_summary(console)


if __name__ == "__main__":
    main()
