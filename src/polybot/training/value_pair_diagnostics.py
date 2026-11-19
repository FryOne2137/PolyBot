"""Offline loss ablations and representation checks for saved value pairs."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import torch

from .flat_value import _paired_value_tensors, _value_gap, fit_paired_value_heads
from .saved_value_pairs import load_saved_value_pairs, split_value_pairs


LOSSES = ("mse", "delta", "mse_delta")
METRICS = ("value_mse", "gap_mse", "gap_rmse", "preference_accuracy", "informative_pairs", "bias", "rmse")


def inspect_value_pairs(model, examples, *, difference_weight, phase_balanced=True):
    """Measure input separation and gradients without accumulating parameter grads."""
    states, targets, signs, pairs = _paired_value_tensors(examples, next(model.parameters()).device)
    parameters = tuple(parameter for parameter in model.value_head.parameters() if parameter.requires_grad)
    if not parameters:
        raise ValueError("value pair diagnostics require trainable value-head parameters")
    model.eval()
    predictions = model.value_head(states).squeeze(-1).float()
    gaps = _value_gap(predictions, signs)
    reference_gaps = _value_gap(targets, signs)
    informative = reference_gaps.abs() > 1e-8
    value_errors = (predictions - targets).square().mean(dim=1)
    gap_errors = ((gaps - reference_gaps) / 2).square()

    def gradient(loss):
        return torch.cat([
            value.reshape(-1).float()
            for value in torch.autograd.grad(loss, parameters, retain_graph=True)
        ])

    # Match either phase sampling or the explicit full-batch objective.
    phases = [pair[0].phase for pair in pairs]
    present = set(phases)
    weights = torch.tensor(
        [1.0 / (len(present) * phases.count(phase)) if phase_balanced else 1.0 / len(pairs) for phase in phases],
        device=states.device, dtype=torch.float32,
    )
    value_loss = (weights * value_errors).sum()
    gap_loss = (weights * gap_errors).sum()
    value_gradient = gradient(value_loss)
    gap_gradient = gradient(gap_loss)
    value_norm = float(value_gradient.norm().item())
    gap_norm = float(gap_gradient.norm().item())
    cosine = float(torch.dot(value_gradient, gap_gradient).item() / (value_norm * gap_norm)) if value_norm and gap_norm else None
    difference = states[:, 1] - states[:, 0]
    distances = difference.norm(dim=1).detach().cpu().numpy()
    norms = states.norm(dim=2).mean(dim=1).detach().cpu().numpy()
    roots = (predictions * signs).detach().cpu().numpy()
    reference = (targets * signs).detach().cpu().numpy()
    sensitivity = gradient_failures = collisions = 0
    rows = []
    for index, pair in enumerate(pairs):
        same_input = bool(torch.equal(states[index, 0], states[index, 1]))
        same_perspective = pair[0].to_root_sign == pair[1].to_root_sign
        is_informative = bool(informative[index])
        gap_sensitivity = loss_gradient = None
        if is_informative:
            gap_sensitivity = float(gradient(gaps[index] / 2).norm().item())
            loss_gradient = float(gradient(gap_errors[index]).norm().item())
            sensitivity += gap_sensitivity == 0.0
            error = float((gaps[index] - reference_gaps[index]).detach().item())
            gradient_failures += loss_gradient == 0.0 and abs(error) > 1e-8
        collision = same_input and same_perspective and is_informative
        collisions += collision
        seed, player, pair_id = pair[0].game_key
        rows.append({
            "pair_id": int(pair_id), "map_seed": int(seed), "player": int(player), "phase": pair[0].phase,
            "to_root_signs": [row.to_root_sign for row in pair], "informative": is_informative,
            "feature_l2": float(distances[index]), "feature_max_abs": float(difference[index].abs().max().item()),
            "feature_relative_l2": float(distances[index] / max(float(norms[index]), 1e-12)),
            "identical_features": same_input, "same_perspective": same_perspective,
            "identical_features_with_conflicting_gap": collision,
            "root_predictions": roots[index].tolist(), "root_targets": reference[index].tolist(),
            "predicted_gap": float(gaps[index].detach().item()),
            "target_gap": float(reference_gaps[index].item()),
            "normalized_gap_sensitivity_gradient_norm": gap_sensitivity,
            "delta_loss_gradient_norm": loss_gradient,
        })
    informative_distances = distances[informative.cpu().numpy()]
    summary = {
        "pairs": len(pairs), "informative_pairs": int(informative.sum().item()),
        "identical_features_with_conflicting_gap": collisions,
        "informative_pairs_with_zero_sensitivity": sensitivity,
        "informative_pairs_with_zero_loss_gradient_and_error": gradient_failures,
        "informative_feature_l2": {
            "min": float(informative_distances.min()), "median": float(np.median(informative_distances)),
            "max": float(informative_distances.max()),
        } if len(informative_distances) else None,
        "saturated_value_fraction": float((predictions.detach().abs() >= 0.99).float().mean().item()),
        "gradients": {
            "scope": "phase_balanced_split" if phase_balanced else "full_batch_split",
            "value_mse": float(value_loss.detach().item()),
            "normalized_delta_mse": float(gap_loss.detach().item()),
            "value_gradient_norm": value_norm, "delta_gradient_norm": gap_norm,
            "weighted_delta_gradient_norm": difference_weight * gap_norm,
            "delta_to_value_gradient_ratio": difference_weight * gap_norm / value_norm if value_norm else None,
            "value_delta_gradient_cosine": cosine,
        },
    }
    return summary, rows


def compare_saved_value_losses(model, training, directory, settings, run, *, history,
                               split_seed, selection_fraction, difference_weight, log_interval,
                               memorization_test=False):
    examples, source = load_saved_value_pairs(model, training, directory, history=history)
    datasets, split = split_value_pairs(examples, seed=split_seed, selection_fraction=selection_fraction)
    memorization = None
    losses = LOSSES
    if memorization_test:
        pairs = {}
        for example in datasets["train"]:
            pairs.setdefault(example.game_key, []).append(example)
        informative = [
            pair for pair in pairs.values()
            if abs(pair[1].return_target * pair[1].to_root_sign - pair[0].return_target * pair[0].to_root_sign) > 1e-8
        ]
        if not informative:
            raise ValueError("the training split contains no informative pairs for the memorization test")
        datasets["train"] = [example for pair in informative for example in pair]
        settings = replace(settings, value_batch_size=len(datasets["train"]),
                           value_capacity=max(settings.value_capacity, len(datasets["train"])))
        losses = ("delta",)
        memorization = {
            "train_pairs": len(informative), "pair_ids": [int(pair[0].game_key[-1]) for pair in informative],
            "full_batch": True, "first_perfect_training_update": None, "max_training_preference_accuracy": 0.0,
        }
        print(f"value memorization | informative_train_pairs={len(informative)} | full_batch=1 | loss=delta", flush=True)
    run.write_config(
        training=training, model=model.config, settings=settings, source=source, split=split,
        losses={name: {"mse": "value_mse", "delta": "weight * normalized_delta_mse",
                       "mse_delta": "value_mse + weight * normalized_delta_mse"}[name] for name in losses},
        difference_weight=difference_weight, initial_weights="shared", minibatch_draws="shared",
        checkpoint_selection="each variant's held-out objective including the initial head",
        memorization=memorization,
    )
    children = {name: run.child(name) for name in losses}
    diagnostics = {}
    last_diagnostics = {}
    pair_path = run.directory / "pair_diagnostics.jsonl"
    with pair_path.open("w", encoding="utf-8", buffering=1) as stream:
        def record_pairs(stage, rows):
            for row in rows:
                stream.write(json.dumps({"stage": stage, **row}, allow_nan=False) + "\n")

        for name, dataset in datasets.items():
            diagnostics[name], rows = inspect_value_pairs(
                model, dataset, difference_weight=difference_weight, phase_balanced=not memorization_test,
            )
            record_pairs(f"initial_{name}", rows)
            stats = diagnostics[name]
            print(
                f"value pair inputs | split={name} | pairs={stats['pairs']} | informative={stats['informative_pairs']} | "
                f"conflicting_collisions={stats['identical_features_with_conflicting_gap']} | "
                f"zero_gap_sensitivity={stats['informative_pairs_with_zero_sensitivity']}",
                flush=True,
            )

        def update(name, step, measurements):
            if memorization is not None:
                accuracy = measurements["train"]["preference_accuracy"]
                memorization["max_training_preference_accuracy"] = max(
                    memorization["max_training_preference_accuracy"], accuracy,
                )
                if accuracy == 1.0 and memorization["first_perfect_training_update"] is None:
                    memorization["first_perfect_training_update"] = step
            metrics = {
                "update": step, "best_update": measurements["best_update"],
                "minibatch_loss": measurements["minibatch_loss"], "gradient_norm": measurements["gradient_norm"],
                "learning_rate": measurements["learning_rate"], "best_objective_loss": measurements["best_objective_loss"],
            }
            for part in ("train", "selection"):
                metrics.update({f"{part}_{metric}": measurements[part][metric] for metric in METRICS})
                objective = measurements["selection_metric"]
                metrics[f"{part}_objective_loss"] = measurements[part][objective]
                metrics[f"{part}_combined_loss"] = measurements[part]["selection_loss"]
            children[name].log(**metrics)
            if step == 0 or step % log_interval == 0 or step == settings.value_updates:
                print(
                    f"value loss comparison | loss={name} | update={step}/{settings.value_updates} | "
                    f"train_gap_rmse={metrics['train_gap_rmse']:.4f} | "
                    f"selection_gap_rmse={metrics['selection_gap_rmse']:.4f} | "
                    f"train_rank={metrics['train_preference_accuracy']} | "
                    f"selection_rank={metrics['selection_preference_accuracy']}",
                    flush=True,
                )
            if step == settings.value_updates:
                last_diagnostics[name], rows = inspect_value_pairs(
                    model, datasets["train"], difference_weight=difference_weight,
                    phase_balanced=not memorization_test,
                )
                record_pairs(f"last_{name}_train", rows)

        _, fits = fit_paired_value_heads(
            model, datasets["train"], datasets["selection"], settings, seed=training.seed,
            difference_weight=difference_weight, loss_variants=losses,
            preserve_reference=True, selection_by_variant=True, update_callback=update,
            full_batch=memorization_test,
        )
    comparison = {}
    for name, result in fits.items():
        children[name].finish(result, plots=False)
        last = result.get("last_update")
        comparison[name] = {
            "selection_metric": result["selection_metric"], "selected_updates": result["selected_updates"],
            "kept_update": result["kept_update"], "updates": result["updates"],
            "train_before": {key: result["train_before"][key] for key in METRICS},
            "train_last": {key: last["train"][key] for key in METRICS} if last else None,
            "train_selected": {key: result["train_after"][key] for key in METRICS},
            "selection_before": {key: result["before"][key] for key in METRICS},
            "selection_last": {key: last["selection"][key] for key in METRICS} if last else None,
            "selection_selected": {key: result["after"][key] for key in METRICS},
            "history": str(children[name].directory / "metrics.jsonl"),
        }
    return {
        "mode": "offline_value_loss_comparison", "benchmark_performed": False,
        "source": source, "split": split, "initial_diagnostics": diagnostics,
        "last_training_diagnostics": last_diagnostics, "pair_diagnostics": str(pair_path),
        "loss_comparison": comparison, "memorization": memorization,
    }
