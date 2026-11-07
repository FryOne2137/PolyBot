"""Compare flat model throughput on synthetic public tensor workloads."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

from _bootstrap import add_local_source_root

add_local_source_root()

import torch
from torch.nn import functional as F

from polybot.config import ModelConfig
from polybot.infrastructure.device import select_device
from polybot.infrastructure.randomness import seed_everything
from polybot.models.policy_value_belief import PolicyValueBeliefNet
from polybot.data.token_schema import MAP_FEATURE_MAX


def inputs(batch, side, actions, history, device):
    board = torch.stack([
        torch.randint(0, largest + 1, (batch, side * side), device=device)
        for largest in MAP_FEATURE_MAX
    ], dim=-1)
    board[..., 0] = 1
    board[..., 14:16] = -1
    centres = min(2, side * side)
    board[:, :centres, 14] = 2
    board[:, :centres, 15] = torch.arange(centres, device=device)
    categorical = torch.full((batch, actions, 12), -1, dtype=torch.long, device=device)
    categorical[..., 0] = torch.randint(0, 10, (batch, actions), device=device)
    categorical[..., 1:3] = torch.randint(0, side * side, (batch, actions, 2), device=device)
    categorical[..., 4] = torch.arange(actions, device=device) % 3 - 1
    categorical[..., 5:] = 0
    return (
        board, torch.randn(batch, 9, device=device), categorical,
        torch.rand(batch, actions, 4, device=device),
        torch.ones(batch, actions, dtype=torch.bool, device=device),
        torch.zeros(batch, 12, device=device),
        torch.randint(0, 5, (batch, history, 12), device=device).float(),
        torch.ones(batch, history, dtype=torch.bool, device=device),
    )


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def measure(model, tensors, repeats, warmup, train, amp):
    device = tensors[0].device
    model.train(train)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5) if train else None
    scaler = torch.amp.GradScaler("cuda", enabled=train and amp)
    samples = []
    for index in range(warmup + repeats):
        synchronize(device)
        start = perf_counter()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        context = nullcontext() if train else torch.inference_mode()
        autocast = torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
        with context, autocast:
            output = model(*tensors, include_belief=False)
            if train:
                loss = F.cross_entropy(output.policy_logits.float(), torch.zeros(tensors[0].shape[0], dtype=torch.long, device=device))
                loss = loss + F.mse_loss(output.value.float(), torch.zeros_like(output.value).float())
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        synchronize(device)
        if index >= warmup:
            samples.append(1000 * (perf_counter() - start))
    return {"median_ms": median(samples), "minimum_ms": min(samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--map-size", type=int, default=11)
    parser.add_argument("--actions", default="32,256,1024")
    parser.add_argument("--history", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--action-interaction-dim", type=int, default=64)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--training", action="store_true", help="Measure forward/backward/AdamW as well")
    args = parser.parse_args()
    counts = [int(value) for value in args.actions.split(",")]
    if min(args.batch_size, args.map_size, args.threads, args.repeats, *counts) < 1 or args.history < 0 or args.history > 1024:
        parser.error("invalid workload size or history length")
    if args.warmup < 0 or args.map_size > 32:
        parser.error("warmup must be nonnegative and map-size <= 32")
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    rows = []
    modes = (False, True) if args.training else (False,)
    for actions in counts:
        for train in modes:
            pair = []
            for architecture in ("factorized", "relational"):
                seed_everything(2137)
                config = ModelConfig(
                    d_model=args.d_model, transformer_layers=args.layers,
                    attention_heads=args.heads, policy_architecture=architecture,
                    action_interaction_dim=args.action_interaction_dim, dropout=0.0,
                )
                model = PolicyValueBeliefNet(config).to(device)
                # Identical input tensors, independent of constructor RNG.
                seed_everything(7319)
                tensors = inputs(args.batch_size, args.map_size, actions, args.history, device)
                measurement = measure(model, tensors, args.repeats, args.warmup, train, args.amp and device.type == "cuda")
                row = {"architecture": architecture, "config": asdict(config),
                       "actions": actions, "mode": "training" if train else "inference",
                       "parameters": sum(p.numel() for p in model.parameters()), **measurement}
                pair.append(row)
                del model, tensors
            pair[1]["time_relative_to_factorized"] = pair[1]["median_ms"] / pair[0]["median_ms"]
            rows.extend(pair)
    result = {"device": str(device), "torch": torch.__version__,
              "platform": platform.platform(),
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "batch_size": args.batch_size, "map_size": args.map_size,
              "history": args.history, "threads": args.threads,
              "repeats": args.repeats, "warmup": args.warmup,
              "amp": args.amp and device.type == "cuda",
              "synthetic_workload": True, "results": rows}
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
