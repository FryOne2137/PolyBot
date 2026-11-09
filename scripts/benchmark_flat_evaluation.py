"""Compare real flat evaluation throughput on one checkpoint and device.

Short horizons make this a speed probe, not a playing-strength benchmark.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
from time import perf_counter

from _bootstrap import add_local_source_root

add_local_source_root()

import torch

from polybot.config import FlatCurriculumConfig, TrainingConfig
from polybot.evaluation.matches import evaluate_direct_matches
from polybot.infrastructure.device import select_device
from polybot.infrastructure.model_io import load_model
from polybot.training.flat_experiments import isolated_rng, model_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps', 'auto'), default='cuda')
    parser.add_argument('--games', type=int, default=40)
    parser.add_argument('--max-moves', type=int, default=100, help='Short 11x11 probe horizon; mixed maps scale by area')
    parser.add_argument('--seed', type=int, default=351531851)
    parser.add_argument('--history', type=int, default=32)
    parser.add_argument('--num-threads', type=int, default=4)
    parser.add_argument('--batch-games', type=int, default=64)
    parser.add_argument('--warmup-moves', type=int, default=2)
    parser.add_argument('--modes', default='single,mixed-one-size,mixed-nearby,mixed-exact')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    modes = args.modes.split(',')
    allowed = {'single', 'mixed-one-size', 'mixed-nearby', 'mixed-exact'}
    if not modes or any(mode not in allowed for mode in modes):
        parser.error('unknown benchmark mode')
    if args.games < 2 or args.games % 2 or args.batch_games < 2 or args.batch_games % 2:
        parser.error('games and batch-games must be positive even numbers')
    if args.max_moves < 1 or args.warmup_moves < 0 or not 0 <= args.history <= 1024 or args.num_threads < 1:
        parser.error('invalid move limit, history or thread count')
    device = select_device(args.device)
    model, config = load_model(args.model, device)
    training = TrainingConfig(device=str(device), seed=args.seed, max_moves=args.max_moves,
        flat_evaluation_batch_games=args.batch_games)
    if config.max_board_side < 16 and any(mode in {'mixed-nearby', 'mixed-exact'} for mode in modes):
        parser.error('mixed probe needs a model supporting boards through 16x16')
    report = dict(
        purpose='short throughput probe; not a playing-strength evaluation',
        python=platform.python_version(), torch=torch.__version__, cuda_runtime=torch.version.cuda,
        device=str(device), checkpoint=str(args.model), model_config=asdict(config),
        fingerprint=model_fingerprint(model), settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        cases=[])
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        report.update(device_name=properties.name, vram_bytes=properties.total_memory)
    for mode in modes:
        local = training
        if mode != 'single':
            settings = FlatCurriculumConfig(map_sizes=(11,)) if mode == 'mixed-one-size' else FlatCurriculumConfig()
            local = replace(training, flat_curriculum=settings,
                flat_evaluation_size_bucketing='exact' if mode == 'mixed-exact' else 'nearby')
        print(f'flat throughput probe | mode={mode} | games={args.games} | base_horizon={args.max_moves}', flush=True)
        if args.warmup_moves:
            with isolated_rng():
                evaluate_direct_matches(model, model, device, replace(local, max_moves=args.warmup_moves), args.games,
                    visible_action_history=args.history, num_threads=args.num_threads,
                    model_a_label=mode + ' warmup', model_b_label='same policy')
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        started = perf_counter()
        with isolated_rng():
            result = evaluate_direct_matches(model, model, device, local, args.games,
                visible_action_history=args.history, num_threads=args.num_threads,
                model_a_label=mode, model_b_label='same policy')
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - started
        row = dict(mode=mode, wall_seconds=elapsed, seconds_per_game=elapsed/args.games,
            environment_steps=result['moves'], environment_steps_per_second=result['moves']/elapsed,
            mean_moves=result['mean_moves'], model_score=result['model_a_score'],
            **{key: value for key, value in result.items() if key.startswith('flat_evaluation_')})
        report['cases'].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(row, indent=2), flush=True)
    print(f'flat throughput report: {args.output}', flush=True)


if __name__ == '__main__':
    main()
