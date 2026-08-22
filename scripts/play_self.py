"""Play one PolyBot checkpoint against itself and save native .polygame replays."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from _bootstrap import add_local_source_root

add_local_source_root()

from polybot.config import MCTSConfig
from polybot.data.polyenv_features import packet_to_tensors
from polybot.infrastructure.device import select_device
from polybot.infrastructure.model_io import load_model
from polybot.infrastructure.randomness import seed_everything
from polybot.models.policy_value_belief import PolicyValueBeliefNet
from polybot.planning.belief_mcts import BeliefMCTS, MCTSDecision


FORMAT_VERSION = 1
DEFAULT_TRIBES = ("Bardur", "Imperius")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _select_self_play_device(requested: str) -> torch.device:
    """Resolve auto as CUDA, then Apple MPS, then CPU for this CLI only."""
    if requested != "auto":
        return select_device(requested)
    if torch.cuda.is_available():
        return select_device("cuda")
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return select_device("mps")
    return select_device("cpu")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    """Convert one NumPy/Python action field into strict JSON data."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _selected_action(actions: Mapping[str, Any], row: int) -> Dict[str, Any]:
    """Save the complete public description of the selected legal action."""
    selected: Dict[str, Any] = {}
    for name, values in actions.items():
        try:
            value = values[row]
        except (IndexError, KeyError, TypeError):
            continue
        selected[str(name)] = _json_value(value)
    return selected


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a JSON file, including during a long self-play run."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _unique_output_dir(requested: Path) -> Path:
    """Use Ultralytics-style numeric suffixes rather than overwrite old games."""
    if not requested.exists():
        return requested
    for suffix in range(2, 10_000):
        candidate = requested.with_name(f"{requested.name}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not allocate a new output directory next to {requested}")


def _resolve_tribes(names: Sequence[str]) -> tuple[tuple[object, object], tuple[str, str]]:
    """Validate CLI tribe names with PolyEnv and return canonical spellings."""
    try:
        from PolyEnv import SUPPORTED_TRIBES, get_tribe
    except ImportError as error:
        raise RuntimeError(
            "PolyEnv is missing. Run this script in the same environment used for training."
        ) from error
    try:
        resolved = tuple(get_tribe(name) for name in names)
    except ValueError as error:
        available = ", ".join(tribe.name for tribe in SUPPORTED_TRIBES)
        raise ValueError(f"{error}. Available tribes: {available}") from error
    if len(resolved) != 2:
        raise ValueError("exactly two tribes are required")
    canonical = tuple(tribe.name for tribe in resolved)
    return (resolved[0], resolved[1]), (canonical[0], canonical[1])


def _new_env(
    seed: int,
    map_size: int,
    map_type: str,
    tribes: Sequence[object],
) -> object:
    try:
        from PolyEnv import GameEnv
    except ImportError as error:
        raise RuntimeError(
            "PolyEnv is missing. Run this script in the same environment used for training."
        ) from error
    return GameEnv(
        seed=int(seed),
        map_size=int(map_size),
        players=tuple(tribes),
        map_type=map_type,
    )


@torch.inference_mode()
def _flat_choice(
    model: PolicyValueBeliefNet,
    packet: Dict[str, object],
    device: torch.device,
    temperature: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Sample the raw policy in the same style as a flat training rollout."""
    tensors = packet_to_tensors(packet, device)
    output = model(
        tensors.map_tokens,
        tensors.global_features,
        tensors.action_categorical,
        tensors.action_continuous,
        tensors.action_mask,
        tensors.event_features,
        action_history=tensors.action_history,
        action_history_mask=tensors.action_history_mask,
        include_belief=False,
    )
    action_ids = np.asarray(packet["actions"]["action_id"], dtype=np.int64)
    logits = output.policy_logits[0, : action_ids.size].float()
    if temperature <= 1e-8:
        action_row = int(logits.argmax().item())
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
    else:
        probabilities = torch.softmax(logits / float(temperature), dim=-1).cpu().numpy()
        probabilities = probabilities.astype(np.float64)
        probabilities /= probabilities.sum()
        action_row = int(rng.choice(action_ids.size, p=probabilities))
    safe_probabilities = np.clip(probabilities, 1e-12, 1.0)
    return {
        "action_id": int(action_ids[action_row]),
        "action_row": action_row,
        "policy_probability": float(probabilities[action_row]),
        "policy_entropy": float(-(probabilities * np.log(safe_probabilities)).sum()),
        "value": float(output.value[0].item()),
    }


def _mcts_choice(
    planner: BeliefMCTS,
    env: object,
    on_progress: Optional[Callable[[int, int, int], None]] = None,
) -> Dict[str, Any]:
    decision: MCTSDecision = planner.decide(env, on_progress=on_progress)
    visits = np.asarray(decision.visit_counts, dtype=np.float64)
    priors = np.asarray(decision.priors, dtype=np.float64)
    return {
        "action_id": int(decision.action_id),
        "action_row": int(decision.action_row),
        "policy_probability": float(priors[decision.action_row]),
        "mcts_selected_visits": int(visits[decision.action_row]),
        "mcts_total_visits": int(visits.sum()),
        "mcts_valid_particles": int(decision.valid_particles),
    }


def _verify_polygame(
    path: Path,
    *,
    seed: int,
    map_size: int,
    map_type: str,
    tribes: Sequence[object],
    expected_moves: int,
    expected_winner: int,
    expected_terminated: bool,
) -> None:
    """Load a native replay through PolyEnv and validate its final state."""
    replay = _new_env(seed, map_size, map_type, tribes)
    load = getattr(replay, "load", None)
    if not callable(load):
        raise RuntimeError(
            "installed PolyEnv does not provide GameEnv.load(); update PolyEnv"
        )
    observation = load(path)
    action_ids = replay.replay_action_ids()
    if len(action_ids) != expected_moves:
        raise RuntimeError(
            f"native replay move count mismatch: {len(action_ids)} != {expected_moves}"
        )
    replay_winner = int(observation.get("winner", -1))
    if replay_winner != expected_winner:
        raise RuntimeError(
            f"native replay winner mismatch: {replay_winner} != {expected_winner}"
        )
    if bool(replay.is_done()) != expected_terminated:
        raise RuntimeError("native replay terminal state does not match the game")


def _game_summary(game: Mapping[str, Any], filename: str) -> Dict[str, Any]:
    return {
        "file": filename,
        "game_index": int(game["game_index"]),
        "map_seed": int(game["map_seed"]),
        "moves": int(game["moves"]),
        "winner_player": int(game["winner_player"]),
        "winner_tribe": game["winner_tribe"],
        "terminated": bool(game["terminated"]),
        "truncated": bool(game["truncated"]),
        "replay_verified": bool(game["replay_verified"]),
    }


def _totals(
    games: Sequence[Mapping[str, Any]],
    tribe_names: Sequence[str],
) -> Dict[str, Any]:
    move_counts = [int(game["moves"]) for game in games]
    wins_by_player = {
        str(player): sum(
            int(game["winner_player"] == player) for game in games
        )
        for player in range(2)
    }
    wins_by_tribe = {
        name: sum(int(game["winner_tribe"] == name) for game in games)
        for name in dict.fromkeys(tribe_names)
    }
    return {
        "completed_games": len(games),
        "wins_by_player": wins_by_player,
        "wins_by_tribe": wins_by_tribe,
        "draws": sum(int(game["winner_player"] < 0) for game in games),
        "truncated_games": sum(int(game["truncated"]) for game in games),
        "total_moves": sum(move_counts),
        "mean_moves": float(np.mean(move_counts)) if move_counts else 0.0,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Play one PolyBot checkpoint against itself and save one native "
            ".polygame replay per game plus manifest.json."
        )
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to a .pt checkpoint")
    parser.add_argument("--mode", choices=("flat", "mcts"), default="flat")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--map-size", type=int, default=11)
    parser.add_argument("--map-type", choices=("lakes", "drylands"), default="lakes")
    parser.add_argument(
        "--tribes",
        nargs=2,
        metavar=("TRIBE_1", "TRIBE_2"),
        default=DEFAULT_TRIBES,
        help="Two PolyEnv tribes (default: Bardur Imperius)",
    )
    parser.add_argument(
        "--moves",
        "--max-moves",
        dest="max_moves",
        type=int,
        default=300,
        help="Maximum moves per game (default: 300)",
    )
    parser.add_argument("--seed", type=int, default=2137)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Compute device; auto tries CUDA, then MPS, then CPU",
    )
    parser.add_argument("--simulations", type=int, default=32, help="MCTS only")
    parser.add_argument("--particles", type=int, default=2, help="MCTS only")
    parser.add_argument("--inference-batch-size", type=int, default=32, help="MCTS only")
    parser.add_argument("--mcts-threads", type=int, default=1, help="MCTS only")
    parser.add_argument("--mcts-max-actions", type=int, default=1024, help="MCTS only")
    parser.add_argument(
        "--save-path",
        "--output-dir",
        dest="output_dir",
        type=Path,
        help=(
            "Directory for saved games; default: "
            "runs/self_play_games/MODEL_MODE"
        ),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace, max_board_side: int) -> None:
    if args.games < 1:
        raise ValueError("--games must be positive")
    if args.map_size < 1 or args.map_size > max_board_side:
        raise ValueError(f"--map-size must be between 1 and {max_board_side}")
    if args.max_moves < 1:
        raise ValueError("--max-moves must be positive")
    if args.temperature < 0.0:
        raise ValueError("--temperature cannot be negative")
    if args.simulations < 1 or args.particles < 1:
        raise ValueError("--simulations and --particles must be positive")
    if args.inference_batch_size < 1 or args.mcts_threads < 1:
        raise ValueError("--inference-batch-size and --mcts-threads must be positive")


def main() -> None:
    args = _parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {model_path}")

    tribes, tribe_names = _resolve_tribes(args.tribes)
    device = _select_self_play_device(args.device)
    model, model_config = load_model(model_path, device)
    model.eval()
    _validate_args(args, model_config.max_board_side)
    seed = seed_everything(args.seed)
    rng = np.random.default_rng(seed ^ 0x51F1A7)

    requested_output = args.output_dir or Path("runs/self_play_games") / (
        f"{model_path.stem}_{args.mode}"
    )
    output_dir = _unique_output_dir(requested_output.expanduser().resolve())
    output_dir.mkdir(parents=True)
    device_reason = (
        "auto priority: cuda -> mps -> cpu"
        if args.device == "auto"
        else f"requested explicitly: {args.device}"
    )
    tqdm.write(f"device: {device} ({device_reason})")
    tqdm.write(f"checkpoint: {model_path}")
    tqdm.write(f"save path: {output_dir}")

    mcts_config = MCTSConfig(
        simulations=args.simulations,
        particles=args.particles,
        temperature=args.temperature,
        inference_batch_size=args.inference_batch_size,
        native_threads=args.mcts_threads,
        native_max_actions=args.mcts_max_actions,
        # Saved games measure the checkpoint itself, without teacher rules.
        strategic_prior_weight=0.0,
        teacher_prior_weight=0.0,
    )
    planner = (
        BeliefMCTS(
            model,
            device,
            mcts_config,
            map_size=args.map_size,
            map_type=args.map_type,
        )
        if args.mode == "mcts"
        else None
    )

    common_config: Dict[str, Any] = {
        "mode": args.mode,
        "games": args.games,
        "map_size": args.map_size,
        "map_type": args.map_type,
        "tribes": list(tribe_names),
        "max_moves": args.max_moves,
        "base_seed": seed,
        "temperature": args.temperature,
        "device": str(device),
        "device_request": args.device,
        "device_auto_priority": ["cuda", "mps", "cpu"],
        "simulations": args.simulations if args.mode == "mcts" else None,
        "particles": args.particles if args.mode == "mcts" else None,
        "inference_batch_size": args.inference_batch_size if args.mode == "mcts" else None,
        "mcts_threads": args.mcts_threads if args.mode == "mcts" else None,
        "mcts_max_actions": args.mcts_max_actions if args.mode == "mcts" else None,
    }
    manifest: Dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "status": "running",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "checkpoint": {
            "path": str(model_path),
            "sha256": _sha256(model_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "replay_format": "polyenv-game",
        "replay_extension": ".polygame",
        "config": common_config,
        "games": [],
        "totals": _totals([], tribe_names),
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    completed_summaries: list[Dict[str, Any]] = []
    try:
        with tqdm(
            total=args.games,
            desc=f"self-play {args.mode} [{device}]",
            unit="game",
            dynamic_ncols=True,
            position=0,
        ) as games_progress:
            for game_index in range(args.games):
                map_seed = seed + game_index
                env = _new_env(
                    map_seed,
                    args.map_size,
                    args.map_type,
                    tribes,
                )
                winner = -1
                moves = 0
                with tqdm(
                    total=args.max_moves,
                    desc=f"game {game_index + 1}/{args.games}",
                    unit="move",
                    dynamic_ncols=True,
                    leave=True,
                    position=1,
                ) as move_progress:
                    while not env.is_done() and moves < args.max_moves:
                        packet = env.model_request_numpy()
                        player = int(
                            packet["obs"].get(
                                "current_player", env.current_player()
                            )
                        )
                        tribe = (
                            tribe_names[player]
                            if 0 <= player < len(tribe_names)
                            else str(player)
                        )

                        def report_search(
                            completed: int, total: int, particles: int
                        ) -> None:
                            move_progress.set_postfix(
                                player=tribe,
                                search=f"{completed}/{total}",
                                particles=particles,
                            )

                        choice = (
                            _mcts_choice(planner, env, report_search)
                            if planner is not None
                            else _flat_choice(
                                model,
                                packet,
                                device,
                                args.temperature,
                                rng,
                            )
                        )
                        action_row = int(choice.pop("action_row"))
                        action = _selected_action(packet["actions"], action_row)
                        ok, done, terminal_reward, winner, _ = env.step_fast(
                            int(choice["action_id"])
                        )
                        if not ok:
                            raise RuntimeError(
                                "PolyEnv rejected the selected action in game "
                                f"{game_index + 1}, move {moves}"
                            )
                        moves += 1
                        move_progress.update(1)
                        move_progress.set_postfix(
                            player=tribe,
                            action=action.get("type_fullname", choice["action_id"]),
                        )

                terminated = bool(env.is_done())
                game_data: Dict[str, Any] = {
                    "game_index": game_index + 1,
                    "map_seed": map_seed,
                    "moves": moves,
                    "winner_player": int(winner),
                    "winner_tribe": (
                        tribe_names[winner]
                        if 0 <= winner < len(tribe_names)
                        else None
                    ),
                    "terminated": terminated,
                    "truncated": bool(not terminated),
                    "replay_verified": False,
                }
                game_filename = f"game_{game_index + 1:03d}.polygame"
                game_path = output_dir / game_filename
                save = getattr(env, "save", None)
                if not callable(save):
                    raise RuntimeError(
                        "installed PolyEnv does not provide GameEnv.save(); "
                        "update PolyEnv"
                    )
                save(game_path)
                _verify_polygame(
                    game_path,
                    seed=map_seed,
                    map_size=args.map_size,
                    map_type=args.map_type,
                    tribes=tribes,
                    expected_moves=moves,
                    expected_winner=int(winner),
                    expected_terminated=terminated,
                )
                game_data["replay_verified"] = True

                summary = _game_summary(game_data, game_filename)
                completed_summaries.append(summary)
                manifest["games"] = completed_summaries
                manifest["totals"] = _totals(
                    completed_summaries, tribe_names
                )
                manifest["updated_at"] = _utc_now()
                _write_json(manifest_path, manifest)
                result = summary["winner_tribe"] or (
                    "move limit" if summary["truncated"] else "draw"
                )
                tqdm.write(
                    f"game {game_index + 1}/{args.games}: "
                    f"moves={moves}, result={result}, saved={output_dir / game_filename}"
                )
                games_progress.update(1)
                games_progress.set_postfix(
                    player_0=manifest["totals"]["wins_by_player"]["0"],
                    player_1=manifest["totals"]["wins_by_player"]["1"],
                    draws=manifest["totals"]["draws"],
                )
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["updated_at"] = _utc_now()
        _write_json(manifest_path, manifest)
        raise

    manifest["status"] = "complete"
    manifest["updated_at"] = _utc_now()
    _write_json(manifest_path, manifest)
    print(json.dumps({"output_dir": str(output_dir), **manifest["totals"]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
