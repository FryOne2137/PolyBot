"""Check the runtime and native APIs without printing local installation paths."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import platform
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Reset a real native environment and check its public packet")
    args = parser.parse_args()
    result = {"python": platform.python_version(), "platform": platform.system(), "packages": {}, "errors": []}
    for name in ("numpy", "torch", "tqdm", "matplotlib", "PolyEnv"):
        try:
            distribution = metadata.distribution(name)
            entry = {"version": distribution.version}
            if name == "PolyEnv":
                direct = json.loads(distribution.read_text("direct_url.json") or "{}")
                entry["source_commit"] = direct.get("vcs_info", {}).get("commit_id")
            result["packages"][name] = entry
        except metadata.PackageNotFoundError:
            result["errors"].append(f"missing_package:{name}")
    try:
        import torch
        import PolyEnv
        result["devices"] = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(), "cpu": True}
        result["torch_amp_api"] = hasattr(torch.amp, "GradScaler")
        required = {
            "VectorGameEnv": ("reset", "step_into"),
            "SelfPlayPool": ("select_leaves_masked", "select_leaves_masked_into", "submit_beliefs", "step_into_checked"),
            "GameEnv": ("model_request_numpy", "full_map_numpy", "make_belief_env"),
        }
        result["native_api"] = {}
        for name, methods in required.items():
            cls = getattr(PolyEnv, name, None)
            result["native_api"][name] = {method: cls is not None and hasattr(cls, method) for method in methods}
        if not result["torch_amp_api"] or not all(ok for methods in result["native_api"].values() for ok in methods.values()):
            result["errors"].append("unsupported_torch_or_native_api")
        if args.smoke:
            engine = PolyEnv.VectorGameEnv(num_envs=1, map_size=11, num_threads=1, max_actions=1024,
                                           seed=2137, players=(PolyEnv.Bardur, PolyEnv.Imperius), visible_event_history=0)
            batch = engine.reset(seed=2137)
            result["smoke"] = {"public_map_shape": list(batch["map_tokens"].shape),
                               "legal_actions": int(batch["action_mask"].sum())}
            expected = {"own_techs": (1, 25), "own_tribe": (1,),
                        "initial_player_count": (1,), "player_scores": (1, 16),
                        "player_scores_mask": (1, 16)}
            result["smoke"]["context_fields"] = {
                name: name in batch and tuple(batch[name].shape) == shape
                for name, shape in expected.items()
            }
            if tuple(batch["map_tokens"].shape) != (1, 121, 24) or not all(
                result["smoke"]["context_fields"].values()
            ):
                result["errors"].append("unsupported_observation_context_api")
    except (ImportError, AttributeError, RuntimeError, ValueError, TypeError) as error:
        # Detailed exceptions may contain private paths; report the class only.
        result["errors"].append(f"runtime_check_failed:{type(error).__name__}")
    print(json.dumps(result, indent=2))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
