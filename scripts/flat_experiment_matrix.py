"""Plan paired-seed flat ablations; --execute explicitly starts the jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sqlite3
import subprocess
import sys


VARIANTS = {
    "control": {"--flat-value-updates": "0", "--flat-paired-rollback": False,
                "--flat-rollout-steps-final": "0", "--flat-exploiter-objective": False},
    "critic": {"--flat-value-updates": "8", "--flat-paired-rollback": False,
               "--flat-rollout-steps-final": "0", "--flat-exploiter-objective": False},
    "critic_credit": {"--flat-value-updates": "8", "--flat-paired-rollback": False,
                      "--flat-rollout-steps-final": "256", "--flat-exploiter-objective": False},
    "full": {"--flat-value-updates": "8", "--flat-paired-rollback": True,
             "--flat-rollout-steps-final": "256", "--flat-exploiter-objective": True},
    "weak_anchor": {"--flat-value-updates": "8", "--flat-paired-rollback": True,
                    "--flat-rollout-steps-final": "256",
                    "--flat-exploiter-objective": True,
                    "--post-bootstrap-policy-anchor-weight": "0.05",
                    "--flat-population-policy-anchor-final-weight": "0.01"},
    "exploiter": {"--flat-value-updates": "8", "--flat-paired-rollback": True,
                  "--flat-rollout-steps-final": "256", "--flat-exploiter-objective": True,
                  "--breakthrough-exploiter": True,
                  "--flat-main-exploiter-probability": "0.90"},
}


def set_option(command: list[str], name: str, value) -> list[str]:
    negative = "--no-" + name[2:]
    clean = []
    index = 0
    while index < len(command):
        token = command[index]
        if token in (name, negative) or token.startswith(name + "="):
            if "=" not in token and index + 1 < len(command) and not command[index + 1].startswith("--"):
                index += 1
        else:
            clean.append(token)
        index += 1
    if isinstance(value, bool):
        return clean + [name if value else negative]
    return clean + [name, str(value)]


def build_plan(base: list[str], seeds: list[int], variants: list[str], evaluation_seed: int):
    if any(token.split("=")[0] == "--flat-resume-state" for token in base):
        raise ValueError("ablations must share an initial model, not resume different optimizer states")
    plan = []
    for seed in seeds:
        for variant in variants:
            name = f"flat-{variant}-seed{seed}"
            command = [token for token in base if token != "--random-seed"]
            options = {
                "--flat-improvements": True,
                "--flat-cache-evaluations": True,
                "--flat-scientific-artifacts": True,
                "--breakthrough-exploiter": False,
                "--rounds": 1, "--seed": seed,
                "--evaluation-seed": evaluation_seed, "--name": name,
                **VARIANTS[variant],
            }
            for option, value in options.items():
                command = set_option(command, option, value)
            plan.append({"name": name, "variant": variant, "seed": seed,
                         "evaluation_seed": evaluation_seed, "command": command,
                         "status": "planned"})
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--evaluation-seed", type=int, default=113846927)
    parser.add_argument("--variants", default="control,critic,critic_credit,full")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    base = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not base or not any("train_flat.py" in token for token in base):
        parser.error("provide a base train_flat.py command after --")
    variants = args.variants.split(",")
    if set(variants) - VARIANTS.keys():
        parser.error("unknown ablation variant")
    try:
        plan = build_plan(base, [int(seed) for seed in args.seeds.split(",")], variants, args.evaluation_seed)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    def option_value(command, name, default):
        for index, token in enumerate(command):
            if token == name:
                return command[index + 1]
            if token.startswith(name + "="):
                return token.split("=", 1)[1]
        return default
    source_db = Path(option_value(base, "--league-db", "runs/league.sqlite"))
    if not source_db.is_absolute():
        source_db = root / source_db
    frozen_db = args.output / "initial-league.sqlite"
    def copy_database(source_path, destination_path):
        with sqlite3.connect(source_path.resolve().as_uri() + "?mode=ro", uri=True) as source:
            with sqlite3.connect(destination_path) as destination:
                source.backup(destination)
    # Planning must not multiply a multi-gigabyte league database. Freeze
    # the initial league once only when the user explicitly executes trials.
    if args.execute and source_db.exists():
        copy_database(source_db, frozen_db)
    for trial in plan:
        folder = args.output / trial["name"]
        folder.mkdir()
        destination_db = folder / "league.sqlite"
        # Independent copies avoid teaching a later trial an earlier trial's
        # counters. sqlite.backup includes committed WAL contents consistently.
        trial["initial_database"] = str(source_db.resolve())
        trial["command"] = set_option(trial["command"], "--league-db", destination_db.resolve())
        trial["command"] = set_option(trial["command"], "--checkpoint-dir", (folder / "models").resolve())
        trial["command"] = set_option(trial["command"], "--project", (folder / "runs").resolve())
    def save():
        (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    save()
    for trial in plan:
        print(shlex.join(trial["command"]))
        if args.execute:
            if frozen_db.exists():
                copy_database(frozen_db, args.output / trial["name"] / "league.sqlite")
            command = list(trial["command"])
            if command[0].lower() in ("python", "python3", "python.exe"):
                command[0] = sys.executable
            with (args.output / trial["name"] / "console.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            trial["status"] = "complete" if result.returncode == 0 else "failed"
            trial["returncode"] = result.returncode
            save()
            if result.returncode:
                raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
