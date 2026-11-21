"""Keep map roles fixed when saved pairs acquire more labels or new alternatives."""

from __future__ import annotations

from hashlib import sha256
import json
import os
import shutil

from .saved_value_pairs import split_value_pairs


def snapshot_pair_dataset(directory, output):
    """Keep the exact diagnostic targets after a resumable collection evolves."""
    directory = directory.resolve()
    config = json.loads((directory / "config.json").read_text(encoding="utf-8-sig"))
    label_file = config.get("labels_file", "labelled_pairs.jsonl")
    if label_file not in ("weighted_pairs.jsonl", "labelled_pairs.jsonl"):
        raise ValueError("invalid pair snapshot label file")
    names = ["config.json", "pairs.jsonl", "rollouts.jsonl", label_file]
    contents = {name: (directory / name).read_bytes() for name in names}
    output.mkdir(parents=True, exist_ok=False)
    for name, content in contents.items():
        (output / name).write_bytes(content)
    labels = [json.loads(line) for line in contents[label_file].splitlines() if line.strip()]
    for row in labels:
        for branch in row["branches"].values():
            source = directory / branch["successor_file"]
            if source.parent != directory or source.suffix != ".npz":
                raise ValueError("invalid snapshot successor path")
            shutil.copyfile(source, output / source.name)
    if any((directory / name).read_bytes() != content for name, content in contents.items()):
        raise ValueError("pair dataset changed while creating its training snapshot")
    return output


def persistent_pair_split(examples, *, seed, selection_fraction, path):
    if not path.exists():
        _, original = split_value_pairs(examples, seed=seed, selection_fraction=selection_fraction)
        roles = {str(map_seed): name for name in ("train", "selection") for map_seed in original[name]["map_seeds"]}
        manifest = {"schema": 1, "seed": seed, "selection_fraction": selection_fraction, "map_roles": roles}
    else:
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        if (manifest.get("schema") != 1 or manifest["seed"] != seed
                or manifest["selection_fraction"] != selection_fraction):
            raise ValueError("saved value map split differs from the requested split settings")
        roles = manifest["map_roles"]
        if any(value not in ("train", "selection") for value in roles.values()):
            raise ValueError("invalid saved map roles")
    for example in examples:
        key = str(example.game_key[0])
        if key not in roles:
            fraction = int.from_bytes(sha256(f"{seed}:{key}".encode()).digest()[:8], "big") / 2**64
            roles[key] = "selection" if fraction < selection_fraction else "train"
    datasets = {name: [row for row in examples if roles[str(row.game_key[0])] == name] for name in ("train", "selection")}
    if not all(datasets.values()):
        raise ValueError("persistent value split needs labelled maps in both train and selection")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    report = {"seed": seed, "selection_fraction": selection_fraction, "unit": "whole_map_seed",
              "method": "persistent_map_roles", "manifest": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
    for name, rows in datasets.items():
        report[name] = {"map_seeds": sorted({int(row.game_key[0]) for row in rows}),
                        "pairs": len(rows) // 2, "pair_ids": list(dict.fromkeys(int(row.game_key[-1]) for row in rows))}
    return datasets, report
