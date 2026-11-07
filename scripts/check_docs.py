"""Offline checks for local documentation links, CLI options and figure freshness."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shlex
import struct
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET


def headings(text):
    result = set()
    counts = {}
    fence = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            fence = None if fence == marker else marker
            continue
        if fence is None and re.match(r"^#{1,6}\s", stripped):
            title = re.sub(r"^#+\s+", "", stripped).strip().strip("#").strip()
            slug = re.sub(r"[^\w\s-]", "", title.lower()).replace(" ", "-")
            duplicate = counts.get(slug, 0)
            counts[slug] = duplicate + 1
            result.add(slug + (f"-{duplicate}" if duplicate else ""))
    return result


def options(root, script):
    paths = [root / script]
    if Path(script).name in {"train_flat.py", "train_belief.py", "train_league_mcts.py"}:
        paths.append(root / "scripts/train_league.py")
    available = {"--help"}
    trees = []
    helpers = set()
    for path in paths:
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        trees.append(tree)
        helpers.update(node.func.id for node in ast.walk(tree)
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                       and node.func.id in {"add_environment_options", "add_run_options"})
    if helpers:
        common = ast.parse((root / "src/polybot/training/cli.py").read_text(encoding="utf-8"))
        trees.extend(node for node in common.body if isinstance(node, ast.FunctionDef) and node.name in helpers)
    for tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
                continue
            names = [arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--")]
            available.update(names)
            optional_boolean = any(keyword.arg == "action" and isinstance(keyword.value, ast.Attribute)
                                   and keyword.value.attr == "BooleanOptionalAction" for keyword in node.keywords)
            if optional_boolean:
                available.update("--no-" + name[2:] for name in names)
    if Path(script).name in {"train_flat.py", "train_belief.py", "train_league_mcts.py"}:
        available.discard("--training-stage")
    return available


def check_command(root, tokens):
    errors = []
    if len(tokens) < 2 or tokens[0] not in {"python", "python3", "py"} or not tokens[1].endswith(".py"):
        return errors
    script = tokens[1]
    if not (root / script).is_file():
        return [f"missing script: {script}"]
    end = tokens.index("--") if "--" in tokens else len(tokens)
    available = options(root, script)
    for token in tokens[2:end]:
        if token.startswith("--") and token.split("=", 1)[0] not in available:
            errors.append(f"unknown option for {script}: {token.split('=', 1)[0]}")
    if end < len(tokens):
        errors.extend(check_command(root, tokens[end + 1:]))
    return errors


def check(root: Path):
    documents = [*root.glob("*.md"), *(root / "docs").rglob("*.md"), *(root / ".github").rglob("*.md")]
    errors = []
    commands = 0
    for path in documents:
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            target = target.split(' "', 1)[0].strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc:
                continue
            destination = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path.resolve()
            if not destination.is_relative_to(root.resolve()) or not destination.exists():
                errors.append(f"{path.relative_to(root)}: broken local link {target}")
            elif parsed.fragment and destination.suffix == ".md" and unquote(parsed.fragment) not in headings(destination.read_text(encoding="utf-8")):
                errors.append(f"{path.relative_to(root)}: missing heading {target}")
        fence = None
        for number, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("```", "~~~")):
                marker = stripped[:3]
                fence = None if fence == marker else marker
            elif fence is not None and re.match(r"^(python|python3|py)\s", stripped):
                commands += 1
                try:
                    for error in check_command(root, shlex.split(stripped)):
                        errors.append(f"{path.relative_to(root)}:{number}: {error}")
                except ValueError as error:
                    errors.append(f"{path.relative_to(root)}:{number}: invalid command quoting: {error}")
        if fence is not None:
            errors.append(f"{path.relative_to(root)}: unclosed code fence")
    assets = root / "docs/assets"
    manifest_file = assets / "diagrams.json"
    if not manifest_file.is_file():
        errors.append("missing generated diagram manifest")
    else:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        actual = hashlib.sha256((root / manifest["generator"]).read_bytes()).hexdigest()
        if actual != manifest["generator_sha256"]:
            errors.append("diagrams are stale; rerun scripts/render_diagrams.py")
        for spec in manifest["figures"].values():
            svg = assets / spec["svg"]
            png = assets / spec["png"]
            if not svg.is_file() or not png.is_file():
                errors.append(f"missing diagram output: {spec['svg']} / {spec['png']}")
                continue
            try:
                ET.parse(svg)
            except ET.ParseError:
                errors.append(f"invalid SVG: {spec['svg']}")
            data = png.read_bytes()
            if not data.startswith(b"\x89PNG\r\n\x1a\n") or struct.unpack(">II", data[16:24]) != (2700, 1500):
                errors.append(f"invalid PNG size/format: {spec['png']}")
        for alias, canonical in manifest["compatibility_copies"].items():
            if not (root / alias).is_file() or (root / alias).read_bytes() != (assets / canonical).read_bytes():
                errors.append(f"stale compatibility figure: {alias}")
    return {"documents": len(documents), "commands": commands, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = check(args.repo)
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if result["errors"] else 0)


if __name__ == "__main__":
    main()
