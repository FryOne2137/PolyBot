"""Export reviewed source without Git history, models or local experiment data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

from audit_public import audit_worktree, candidate_paths, is_private_artifact


ROOT_FILES = {
    ".gitignore", ".gitattributes", "LICENSE", "README.md",
    "CITATION.cff", "pyproject.toml", "requirements.txt",
}
SOURCE_DIRECTORIES = {"src", "scripts", "tests", "docs", ".github"}
SOURCE_SUFFIXES = {".py", ".md", ".txt", ".toml", ".json", ".yml", ".yaml", ".svg", ".png", ".pdf", ".webp", ".cff"}


def export(root: Path, destination: Path):
    report = audit_worktree(root)
    if report["findings"]:
        raise ValueError("publication audit failed; run audit_public.py and review the redacted findings")
    files = {}
    for name in candidate_paths(root):
        path = root / name
        parts = PurePosixPath(name).parts
        if not path.is_file() or is_private_artifact(name):
            continue
        allowed = name in ROOT_FILES or (
            parts[0] in SOURCE_DIRECTORIES and path.suffix.lower() in SOURCE_SUFFIXES
        )
        if not allowed:
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("source export refuses symlinks and external paths")
        files[name] = path.read_bytes()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    manifest = {"schema_version": 1, "source_revision": revision,
                "includes_working_tree_changes": True, "git_history_included": False,
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}}
    files["PUBLIC_EXPORT.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle, zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    return {"archive": str(destination), "files": len(files), "git_history_included": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("dist/polybot-public.zip"))
    args = parser.parse_args()
    try:
        result = export(args.repo, args.output)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
