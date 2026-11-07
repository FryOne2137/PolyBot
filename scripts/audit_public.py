"""Read-only publication checks. Findings never print matched secret values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess


PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "github_token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})"),
    "aws_access_key": re.compile(rb"(?:AKIA|ASIA)[0-9A-Z]{16}"),
    "slack_token": re.compile(rb"xox[baprs]-[0-9A-Za-z-]{20,}"),
    "api_token": re.compile(rb"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}"),
    "credential_url": re.compile(rb"https?://[^\s/:@]+:[^\s/@]+@"),
    "possible_secret_assignment": re.compile(
        rb"(?i)\b(?:api[_-]?key|password|client[_-]?secret|access[_-]?token)\b\s*[:=]\s*['\"][^'\"\s]{12,}['\"]"
    ),
    "personal_path": re.compile(rb"(?:/(?:Users|home)/[A-Za-z0-9._-]+/|[A-Za-z]:[\\/]Users[\\/][A-Za-z0-9._-]+[\\/])"),
}
ARTIFACT_SUFFIXES = (".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".sqlite", ".sqlite3", ".db", ".polygame")
LOCAL_DIRECTORIES = {"models", "runs", "tmp", ".idea", ".vscode", ".venv", "venv", "__pycache__"}
MAX_TEXT_BYTES = 2_000_000


def git(root: Path, *arguments: str, input_bytes=None):
    return subprocess.check_output(["git", "-C", str(root), *arguments], input=input_bytes, stderr=subprocess.PIPE)


def candidate_paths(root: Path):
    output = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    return sorted(set(name.decode("utf-8") for name in output.split(b"\0") if name))


def is_private_artifact(name: str):
    path = Path(name.lower())
    return (
        (bool(path.parts) and path.parts[0] in LOCAL_DIRECTORIES)
        or bool(set(path.parts) & {"__pycache__", ".idea", ".vscode", ".venv"})
        or path.name in {".ds_store", "thumbs.db", ".env", ".envrc"}
        or (path.name.startswith(".env.") and path.name != ".env.example")
        or name.lower().endswith(ARTIFACT_SUFFIXES)
        or any(marker in path.name for marker in (".sqlite-", ".sqlite3-", ".db-"))
        or path.suffix.lower() in {".pem", ".key"}
        or path.name.lower().startswith(("credentials", "service-account"))
    )


def scan_bytes(data: bytes, path: str, *, object_id=None):
    if b"\0" in data:
        return []
    findings = []
    for kind, pattern in PATTERNS.items():
        matches = list(pattern.finditer(data))
        if matches:
            findings.append({
                "severity": "warning" if kind == "personal_path" else "error",
                "kind": kind, "path": path,
                "lines": sorted({data[:m.start()].count(b"\n") + 1 for m in matches}),
                "count": len(matches),
                **({"object": object_id} if object_id is not None else {}),
            })
    return findings


def audit_worktree(root: Path):
    findings = []
    scanned = 0
    for name in candidate_paths(root):
        path = root / name
        if not path.exists() and not path.is_symlink():
            continue  # A tracked deletion is absent from the publication.
        if is_private_artifact(name):
            findings.append({"severity": "error", "kind": "local_artifact", "path": name})
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            findings.append({"severity": "error", "kind": "symlink_or_external_path", "path": name})
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > MAX_TEXT_BYTES:
            findings.append({"severity": "warning", "kind": "large_unscanned_file", "path": name, "bytes": size})
            continue
        scanned += 1
        findings.extend(scan_bytes(path.read_bytes(), name))
    return {"files_scanned": scanned, "findings": findings}


def audit_history(root: Path):
    objects = git(root, "rev-list", "--objects", "--all")
    checked = git(root, "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize) %(rest)", input_bytes=objects)
    blobs = []
    for line in checked.decode("utf-8", errors="replace").splitlines():
        fields = line.split(" ", 3)
        if len(fields) >= 3 and fields[1] == "blob":
            blobs.append((fields[0], int(fields[2]), fields[3] if len(fields) == 4 else ""))
    findings = []
    scanned = 0
    process = subprocess.Popen(["git", "-C", str(root), "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for object_id, size, name in blobs:
            if size > MAX_TEXT_BYTES:
                continue
            process.stdin.write((object_id + "\n").encode())
            process.stdin.flush()
            header = process.stdout.readline().split()
            data = process.stdout.read(int(header[2]))
            process.stdout.read(1)
            scanned += 1
            findings.extend(scan_bytes(data, name, object_id=object_id))
    finally:
        process.stdin.close()
        process.stdout.close()
        process.wait()
    emails = set(git(root, "log", "--all", "--format=%ae%n%ce").decode().splitlines()) if blobs else set()
    non_noreply = sum(not email.endswith("@users.noreply.github.com") for email in emails if email)
    artifacts = [(oid, size, name) for oid, size, name in blobs if is_private_artifact(name)]
    if artifacts:
        findings.append({"severity": "warning", "kind": "historical_local_artifacts",
                         "objects": len(artifacts), "bytes": sum(size for _, size, _ in artifacts)})
    if non_noreply:
        findings.append({"severity": "warning", "kind": "historical_author_email",
                         "unique_non_noreply_addresses": non_noreply})
    return {"blobs": len(blobs), "text_size_blobs_scanned": scanned, "findings": findings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--require-clean-history", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.require_clean_history and not args.history:
        parser.error("--require-clean-history requires --history")
    report = {"worktree": audit_worktree(args.repo), "history_scanned": args.history,
              "limitations": "Pattern-based text scan, not a guarantee of absence of secrets. Large/binary blobs are not content-scanned."}
    if args.history:
        report["history"] = audit_history(args.repo)
        if args.require_clean_history:
            for finding in report["history"]["findings"]:
                finding["severity"] = "error"
    findings = report["worktree"]["findings"] + report.get("history", {}).get("findings", [])
    report["errors"] = sum(item["severity"] == "error" for item in findings)
    report["warnings"] = sum(item["severity"] == "warning" for item in findings)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
    raise SystemExit(2 if report["errors"] else 0)


if __name__ == "__main__":
    main()
