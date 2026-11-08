from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import zipfile

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
        spec.loader.exec_module(module)
    return module


@pytest.fixture
def source_repo(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])
    git("config", "user.name", "Public Test")
    git("config", "user.email", "123+test@users.noreply.github.com")
    (root / ".gitignore").write_text("/models/\n/runs/\n/tmp/\n__pycache__/\n.env\n")
    (root / "README.md").write_text("# Example\n")
    (root / "src/polybot/models").mkdir(parents=True)
    (root / "src/polybot/models/new.py").write_text("class Model: pass\n")
    git("add", ".")
    git("commit", "-qm", "Initial public source")
    return root, git


def test_ignore_and_audit_preserve_python_models_but_exclude_training_weights(source_repo):
    audit = script("audit_public")
    root, git = source_repo
    (root / "models").mkdir()
    (root / "models/local.pt").write_bytes(b"private weights")
    paths = audit.candidate_paths(root)
    assert "src/polybot/models/new.py" in paths
    assert "models/local.pt" not in paths
    assert not audit.is_private_artifact("src/polybot/models/new.py")
    assert audit.is_private_artifact("MODELS/private.PT")
    assert not audit.audit_worktree(root)["findings"]
    git("add", "-f", "models/local.pt")
    findings = audit.audit_worktree(root)["findings"]
    assert any(f["kind"] == "local_artifact" for f in findings)


def test_secret_audit_redacts_values_and_history_detects_removed_credentials(source_repo):
    audit = script("audit_public")
    root, git = source_repo
    token = "ghp_" + "A" * 36
    findings = audit.scan_bytes(token.encode(), "example.py")
    assert findings[0]["kind"] == "github_token"
    assert token not in json.dumps(findings)
    secret = root / "removed.txt"
    secret.write_text(token)
    git("add", "removed.txt")
    git("commit", "-qm", "Temporary private data")
    git("rm", "-q", "removed.txt")
    git("commit", "-qm", "Remove temporary data")
    assert not audit.audit_worktree(root)["findings"]
    history = audit.audit_history(root)
    assert any(f["kind"] == "github_token" for f in history["findings"])
    assert token not in json.dumps(history)


def test_history_audit_reports_old_weights_and_author_metadata_without_email(source_repo):
    audit = script("audit_public")
    root, git = source_repo
    (root / "models").mkdir()
    (root / "models/old.pt").write_bytes(b"old experiment")
    git("add", "-f", "models/old.pt")
    email = "private" + "@" + "example.invalid"
    git("-c", "user.email=" + email, "commit", "-qm", "Old private experiment")
    history = audit.audit_history(root)
    assert {f["kind"] for f in history["findings"]} >= {"historical_local_artifacts", "historical_author_email"}
    assert email not in json.dumps(history)


def test_source_export_is_reproducible_and_contains_current_model_source(source_repo, tmp_path):
    exporter = script("export_public")
    root, _ = source_repo
    (root / "src/polybot/models/new.py").write_text("class ImprovedModel: pass\n")
    (root / "models").mkdir()
    (root / "models/private.pt").write_bytes(b"local model")
    (root / ".env").write_text("local settings")
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    exporter.export(root, first)
    exporter.export(root, second)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert "src/polybot/models/new.py" in archive.namelist()
        assert all(not n.startswith((".git/", "models/", "runs/")) and n != ".env" for n in archive.namelist())
        assert archive.read("src/polybot/models/new.py") == b"class ImprovedModel: pass\n"
        manifest = json.loads(archive.read("PUBLIC_EXPORT.json"))
        assert not manifest["git_history_included"]
        for name, digest in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    with pytest.raises(FileExistsError):
        exporter.export(root, first)


def test_source_export_refuses_personal_paths_and_symlinks(source_repo, tmp_path):
    exporter = script("export_public")
    root, _ = source_repo
    source = root / "src/private.py"
    source.write_text("local = '" + "/" + "Users/someone/project/" + "'\n")
    with pytest.raises(ValueError, match="audit failed"):
        exporter.export(root, tmp_path / "blocked.zip")
    source.unlink()
    external = tmp_path / "outside.txt"
    external.write_text("outside source tree")
    try:
        (root / "src/link.py").symlink_to(external)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="audit failed"):
        exporter.export(root, tmp_path / "blocked-link.zip")


def test_documentation_has_valid_local_links_and_current_cli_options():
    checker = script("check_docs")
    assert not checker.check(ROOT)["errors"]
    assert checker.check_command(ROOT, ["python", "scripts/train_belief.py", "--positions", "12"])
    assert checker.check_command(ROOT, ["python", "scripts/train_flat.py", "--training-stage", "mcts"])
    assert not checker.check_command(ROOT, ["python", "scripts/train_flat.py", "--no-flat-improvements"])
    assert checker.check_command(ROOT, ["python", "scripts/init_model.py", "--device", "cpu"])
    assert not checker.check_command(ROOT, ["python", "scripts/init_model.py", "--project", "runs/example"])


def test_diagrams_render_from_source_with_contained_labels(tmp_path):
    renderer = script("render_diagrams")
    manifest = renderer.render(tmp_path, False)
    assert len(manifest["figures"]) == 3
    assert not manifest["compatibility_copies"]
    for spec in manifest["figures"].values():
        assert (tmp_path / spec["png"]).read_bytes().startswith(b"\x89PNG")
        assert spec["title"] in (tmp_path / spec["svg"]).read_text()


def test_public_citation_metadata():
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    assert citation["license"] == "MIT" and citation["authors"][0]["name"] == "FryOne2137"


@pytest.mark.skipif(not (ROOT / ".github/workflows/ci.yml").is_file(),
                    reason="GitHub Actions workflow is not configured")
def test_ci_when_configured_has_read_only_permissions_and_pinned_actions():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    assert workflow["permissions"] == {"contents": "read"}
    assert {"push", "pull_request", "workflow_dispatch"} <= workflow["on"].keys()
    actions = [step["uses"] for step in workflow["jobs"]["test"]["steps"] if "uses" in step]
    assert actions and all(len(action.rsplit("@", 1)[1]) == 40 for action in actions)


def test_portable_model_loader_does_not_allow_extra_pickled_python_calls(tmp_path):
    from polybot.config import ModelConfig
    from polybot.infrastructure.model_io import load_model, save_model
    from polybot.models.policy_value_belief import PolicyValueBeliefNet
    class UnsupportedMetadata:
        def __reduce__(self):
            return (eval, ("1 + 2",))
    model = PolicyValueBeliefNet(ModelConfig(d_model=16, transformer_layers=1, attention_heads=2))
    path = tmp_path / "model.pt"
    save_model(path, model, model.config)
    payload = torch.load(path, weights_only=True)
    payload["unsupported_metadata"] = UnsupportedMetadata()
    torch.save(payload, path)
    with pytest.raises(pickle.UnpicklingError):
        load_model(path, torch.device("cpu"))
