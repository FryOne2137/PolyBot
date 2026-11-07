# Preparing a public source release

Changing repository visibility exposes **history**, not only the current files.
`.gitignore` prevents future accidental additions; it does not remove committed
models, deleted logs, old credentials or author email addresses.

## Review the current tree

```bash
python scripts/check_environment.py --smoke
python -m pytest
python scripts/check_docs.py
python scripts/audit_public.py --history --output runs/publication-audit.json
```

The audit reports locations, categories and counts without printing matched
credential values. Warnings about historical artifacts or non-noreply author
metadata require a deliberate review. The scan is pattern-based; binary/large
blob contents are not exhaustively inspected. Supplement it with an independent
scanner and review any files intended for model/data releases separately.

Check that:

- Documentation commands match each entry point and distinguish random
  initialization from actual training.
- All new source, tests, documentation and figure files are committed before
  pushing. A GitHub visibility change does not publish uncommitted local work.
- Local `models/`, `runs/`, `tmp/`, IDE state, environments, databases, replay
  and continuation files are excluded.
- Checkpoint licenses/provenance, any assets and third-party attribution are
  reviewed before a separate binary release.
- The repository description, topics, issue settings and private security
  reporting are configured deliberately on GitHub.
- CI has actually completed after pushing. A workflow file prepared locally
  is not evidence of a successful remote build.

The MIT license and public author handle are preserved. `CITATION.cff` describes
the software; it does not invent a DOI, release version, real name or affiliation.

## Choose a history policy

**Preserve history:** review all reachable historical content and author/committer
metadata before making the existing repository public. Intentionally retained
old model binaries are still downloadable. Any history rewrite needs a backup
and coordination with collaborators; it changes commit IDs and is not automated
by this preparation.

**Publish a clean source snapshot:** export the current reviewed source and
initialize a separate public history from it. This avoids publishing the private
history while leaving the development checkout intact.

```bash
python scripts/export_public.py --output dist/polybot-public.zip
```

The exporter includes current tracked and unignored source changes through an
allowlist. It excludes `.git`, local artifacts and symlinks; it aborts on current-tree
audit findings and refuses to overwrite an existing archive. `PUBLIC_EXPORT.json`
records file hashes and source revision, not author email or machine paths.
Inspect the archive before uploading it.

Use an empty directory for extraction, then initialize a new repository with
your chosen public author identity. This operation is intentionally separate
from exporting. The exporter does not create a GitHub repository, push code,
rewrite history or change visibility.
If the new publication URL differs, update the clone URL in `README.md` and
the repository fields in `CITATION.cff`. Keep the original private repository
private when using the source-only publication route.

For a strict check of an intentionally clean Git history:

```bash
python scripts/audit_public.py --history --require-clean-history
```

This makes historical-artifact/author-metadata warnings blocking. It is not a
substitute for credential rotation when a real credential has already leaked.
