# Security and sensitive data

This is research software under active development. No long-term support or
formal security assurance is provided.

Portable model checkpoints use PyTorch's explicit `weights_only=True` loader.
This narrows deserialization to the documented tensor/configuration format;
it is not a guarantee that arbitrary third-party files are safe or small.
Pickle-based replay and optimization continuation still require trusted local
artifacts. Do not load those files from untrusted sources.

Do not post credentials, private datasets, full environment dumps, personal paths
or unreviewed source snapshots in issues. If private vulnerability reporting is
enabled for the repository, use GitHub's private report option. Otherwise use an
existing private maintainer contact channel before disclosing sensitive details.

For suspected credential exposure, rotate the credential first. Deleting it in
a later commit does not remove earlier copies from Git history or forks.

`scripts/audit_public.py` is a read-only, redacted pattern scan. It checks
publication candidates and can inspect reachable history, but does not establish
the absence of secrets. Review its limitations and consider an independent
secret-scanning tool before changing repository visibility.

See the [publication checklist](../docs/public_release.md) for source exports,
historical artifacts and author metadata.
