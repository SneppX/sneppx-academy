# Module 04: Security & Compliance

This module covers AI model security: Software Bill of Materials (SBOM), detached Ed25519 digital signatures, and automated compliance auditing using `sneppx-shield` and `sneppx-audits`.

## Key Concepts
- **Integrity**: Detached signatures (Ed25519) allow verifying models without needing the model bytes themselves.
- **Transparency**: SBOM records file-level digests to ensure the model composition hasn't changed.
- **Compliance**: Mapping audit findings to regulations like EU AI Act or ISO 42001.

## Security Tooling
Use `sneppx-shield` for signing and `sneppx-audits` for scanning:
```sh
sneppx-shield sign model.pt --secret-key @keypair.json
sneppx-audits . --format json
```
See the `examples/module04_security.py` file for a simulation using the CLI.

After the lesson, complete the [Module 04 Quiz](module04_quiz.md).
