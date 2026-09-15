"""Module 04 - Security & compliance companion example.

Runs against the `sneppx-shield` signing / SBOM / audit API.

Usage:
    1) From a `sneppx-shield` checkout:
         $env:PYTHONPATH="C:\\path\\to\\sneppx-shield\\src"
         python <this-file>
    2) Or point at a checkout explicitly:
         $env:SNEPPX_SHIELD_PATH="C:\\path\\to\\sneppx-shield\\src"
         python <this-file>
"""

import os
import pathlib
import sys
import tempfile

_SEARCH = os.environ.get("SNEPPX_SHIELD_PATH")
if _SEARCH and _SEARCH not in sys.path:
    sys.path.insert(0, _SEARCH)

from sneppx_shield import audit, sbom, signature


def demo_security():
    tmp = pathlib.Path(tempfile.mkdtemp())
    model = tmp / "model.pt"
    model.write_bytes(b"dummy-model-bytes" * 10)

    # 1. sign with a fresh Ed25519 key (detached .sig next to the model)
    pk, sk = signature.new_keypair(save_to=str(tmp))
    sig_path, payload = signature.sign_file(model, secret_key=sk, signer="demo")

    # 2. verify (uses the embedded public key from the .sig payload)
    ok, detail = signature.verify_signature(model, sig_path=sig_path)
    print("signature verified:", ok, "| signer:", detail.get("signer"))
    assert ok

    # 3. SBOM over the artifact directory
    bom = sbom.collect_sbom(tmp)
    print("sbom files:", bom["file_count"], "total_bytes:", bom["total_bytes"])
    assert bom["file_count"] >= 2  # model.pt + its keypair

    # 4. full audit: compliance rating from evidence
    facts = audit.collect(model, evidence={
        "risk_assessed": True, "human_oversight": True,
        "model_card": True, "monitoring": True,
        "access_control": True, "robustness_tested": True,
    })
    print("compliance rating:", facts["compliance_rating"],
          f"({facts['compliance_passed']}/{facts['compliance_total']})")
    assert facts["compliance_rating"] == "pass"
    print("security example OK")


if __name__ == "__main__":
    demo_security()