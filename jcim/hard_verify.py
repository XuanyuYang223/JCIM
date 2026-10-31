"""Check harder-pilot archive integrity and the declared molecular split."""

import argparse
import json
from pathlib import Path

from .hard_data import audit_hard_splits
from .run import sha256


def verify_hard(record: Path) -> dict:
    record = Path(record).resolve()
    manifest = json.loads((record / "MANIFEST.json").read_text(encoding="utf-8"))
    failures = []
    for relative, expected in manifest["sha256"].items():
        path = (record / relative).resolve()
        if not path.is_relative_to(record) or not path.is_file() or sha256(path) != expected:
            failures.append(relative)
    if failures:
        raise ValueError(f"Scientific-record checksum mismatch: {failures}")
    records = [json.loads(line) for line in (record / "molecules.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = json.loads((record / "splits.json").read_text(encoding="utf-8"))
    audit = audit_hard_splits(records, splits)
    return {"files_verified": len(manifest["sha256"]), "molecules_verified": len(records),
            "split_identity_leakage": audit["split_identity_leakage"],
            "adaptation_support_unseen_by_both_experiences": audit["adaptation_support_unseen_by_both_experiences"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_hard(args.record), indent=2))
