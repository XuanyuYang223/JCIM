"""Verify archived scientific-record checksums without modifying the record."""

import argparse
import json
from pathlib import Path

from .data import audit_splits
from .run import sha256


def verify(result_dir: Path) -> dict:
    result_dir = Path(result_dir).resolve()
    manifest = json.loads((result_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    failed = []
    for name, expected in manifest["sha256"].items():
        target = (result_dir / name).resolve()
        if not target.is_relative_to(result_dir) or not target.is_file() or sha256(target) != expected:
            failed.append(name)
    if failed:
        raise ValueError(f"Scientific-record checksum mismatch or missing files: {failed}")
    records = [json.loads(line) for line in (result_dir / "molecules.jsonl").read_text(encoding="utf-8").splitlines()]
    splits = json.loads((result_dir / "splits.json").read_text(encoding="utf-8"))
    audit = audit_splits(records, splits)
    return {"files_verified": len(manifest["sha256"]), "molecules_verified": len(records),
            "split_identity_leakage": audit["split_identity_leakage"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.result_dir), indent=2))
