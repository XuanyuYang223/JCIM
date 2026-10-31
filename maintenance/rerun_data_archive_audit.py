"""Run the preserved independent audit with a fresh report destination.

Only the OUTPUT assignment changes in memory. The preserved implementation,
the primary scientific record, and the original audit report remain unchanged.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(f"A fresh audit report path is required: {destination}")

    implementation = Path(__file__).resolve().with_name("audit_data_archive.py")
    original_report = implementation.parent / "DATA_ARCHIVE_AUDIT.json"
    if destination == original_report:
        raise ValueError("The original audit report cannot be overwritten.")

    tree = ast.parse(implementation.read_text(encoding="utf-8-sig"), filename=str(implementation))
    output_assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "OUTPUT"
                                  for target in node.targets)]
    if len(output_assignments) != 1:
        raise ValueError("The preserved audit must have exactly one OUTPUT assignment.")
    output_assignments[0].value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()),
                                          args=[ast.Constant(value=str(destination))], keywords=[])
    ast.fix_missing_locations(tree)
    namespace = {"__file__": str(implementation), "__name__": "__main__"}
    exec(compile(tree, filename=str(implementation), mode="exec"), namespace)


if __name__ == "__main__":
    main()
