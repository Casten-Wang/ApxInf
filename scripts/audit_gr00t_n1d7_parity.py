#!/usr/bin/env python3
"""Fail-closed aggregate audit for GR00T N1.7 action-parity reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="+", type=Path)
    parser.add_argument("--expected-fixtures", type=int, required=True)
    parser.add_argument("--min-cosine", type=float, default=0.997)
    parser.add_argument("--max-relative-l2", type=float, default=0.10)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_number(report: dict[str, Any], key: str, path: Path) -> float:
    value = report.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{path}: {key} is not finite: {value!r}")
    return float(value)


def main() -> None:
    args = parse_args()
    if len(args.report) != args.expected_fixtures:
        raise ValueError(
            f"received {len(args.report)} reports, expected {args.expected_fixtures}"
        )
    if len(set(path.resolve() for path in args.report)) != len(args.report):
        raise ValueError("duplicate parity report path")

    rows = []
    for path in args.report:
        document = json.loads(path.read_text())
        if document.get("schema") != "apxinf.gr00t-n1.7.parity-report.v2":
            raise ValueError(f"{path}: unexpected schema {document.get('schema')!r}")
        if document.get("finite") is not True:
            raise ValueError(f"{path}: output is not finite")
        shape = tuple(document.get("shape", []))
        if shape != (1, 40, 132):
            raise ValueError(f"{path}: unexpected action shape {shape}")
        cosine = require_number(document, "cosine", path)
        relative_l2 = require_number(document, "relative_l2", path)
        max_abs = require_number(document, "max_abs", path)
        mean_abs = require_number(document, "mean_abs", path)
        passed = cosine >= args.min_cosine and relative_l2 <= args.max_relative_l2
        rows.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "cosine": cosine,
                "relative_l2": relative_l2,
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "passed": passed,
            }
        )

    result = {
        "schema": "apxinf.gr00t-n1.7.parity-audit.v1",
        "passed": all(row["passed"] for row in rows),
        "thresholds": {
            "min_cosine": args.min_cosine,
            "max_relative_l2": args.max_relative_l2,
            "note": "max_abs and mean_abs are diagnostics, not acceptance gates",
        },
        "fixture_count": len(rows),
        "minimum_cosine": min(row["cosine"] for row in rows),
        "maximum_relative_l2": max(row["relative_l2"] for row in rows),
        "maximum_absolute_error": max(row["max_abs"] for row in rows),
        "maximum_mean_absolute_error": max(row["mean_abs"] for row in rows),
        "reports": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
