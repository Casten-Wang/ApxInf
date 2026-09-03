#!/usr/bin/env python3
"""Collect and merge one BF16 activation calibration sample per LIBERO-10 task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from run_gr00t_n1d7_libero_campaign import LIBERO_10_ENVS


SCHEMA = "apxinf.gr00t-n1.7.fp8-calibration.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--backbone", required=True, type=Path)
    parser.add_argument("--tactics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Path(__file__).resolve().with_name("eval_gr00t_n1d7_libero.py")
    campaign = {
        "schema": "apxinf.gr00t-n1.7.libero-calibration-campaign.v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "tasks": list(LIBERO_10_ENVS),
        "views": 2,
        "seed": args.seed,
        "sample": "first model input after environment initialization",
    }
    encoded = json.dumps(campaign, sort_keys=True, separators=(",", ":")).encode()
    fixture_sha256 = hashlib.sha256(encoded).hexdigest()
    (args.output_dir / "campaign.json").write_text(
        json.dumps({**campaign, "sha256": fixture_sha256}, indent=2, sort_keys=True)
        + "\n"
    )

    documents = []
    for task_id, env_name in enumerate(LIBERO_10_ENVS):
        calibration = args.output_dir / f"task{task_id}-calibration.json"
        receipt = args.output_dir / f"task{task_id}-receipt.json"
        log = args.output_dir / f"task{task_id}.log"
        environment = os.environ.copy()
        environment["APXINF_GR00T_FP8_CALIBRATION_OUTPUT"] = str(calibration)
        environment["APXINF_GR00T_FP8_FIXTURE_SHA256"] = fixture_sha256
        command = [
            str(args.python.absolute()),
            str(evaluator),
            "--source-dir", str(args.source_dir.resolve()),
            "--checkpoint", str(args.checkpoint.resolve()),
            "--backbone", str(args.backbone.resolve()),
            "--precision", "bf16",
            "--tactics", str(args.tactics.resolve()),
            "--env-name", env_name,
            "--views", "2",
            "--episodes", "1",
            "--n-envs", "1",
            "--max-episode-steps", "1",
            "--n-action-steps", "1",
            "--seed", str(args.seed),
            "--noise-mode", "stream",
            "--record-first-normalized-action",
            "--output", str(receipt),
        ]
        print(f"calibrate task={task_id}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(
                command,
                check=True,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        document = json.loads(calibration.read_text())
        if document.get("schema") != SCHEMA:
            raise ValueError(f"{calibration}: unexpected schema")
        if document.get("fixture_manifest_sha256") != fixture_sha256:
            raise ValueError(f"{calibration}: campaign hash mismatch")
        documents.append(document)

    checkpoint_hashes = {d["checkpoint_config_sha256"] for d in documents}
    scale_keys = {frozenset(d["activation_scales"]) for d in documents}
    if len(checkpoint_hashes) != 1 or len(scale_keys) != 1:
        raise ValueError("per-task calibration schemas do not match")
    keys = sorted(documents[0]["activation_scales"])
    merged = {
        "schema": SCHEMA,
        "checkpoint_config_sha256": next(iter(checkpoint_hashes)),
        "fixture_manifest_sha256": fixture_sha256,
        "activation_scales": {
            key: max(float(d["activation_scales"][key]) for d in documents)
            for key in keys
        },
    }
    output = args.output_dir / "fp8-calibration.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "tasks": len(documents)}, indent=2))


if __name__ == "__main__":
    main()
