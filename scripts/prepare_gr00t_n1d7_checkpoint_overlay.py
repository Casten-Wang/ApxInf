#!/usr/bin/env python3
"""Create an auditable local GR00T checkpoint overlay for TRT export.

Model artifacts remain symlinked to the read-only source checkpoint.  Only
``config.json`` (the local backbone path) and ``processor_config.json`` (the
selected physical-camera prefix) are materialized in the overlay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--backbone", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--embodiment", default="libero_sim")
    parser.add_argument("--views", type=int, default=1)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise RuntimeError(f"refusing to overwrite a different file: {path}")
        return
    path.write_text(encoded)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    # Preserve a user-supplied Hugging Face-style alias in ``model_name``.
    # NVIDIA's loader dispatches by the literal ``nvidia/Cosmos-Reason2``
    # substring, so resolving that symlink to a generic model directory makes
    # an otherwise valid local checkpoint fail class selection.
    backbone = args.backbone.absolute()
    output_dir = args.output_dir.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    if not backbone.is_dir():
        raise FileNotFoundError(backbone)
    if args.views < 1:
        raise ValueError("--views must be positive")

    source_config_path = checkpoint / "config.json"
    source_processor_path = checkpoint / "processor_config.json"
    if not source_config_path.is_file() or not source_processor_path.is_file():
        raise FileNotFoundError("checkpoint must contain config.json and processor_config.json")

    config = json.loads(source_config_path.read_text())
    old_model_name = config.get("model_name")
    config["model_name"] = str(backbone)

    processor = json.loads(source_processor_path.read_text())
    try:
        processor_kwargs = processor["processor_kwargs"]
        video = processor_kwargs["modality_configs"][args.embodiment]["video"]
        source_video_keys = list(video["modality_keys"])
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"processor configuration has no video keys for {args.embodiment!r}"
        ) from error
    if args.views > len(source_video_keys):
        raise ValueError(
            f"--views={args.views} exceeds {len(source_video_keys)} configured cameras"
        )
    selected_video_keys = source_video_keys[: args.views]
    video["modality_keys"] = selected_video_keys
    old_processor_model_name = processor_kwargs.get("model_name")
    processor_kwargs["model_name"] = str(backbone)

    output_dir.mkdir(parents=True, exist_ok=True)
    materialized = {"config.json", "processor_config.json", "overlay-manifest.json"}
    for source in sorted(checkpoint.iterdir()):
        if source.name in materialized:
            continue
        target = output_dir / source.name
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError(f"overlay symlink points elsewhere: {target}")
        elif target.exists():
            raise RuntimeError(f"refusing to replace overlay entry: {target}")
        else:
            target.symlink_to(source.resolve(), target_is_directory=source.is_dir())

    overlay_config_path = output_dir / "config.json"
    overlay_processor_path = output_dir / "processor_config.json"
    write_json(overlay_config_path, config)
    write_json(overlay_processor_path, processor)

    manifest = {
        "schema": "apxinf.gr00t-n1.7.checkpoint-overlay.v1",
        "source_checkpoint": str(checkpoint),
        "backbone": str(backbone),
        "backbone_resolved": str(backbone.resolve()),
        "output_dir": str(output_dir),
        "embodiment": args.embodiment,
        "views": args.views,
        "changes": {
            "config.model_name": {
                "source": old_model_name,
                "overlay": str(backbone),
            },
            "processor.model_name": {
                "source": old_processor_model_name,
                "overlay": str(backbone),
            },
            "processor.video.modality_keys": {
                "source": source_video_keys,
                "overlay": selected_video_keys,
            },
        },
        "sha256": {
            "source_config": sha256(source_config_path),
            "overlay_config": sha256(overlay_config_path),
            "source_processor_config": sha256(source_processor_path),
            "overlay_processor_config": sha256(overlay_processor_path),
        },
    }
    write_json(output_dir / "overlay-manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
