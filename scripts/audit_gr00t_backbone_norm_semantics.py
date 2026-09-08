#!/usr/bin/env python3
"""Audit which Qwen tensor Isaac-GR00T exposes as ``backbone_features``.

This is an independent, fail-closed audit.  It hooks the last decoder layer
and the terminal Qwen RMSNorm in the pinned NVIDIA implementation, executes
one real preprocessed fixture, and compares both tensors with the value that
``Qwen3Backbone.forward`` actually returns to the GR00T action head.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import transformers


EXPECTED_TRANSFORMERS = "4.57.3"
EXPECTED_SOURCE_REVISION = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--backbone", required=True, type=Path)
    parser.add_argument("--input-npz", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_reference_helper(repo_root: Path) -> Any:
    path = repo_root / "scripts" / "gr00t_n1d7_reference_dump.py"
    spec = importlib.util.spec_from_file_location("gr00t_reference_dump", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import reference helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_from_hook(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"hook returned unsupported output type {type(output)!r}")


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    lhs = reference.detach().float().cpu().contiguous()
    rhs = candidate.detach().float().cpu().contiguous()
    if lhs.shape != rhs.shape:
        raise RuntimeError(f"shape mismatch: {tuple(lhs.shape)} vs {tuple(rhs.shape)}")
    delta = rhs - lhs
    lhs64 = lhs.double().reshape(-1)
    rhs64 = rhs.double().reshape(-1)
    denominator = torch.linalg.vector_norm(lhs64).clamp_min(torch.finfo(torch.float64).tiny)
    cosine = torch.nn.functional.cosine_similarity(lhs64, rhs64, dim=0)
    return {
        "shape": list(lhs.shape),
        "bitwise_equal": bool(torch.equal(lhs, rhs)),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "relative_l2": float(torch.linalg.vector_norm(delta.double().reshape(-1)) / denominator),
        "cosine": float(cosine),
        "reference_l2": float(torch.linalg.vector_norm(lhs64)),
        "candidate_l2": float(torch.linalg.vector_norm(rhs64)),
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if transformers.__version__ != EXPECTED_TRANSFORMERS:
        raise RuntimeError(
            f"transformers={transformers.__version__}, expected {EXPECTED_TRANSFORMERS}"
        )
    revision = git_revision(args.source_dir)
    if revision != EXPECTED_SOURCE_REVISION:
        raise RuntimeError(f"Isaac-GR00T revision={revision}, expected={EXPECTED_SOURCE_REVISION}")

    helper = load_reference_helper(Path(__file__).resolve().parents[1])
    helper_args = SimpleNamespace(
        checkpoint=args.checkpoint,
        backbone=args.backbone,
        input_npz=args.input_npz,
        source_dir=args.source_dir,
        device=args.device,
        embodiment_id=2,
        seed=0,
    )
    model, config = helper.load_model(helper_args)
    backbone_input, _action_input, _noise = helper.reference_inputs(model, config, helper_args)

    language_model = model.backbone.model.language_model
    if len(language_model.layers) != int(config.select_layer):
        raise RuntimeError(
            f"loaded {len(language_model.layers)} decoder layers, expected select_layer={config.select_layer}"
        )

    captured: dict[str, torch.Tensor] = {}

    def save(name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            captured[name] = tensor_from_hook(output).detach().clone()

        return hook

    layer_handle = language_model.layers[-1].register_forward_hook(save("last_decoder_layer"))
    norm_handle = language_model.norm.register_forward_hook(save("terminal_rmsnorm"))
    try:
        returned = model.backbone(backbone_input).backbone_features.detach().clone()
    finally:
        layer_handle.remove()
        norm_handle.remove()

    missing = sorted({"last_decoder_layer", "terminal_rmsnorm"}.difference(captured))
    if missing:
        raise RuntimeError(f"required hooks did not execute: {missing}")

    pre_norm = metrics(returned, captured["last_decoder_layer"])
    post_norm = metrics(returned, captured["terminal_rmsnorm"])
    verdict = (
        pre_norm["bitwise_equal"]
        and pre_norm["max_abs"] == 0.0
        and not post_norm["bitwise_equal"]
        and post_norm["cosine"] < 0.99
    )
    report = {
        "schema": "apxinf.gr00t-n1.7.backbone-norm-audit.v1",
        "verdict": "pre-terminal-rmsnorm" if verdict else "unexpected",
        "passed": verdict,
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": args.device,
            "device_name": torch.cuda.get_device_name(torch.device(args.device)),
            "isaac_gr00t_revision": revision,
        },
        "artifacts": {
            "checkpoint_config": str((args.checkpoint / "config.json").resolve()),
            "checkpoint_config_sha256": sha256(args.checkpoint / "config.json"),
            "backbone_config": str((args.backbone / "config.json").resolve()),
            "backbone_config_sha256": sha256(args.backbone / "config.json"),
            "input_npz": str(args.input_npz.resolve()),
            "input_npz_sha256": sha256(args.input_npz),
        },
        "model_contract": {
            "select_layer": int(config.select_layer),
            "loaded_decoder_layers": len(language_model.layers),
            "backbone_feature_shape": list(returned.shape),
        },
        "returned_vs_last_decoder_layer": pre_norm,
        "returned_vs_terminal_rmsnorm": post_norm,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not verdict:
        raise SystemExit("GR00T backbone RMSNorm semantics audit failed")


if __name__ == "__main__":
    main()
