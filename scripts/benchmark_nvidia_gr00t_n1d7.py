#!/usr/bin/env python3
"""Benchmark pinned NVIDIA GR00T N1.7 on an exact processor fixture.

This harness mirrors the component timing boundary in NVIDIA's
``scripts/deployment/benchmark_inference.py`` while bypassing dataset video
decoding.  It consumes an already-exported official processor NPZ so ApxInf
and NVIDIA see exactly the same non-zero model inputs.

The NVIDIA ``torch.compile`` mode compiles only
``action_head.model.forward``, matching the pinned official benchmark.  It is
not described as a CUDA Graph implementation. TensorRT mode requires every
engine in NVIDIA's ``n17_full_pipeline`` bundle and fails instead of silently
falling back to PyTorch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import types
from typing import Any, Callable

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

from gr00t_n1d7_reference_dump import (
    EXPECTED_SOURCE_REVISION,
    load_model,
    reference_inputs,
    sha256,
    source_revision,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--backbone", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--input-npz", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--processor-median-ms",
        type=float,
        help=(
            "optional separately measured processor median; retained for "
            "backward-compatible independent-median sums"
        ),
    )
    parser.add_argument(
        "--processor-report",
        type=Path,
        help=(
            "optional processor-benchmark JSON; when supplied, pair its samples "
            "with Model Core samples before computing E2E statistics"
        ),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("eager", "compile", "tensorrt"),
        default=("eager", "compile"),
    )
    parser.add_argument(
        "--trt-engine-dir",
        type=Path,
        help=(
            "official full-pipeline TensorRT engine directory; required when "
            "--modes includes tensorrt"
        ),
    )
    parser.add_argument(
        "--reference-npz",
        type=Path,
        help=(
            "optional deterministic NVIDIA reference dump; TensorRT's final "
            "action is compared against its final_action tensor"
        ),
    )
    return parser.parse_args()


def summarize(samples: list[float]) -> dict[str, Any]:
    if not samples:
        raise ValueError("latency sample list must not be empty")
    ordered = sorted(samples)
    p90_index = min(len(ordered) - 1, int(np.ceil(0.9 * len(ordered))) - 1)
    p95_index = min(len(ordered) - 1, int(np.ceil(0.95 * len(ordered))) - 1)
    return {
        "samples": samples,
        "minimum": min(samples),
        "median": statistics.median(samples),
        "p90": ordered[p90_index],
        "p95": ordered[p95_index],
        "mean": statistics.fmean(samples),
        "maximum": max(samples),
    }


def merged_inputs(
    backbone_input: BatchFeature, action_input: BatchFeature
) -> BatchFeature:
    return BatchFeature(data={**dict(backbone_input), **dict(action_input)})


def configure_tensorrt(model: Any, args: argparse.Namespace) -> tuple[Any, Callable[[], None]]:
    if args.trt_engine_dir is None:
        raise ValueError("--trt-engine-dir is required for TensorRT mode")
    if not args.trt_engine_dir.is_dir():
        raise FileNotFoundError(args.trt_engine_dir)

    required = (
        "llm_bf16.engine",
        "vl_self_attention.engine",
        "state_encoder.engine",
        "action_encoder.engine",
        "dit_bf16.engine",
        "action_decoder.engine",
        "export_metadata.json",
    )
    missing = [
        name for name in required if not (args.trt_engine_dir / name).is_file()
    ]
    if not any(
        (args.trt_engine_dir / name).is_file()
        for name in ("vit.engine", "vit_bf16.engine")
    ):
        missing.append("vit.engine (or legacy vit_bf16.engine)")
    if missing:
        raise FileNotFoundError(
            "TensorRT mode requires a complete n17_full_pipeline bundle; "
            f"missing {missing} in {args.trt_engine_dir}"
        )

    deployment_dir = args.source_dir.resolve() / "scripts" / "deployment"
    if not deployment_dir.is_dir():
        raise FileNotFoundError(deployment_dir)
    sys.path.insert(0, str(deployment_dir))
    from gr00t.deployment.modes import InferenceMode
    from trt_model_forward import close_tensorrt_engines, setup_tensorrt_engines

    # The official helper only needs an object exposing ``.model``.  Keeping the
    # already-loaded model avoids a second checkpoint load and guarantees that
    # eager/compile/TRT consume the exact same BatchFeature tensors.
    policy = types.SimpleNamespace(model=model)
    setup_tensorrt_engines(
        policy,
        str(args.trt_engine_dir.resolve()),
        mode=InferenceMode.n17_full_pipeline,
    )
    loaded = {
        "vit": getattr(model.backbone, "vit_engine", None),
        "llm": getattr(model.backbone, "llm_engine", None),
        "vl_self_attention": getattr(model.action_head, "vl_sa_engine", None),
        "state_encoder": getattr(model.action_head, "state_encoder_engine", None),
        "action_encoder": getattr(model.action_head, "action_encoder_engine", None),
        "dit": getattr(model.action_head, "dit_engine", None),
        "action_decoder": getattr(model.action_head, "action_decoder_engine", None),
    }
    missing_loaded = [name for name, engine in loaded.items() if engine is None]
    if missing_loaded:
        close_tensorrt_engines(policy)
        raise RuntimeError(
            "official TensorRT setup did not activate every full-pipeline "
            f"component: {missing_loaded}"
        )
    return policy, lambda: close_tensorrt_engines(policy)


@torch.inference_mode()
def run_components(
    model: Any,
    inputs: BatchFeature,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        backbone_input, action_input = model.prepare_input(inputs)
        backbone_output = model.backbone(backbone_input)
        model.action_head.get_action(backbone_output, action_input)
    torch.cuda.synchronize()

    # This is the strict A/B boundary used by ApxInf's fixture benchmark: one
    # enclosing host timer around the whole preprocessed-tensor-to-action path,
    # with a single synchronization at each end. Keep it separate from the
    # component-sum convention used by NVIDIA's published benchmark table.
    model_core_wall_clock_samples: list[float] = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        backbone_input, action_input = model.prepare_input(inputs)
        backbone_output = model.backbone(backbone_input)
        model.action_head.get_action(backbone_output, action_input)
        torch.cuda.synchronize()
        model_core_wall_clock_samples.append((time.perf_counter() - start) * 1_000.0)

    # Re-warm before the component loop so the extra synchronization boundary
    # is measured independently rather than immediately after a cold transition.
    for _ in range(warmup):
        backbone_input, action_input = model.prepare_input(inputs)
        backbone_output = model.backbone(backbone_input)
        model.action_head.get_action(backbone_output, action_input)
    torch.cuda.synchronize()

    backbone_samples: list[float] = []
    action_head_samples: list[float] = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        backbone_input, action_input = model.prepare_input(inputs)
        backbone_output = model.backbone(backbone_input)
        torch.cuda.synchronize()
        backbone_samples.append((time.perf_counter() - start) * 1_000.0)

        torch.cuda.synchronize()
        start = time.perf_counter()
        model.action_head.get_action(backbone_output, action_input)
        torch.cuda.synchronize()
        action_head_samples.append((time.perf_counter() - start) * 1_000.0)

    backbone = summarize(backbone_samples)
    action_head = summarize(action_head_samples)
    model_core_samples = [
        backbone_ms + action_head_ms
        for backbone_ms, action_head_ms in zip(
            backbone_samples, action_head_samples, strict=True
        )
    ]
    return {
        "model_core_wall_clock_latency_ms": summarize(model_core_wall_clock_samples),
        "backbone_latency_ms": backbone,
        "action_head_latency_ms": action_head,
        "model_core_component_sum_latency_ms": summarize(model_core_samples),
        "model_core_component_median_sum_ms": (
            backbone["median"] + action_head["median"]
        ),
    }


@torch.inference_mode()
def run_action_once(model: Any, inputs: BatchFeature) -> np.ndarray:
    backbone_input, action_input = model.prepare_input(inputs)
    backbone_output = model.backbone(backbone_input)
    output = model.action_head.get_action(backbone_output, action_input)
    torch.cuda.synchronize()
    action = output["action_pred"]
    return action.detach().float().cpu().contiguous().numpy()


def compare_action(actual: np.ndarray, reference_npz: Path) -> dict[str, Any]:
    if not reference_npz.is_file():
        raise FileNotFoundError(reference_npz)
    with np.load(reference_npz, allow_pickle=False) as archive:
        if "final_action" not in archive.files:
            raise RuntimeError(f"reference dump has no final_action: {reference_npz}")
        expected = np.asarray(archive["final_action"], dtype=np.float32)
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"TensorRT action shape {actual.shape} differs from reference {expected.shape}"
        )
    actual_f32 = actual.astype(np.float32, copy=False)
    difference = np.abs(actual_f32 - expected)
    actual_flat = actual_f32.astype(np.float64, copy=False).reshape(-1)
    expected_flat = expected.astype(np.float64, copy=False).reshape(-1)
    expected_l2 = float(np.linalg.norm(expected_flat))
    actual_l2 = float(np.linalg.norm(actual_flat))
    error_l2 = float(np.linalg.norm(actual_flat - expected_flat))
    if expected_l2 == 0.0:
        relative_l2 = 0.0 if error_l2 == 0.0 else float("inf")
    else:
        relative_l2 = error_l2 / expected_l2
    if expected_l2 == 0.0 or actual_l2 == 0.0:
        cosine = 1.0 if expected_l2 == actual_l2 == 0.0 else 0.0
    else:
        cosine = float(np.dot(expected_flat, actual_flat) / (expected_l2 * actual_l2))
    return {
        "reference_npz": str(reference_npz.resolve()),
        "reference_npz_sha256": sha256(reference_npz),
        "shape": list(actual.shape),
        "finite": bool(np.isfinite(actual).all()),
        "maximum_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
        "relative_l2": relative_l2,
        "cosine": cosine,
        "actual_sum": float(actual.sum()),
        "reference_sum": float(expected.sum()),
    }


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if len(set(args.modes)) != len(args.modes):
        raise ValueError("--modes must not contain duplicates")
    expected_order = {"eager": 0, "compile": 1, "tensorrt": 2}
    if list(args.modes) != sorted(args.modes, key=expected_order.__getitem__):
        raise ValueError(
            "--modes must follow eager, compile, tensorrt order so a patched "
            "model is never mislabeled as an earlier execution mode"
        )
    if "tensorrt" in args.modes and args.trt_engine_dir is None:
        raise ValueError("--trt-engine-dir is required for TensorRT mode")
    if "tensorrt" in args.modes and len(args.modes) != 1:
        raise ValueError(
            "run TensorRT in a separate invocation so compiled-module state, "
            "allocator state and thermal history cannot contaminate its row"
        )
    if args.reference_npz is not None and "tensorrt" not in args.modes:
        raise ValueError("--reference-npz currently applies only to TensorRT mode")
    if source_revision(args.source_dir) != EXPECTED_SOURCE_REVISION:
        raise RuntimeError("pinned Isaac-GR00T source revision changed")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    processor_samples: list[float] | None = None
    processor_report: str | None = None
    if args.processor_report is not None:
        document = json.loads(args.processor_report.read_text())
        try:
            processor_samples = [
                float(value) for value in document["latency_ms"]["samples"]
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid processor report {args.processor_report}: {error}"
            ) from error
        if len(processor_samples) != args.iterations:
            raise ValueError(
                f"processor report has {len(processor_samples)} samples, "
                f"expected {args.iterations}"
            )
        processor_summary = summarize(processor_samples)
        if (
            args.processor_median_ms is not None
            and not np.isclose(
                args.processor_median_ms,
                processor_summary["median"],
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise ValueError(
                "--processor-median-ms disagrees with --processor-report: "
                f"{args.processor_median_ms} versus {processor_summary['median']}"
            )
        args.processor_median_ms = processor_summary["median"]
        processor_report = str(args.processor_report.resolve())

    model, config = load_model(args)
    backbone_input, action_input, fixed_noise = reference_inputs(model, config, args)
    inputs = merged_inputs(backbone_input, action_input)
    # The official TRT action-head path uses ``init_actions`` when available.
    # Pin it to the fixture's deterministic noise so numerical comparison and
    # performance benchmarking share the same input contract.
    model.action_head.init_actions = fixed_noise

    results: dict[str, Any] = {}
    runners: dict[str, Callable[[], dict[str, Any]]] = {
        "eager": lambda: run_components(model, inputs, args.warmup, args.iterations),
        "compile": lambda: run_components(
            model, inputs, args.warmup + 2, args.iterations
        ),
    }
    original_cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        close_tensorrt: Callable[[], None] | None = None
        for mode in args.modes:
            if mode == "compile":
                model.action_head.model.forward = torch.compile(
                    model.action_head.model.forward, mode="max-autotune"
                )
                torch.backends.cudnn.benchmark = True
            elif mode == "tensorrt":
                _policy, close_tensorrt = configure_tensorrt(model, args)
                runners[mode] = lambda: run_components(
                    model, inputs, args.warmup, args.iterations
                )
            try:
                results[mode] = runners[mode]()
                if args.processor_median_ms is not None:
                    results[mode]["e2e_component_median_sum_ms"] = (
                        args.processor_median_ms
                        + results[mode]["model_core_component_median_sum_ms"]
                    )
                if processor_samples is not None:
                    wall_clock_samples = results[mode][
                        "model_core_wall_clock_latency_ms"
                    ]["samples"]
                    results[mode][
                        "e2e_processor_plus_model_core_wall_clock_latency_ms"
                    ] = summarize(
                        [
                            processor_ms + model_core_ms
                            for processor_ms, model_core_ms in zip(
                                processor_samples, wall_clock_samples, strict=True
                            )
                        ]
                    )
                    model_core_samples = results[mode][
                        "model_core_component_sum_latency_ms"
                    ]["samples"]
                    results[mode]["e2e_component_sample_sum_latency_ms"] = summarize(
                        [
                            processor_ms + model_core_ms
                            for processor_ms, model_core_ms in zip(
                                processor_samples, model_core_samples, strict=True
                            )
                        ]
                    )
                if mode == "tensorrt":
                    action = run_action_once(model, inputs)
                    results[mode]["action"] = {
                        "shape": list(action.shape),
                        "finite": bool(np.isfinite(action).all()),
                        "sum": float(action.sum()),
                    }
                    if args.reference_npz is not None:
                        results[mode]["reference_comparison"] = compare_action(
                            action, args.reference_npz
                        )
            finally:
                if mode == "tensorrt" and close_tensorrt is not None:
                    close_tensorrt()
    finally:
        torch.backends.cudnn.benchmark = original_cudnn_benchmark

    report = {
        "schema": "apxinf.gr00t-n1.7.nvidia-benchmark.v1",
        "source_revision": source_revision(args.source_dir),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_config_sha256": sha256(args.checkpoint / "config.json"),
        "backbone": str(args.backbone.resolve()),
        "backbone_config_sha256": sha256(args.backbone / "config.json"),
        "input_npz": str(args.input_npz.resolve()),
        "input_npz_sha256": sha256(args.input_npz),
        "device": args.device,
        "device_name": torch.cuda.get_device_name(torch.device(args.device)),
        "dtype": "bfloat16",
        "batch": 1,
        "action_shape": [1, int(config.action_horizon), int(config.max_action_dim)],
        "flow_steps": int(config.num_inference_timesteps),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "processor_report": processor_report,
        "processor_median_ms": args.processor_median_ms,
        "tensorrt_engine_dir": (
            str(args.trt_engine_dir.resolve())
            if args.trt_engine_dir is not None
            else None
        ),
        "reference_scope": (
            "gr00t_n1d7_reference_dump.py uses NVIDIA's pinned model submodules "
            "and explicitly reproduces the four-step Euler flow loop with fixed "
            "fixture noise; TensorRT mode additionally records its own get_action "
            "output comparison when --reference-npz is supplied"
        ),
        "timing_boundary": {
            "backbone": "model.prepare_input + model.backbone + synchronize",
            "action_head": "action_head.get_action + synchronize",
            "model_core": (
                "model_core_wall_clock_latency_ms is one enclosing synchronized timer; "
                "paired backbone/action-head sample sums reproduce NVIDIA's published "
                "component convention and the independent-median sum is retained separately"
            ),
            "e2e": (
                "processor samples are paired separately with contiguous Model Core and "
                "component-sum samples when --processor-report is supplied; neither is "
                "a single raw-input in-process timer"
            ),
        },
        "compile_scope": "action_head.model.forward only",
        "tensorrt_scope": (
            "official n17_full_pipeline: ViT + LLM + action-head engines, with "
            "the lightweight PyTorch glue retained by NVIDIA's implementation"
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
