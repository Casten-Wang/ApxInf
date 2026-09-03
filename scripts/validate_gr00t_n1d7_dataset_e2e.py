#!/usr/bin/env python3
"""Validate GR00T N1.7 from an official LeRobot sample to ApxInf actions.

Unlike ``validate_gr00t_n1d7_e2e.py``, this harness does not hard-code the
LIBERO camera or state schema.  It asks the checkpoint processor for the
selected embodiment's modality configuration, uses NVIDIA's pinned dataset
loader to extract one step, and exports the exact collated model-boundary
tensors consumed by both ApxInf and the NVIDIA reference implementation.

Processor and Model Core latency remain separate measurements. Samples with
the same iteration index are added before summarization, matching NVIDIA's
component-E2E convention; an independent-median sum is retained as a clearly
labelled compatibility field. Neither is a contiguous wall-clock timer.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from transformers import AutoProcessor

from validate_gr00t_n1d7_e2e import (
    EXPECTED_SOURCE_REVISION,
    numpy_tensor,
    recursive_bf16,
    run_checked,
    sha256,
    source_revision,
    summarize,
)


REQUIRED_INPUTS = {
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "state",
    "embodiment_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--backbone", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--embodiment", required=True)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="stop after exporting the processor NPZ, report, and Rust fixture",
    )
    parser.add_argument("--fixture")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument(
        "--square-image-size",
        type=int,
        help=(
            "resize every decoded camera frame to an explicit square before "
            "the NVIDIA processor; use 256 to reconstruct FlashRT's published "
            "GR00T N1.7 two-view 4-grid/1024-patch benchmark geometry"
        ),
    )
    parser.add_argument(
        "--views",
        type=int,
        help=(
            "keep the first N physical camera keys from the checkpoint modality "
            "configuration; by default all configured views are retained"
        ),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--processor-warmup", type=int, default=10)
    parser.add_argument("--model-warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--noise-kind",
        choices=("fixed-zero", "seeded-normal"),
        default="fixed-zero",
        help="deterministic initial action-noise fixture",
    )
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument(
        "--zero-normalized-state",
        action="store_true",
        help=(
            "replace the official processor's normalized state tensor with zeros; "
            "this explicitly reproduces FlashRT's public latency harness input"
        ),
    )
    parser.add_argument(
        "--tactics",
        type=Path,
        help="optional ApxInf BF16 tactic database installed before graph capture",
    )
    parser.add_argument(
        "--run-nvidia-reference",
        action="store_true",
        help="run NVIDIA once with the same deterministic fixture noise and compare actions",
    )
    parser.add_argument(
        "--run-nvidia-benchmark",
        action="store_true",
        help="benchmark NVIDIA eager/compile Model Core on the same processor NPZ",
    )
    parser.add_argument(
        "--nvidia-modes",
        nargs="+",
        choices=("eager", "compile", "tensorrt"),
        default=("tensorrt",),
    )
    parser.add_argument(
        "--trt-engine-dir",
        type=Path,
        help=(
            "official full-pipeline TensorRT engine directory; required when "
            "--nvidia-modes includes tensorrt"
        ),
    )
    return parser.parse_args()


def json_shape(value: Any) -> list[int]:
    return [int(dimension) for dimension in np.asarray(value).shape]


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def install_pyav_video_fallback(loader_module: Any) -> None:
    """Install an exact frame-index decoder when TorchCodec cannot load.

    Thor's system FFmpeg 8 is newer than the native loaders shipped by
    TorchCodec 0.8.  PyAV carries its own decoder runtime. Decode sequentially
    from frame zero so the fallback does
    not depend on backend-specific keyframe seeking.  The report records the
    decoder and hashes every decoded image, making this path auditable against
    a TorchCodec-produced fixture.
    """
    import av

    def get_frames_by_indices(
        video_path: str,
        indices: list[int] | np.ndarray,
        decoder_kwargs: dict[str, Any] | None = None,
    ) -> np.ndarray:
        del decoder_kwargs
        requested = [int(index) for index in np.asarray(indices).reshape(-1)]
        if not requested:
            raise ValueError("at least one video frame index is required")
        if min(requested) < 0:
            raise ValueError(f"video frame indices must be non-negative: {requested}")

        wanted = set(requested)
        decoded: dict[int, np.ndarray] = {}
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            for frame_index, frame in enumerate(container.decode(stream)):
                if frame_index in wanted:
                    decoded[frame_index] = frame.to_ndarray(format="rgb24")
                if frame_index >= max(requested):
                    break
        missing = sorted(wanted.difference(decoded))
        if missing:
            raise RuntimeError(f"video {video_path} is missing frames {missing}")
        return np.stack([decoded[index] for index in requested])

    loader_module.get_frames_by_indices = get_frames_by_indices


def tensor_signature(inputs: dict[str, Any]) -> dict[str, Any]:
    grids = numpy_tensor(inputs["image_grid_thw"], "image_grid_thw")
    tokens = numpy_tensor(inputs["input_ids"], "input_ids")
    attention = numpy_tensor(inputs["attention_mask"], "attention_mask")
    pixels = numpy_tensor(inputs["pixel_values"], "pixel_values")
    state = numpy_tensor(inputs["state"], "state")
    embodiment = numpy_tensor(inputs["embodiment_id"], "embodiment_id")
    return {
        "input_ids_shape": json_shape(tokens),
        "token_count": int(tokens.shape[-1]),
        "active_token_count": int(attention.sum()),
        "pixel_values_shape": json_shape(pixels),
        "vision_patch_rows": int(pixels.shape[0]),
        "image_grid_thw_shape": json_shape(grids),
        "image_grid_thw": grids.astype(np.int64).tolist(),
        "processor_image_count": int(grids.shape[0]),
        "state_shape": json_shape(state),
        "embodiment_id": int(embodiment.reshape(-1)[0]),
    }


def assert_deterministic(first: dict[str, Any], second: dict[str, Any]) -> None:
    for name in sorted(REQUIRED_INPUTS):
        left = first[name]
        right = second[name]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise TypeError(f"processor output {name!r} is not a torch.Tensor")
        if not torch.equal(left, right):
            raise RuntimeError(
                f"processor output {name!r} changed across identical eval calls"
            )


def main() -> None:
    args = parse_args()
    if args.episode_index < 0 or args.step_index < 0:
        raise ValueError("episode and step indices must be non-negative")
    if args.processor_warmup < 0 or args.model_warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    revision = source_revision(args.source_dir)
    if revision != EXPECTED_SOURCE_REVISION:
        raise RuntimeError(
            f"Isaac-GR00T source revision is {revision}, expected "
            f"{EXPECTED_SOURCE_REVISION}"
        )
    for path in (args.checkpoint, args.backbone, args.source_dir, args.dataset):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if not args.prepare_only and args.engine is None:
        raise ValueError("--engine is required unless --prepare-only is selected")
    if args.engine is not None and not args.engine.is_file():
        raise FileNotFoundError(args.engine)
    if args.tactics is not None and not args.tactics.is_file():
        raise FileNotFoundError(args.tactics)
    if args.prepare_only and (
        args.run_nvidia_reference or args.run_nvidia_benchmark
    ):
        raise ValueError("NVIDIA validation modes cannot be used with --prepare-only")
    if len(set(args.nvidia_modes)) != len(args.nvidia_modes):
        raise ValueError("--nvidia-modes must not contain duplicates")
    if args.run_nvidia_benchmark and "tensorrt" in args.nvidia_modes:
        if args.trt_engine_dir is None:
            raise ValueError(
                "--trt-engine-dir is required when --nvidia-modes includes tensorrt"
            )
        if not args.trt_engine_dir.is_dir():
            raise FileNotFoundError(args.trt_engine_dir)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    sys.path.insert(0, str(args.source_dir.resolve()))
    import gr00t.model  # noqa: F401
    import gr00t.data.dataset.lerobot_episode_loader as loader_module
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType, VLAStepData

    video_decoder = "torchcodec"
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401
    except (ImportError, OSError, RuntimeError):
        install_pyav_video_fallback(loader_module)
        video_decoder = "pyav-sequential-frame-index-fallback"
    LeRobotEpisodeLoader = loader_module.LeRobotEpisodeLoader

    processor_dir = (
        args.checkpoint / "processor"
        if (args.checkpoint / "processor").is_dir()
        and not (args.checkpoint / "processor_config.json").exists()
        else args.checkpoint
    )
    processor = AutoProcessor.from_pretrained(
        processor_dir,
        model_name=str(args.backbone.resolve()),
        local_files_only=True,
        trust_remote_code=True,
        transformers_loading_kwargs={
            "local_files_only": True,
            "trust_remote_code": True,
        },
    )
    processor.eval()
    embodiment = EmbodimentTag.resolve(args.embodiment)
    all_configs = processor.get_modality_configs()
    if embodiment.value not in all_configs:
        raise ValueError(
            f"checkpoint does not support embodiment {embodiment.value!r}; "
            f"available: {sorted(all_configs)}"
        )
    modality_configs = copy.deepcopy({
        name: config
        for name, config in all_configs[embodiment.value].items()
        if name != "rl_info"
    })
    configured_video_keys = list(modality_configs["video"].modality_keys)
    if args.views is not None:
        if args.views < 1 or args.views > len(configured_video_keys):
            raise ValueError(
                f"--views must be in [1, {len(configured_video_keys)}] for "
                f"{embodiment.value!r}"
            )
        # This is the same selection rule used by FlashRT's public
        # capture_aux_multi.py: the dataset loader and processor share the
        # truncated ModalityConfig, so no discarded camera leaks downstream.
        modality_configs["video"].modality_keys = configured_video_keys[: args.views]
        # ``get_modality_configs`` exposes the processor's own configuration,
        # while ``modality_configs`` above is deliberately a deep copy for the
        # dataset loader. Keep the processor instance on the same view contract;
        # otherwise it still asks VLAStepData for every checkpoint-default view.
        processor.modality_configs[embodiment.value]["video"].modality_keys = list(
            modality_configs["video"].modality_keys
        )
    dataset = LeRobotEpisodeLoader(
        dataset_path=str(args.dataset),
        modality_configs=modality_configs,
    )
    if args.episode_index >= len(dataset):
        raise IndexError(
            f"episode index {args.episode_index} is outside dataset size {len(dataset)}"
        )
    episode = dataset[args.episode_index]
    if args.step_index >= len(episode):
        raise IndexError(
            f"step index {args.step_index} is outside episode length {len(episode)}"
        )
    step = extract_step_data(
        episode,
        step_index=args.step_index,
        modality_configs=modality_configs,
        embodiment_tag=embodiment,
        allow_padding=False,
    )
    raw_images = {name: np.stack(step.images[name]) for name in step.images}
    decoded_raw_images = raw_images
    if args.square_image_size is not None:
        if args.square_image_size <= 0:
            raise ValueError("--square-image-size must be positive")
        import cv2

        raw_images = {
            name: np.stack(
                [
                    cv2.resize(
                        frame,
                        (args.square_image_size, args.square_image_size),
                        interpolation=cv2.INTER_AREA,
                    )
                    for frame in frames
                ]
            )
            for name, frames in raw_images.items()
        }
    raw_states = {name: np.asarray(step.states[name]) for name in step.states}
    message = VLAStepData(
        images=raw_images,
        states=raw_states,
        actions={},
        text=step.text,
        embodiment=embodiment,
    )
    messages = [{"type": MessageType.EPISODE_STEP.value, "content": message}]

    def process() -> dict[str, Any]:
        processed = processor(messages)
        inputs = recursive_bf16(processor.collator([processed])["inputs"])
        missing = sorted(REQUIRED_INPUTS.difference(inputs))
        if missing:
            raise RuntimeError(f"processor output is missing {missing}")
        return inputs

    gc.collect()
    for _ in range(args.processor_warmup):
        process()
    gc.collect()
    processor_samples: list[float] = []
    inputs: dict[str, Any] | None = None
    for _ in range(args.iterations):
        start = time.perf_counter()
        inputs = process()
        processor_samples.append((time.perf_counter() - start) * 1_000.0)
    assert inputs is not None
    repeated_inputs = process()
    assert_deterministic(inputs, repeated_inputs)
    processor_state = numpy_tensor(inputs["state"], "state").copy()
    if args.zero_normalized_state:
        inputs = dict(inputs)
        inputs["state"] = torch.zeros_like(inputs["state"])

    config = json.loads((args.checkpoint / "config.json").read_text())
    action_horizon = int(config["action_horizon"])
    action_dim = int(config["max_action_dim"])
    if int(config["num_inference_timesteps"]) != 4:
        raise RuntimeError("the comparison contract requires four flow steps")

    fixture_name = args.fixture or (
        f"{args.checkpoint.name}-{embodiment.value}-"
        f"views{len(raw_images)}-episode{args.episode_index}-step{args.step_index}-v1"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor_npz = args.output_dir / "processor-output.npz"
    if args.noise_kind == "fixed-zero":
        initial_noise = np.zeros((1, action_horizon, action_dim), dtype=np.float32)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.noise_seed)
        # Round through BF16 before serializing so both runtimes consume exactly
        # the same representable values rather than rounding independently.
        initial_noise = (
            torch.randn(
                (1, action_horizon, action_dim),
                generator=generator,
                dtype=torch.float32,
            )
            .to(torch.bfloat16)
            .float()
            .numpy()
        )
    processor_payload = {
        name: numpy_tensor(inputs[name], name) for name in sorted(REQUIRED_INPUTS)
    }
    processor_payload["initial_noise"] = initial_noise
    np.savez(processor_npz, **processor_payload)
    signature = tensor_signature(inputs)
    processor_latency = summarize(processor_samples)
    processor_report = args.output_dir / "processor-benchmark.json"
    raw_input = {
        "dataset": str(args.dataset.resolve()),
        "dataset_episode_count": len(dataset),
        "episode_index": args.episode_index,
        "episode_length": len(episode),
        "step_index": args.step_index,
        "embodiment_name": args.embodiment,
        "embodiment_value": embodiment.value,
        "configured_video_keys": configured_video_keys,
        "square_image_size": args.square_image_size,
        "square_resize_interpolation": (
            "opencv.INTER_AREA" if args.square_image_size is not None else None
        ),
        "decoded_video_shapes": {
            name: json_shape(value) for name, value in decoded_raw_images.items()
        },
        "decoded_video_sha256": {
            name: array_sha256(value) for name, value in decoded_raw_images.items()
        },
        "video_keys": list(raw_images),
        "physical_camera_count": len(raw_images),
        "video_shapes": {name: json_shape(value) for name, value in raw_images.items()},
        "video_sha256": {name: array_sha256(value) for name, value in raw_images.items()},
        "video_delta_indices": list(modality_configs["video"].delta_indices),
        "state_keys": list(raw_states),
        "state_shapes": {name: json_shape(value) for name, value in raw_states.items()},
        "state_sha256": {name: array_sha256(value) for name, value in raw_states.items()},
        "normalized_state_override": {
            "kind": "zeros" if args.zero_normalized_state else "none",
            "processor_state_sha256": array_sha256(processor_state),
            "model_state_sha256": array_sha256(numpy_tensor(inputs["state"], "state")),
        },
        "language_key": modality_configs["language"].modality_keys[0],
        "language": step.text,
    }
    processor_report.write_text(
        json.dumps(
            {
                "schema": "apxinf.gr00t-n1.7.dataset-processor-benchmark.v1",
                "source_revision": revision,
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_config_sha256": sha256(args.checkpoint / "config.json"),
                "processor_input_npz": str(processor_npz.resolve()),
                "processor_input_npz_sha256": sha256(processor_npz),
                "raw_input": raw_input,
                "video_decoder": video_decoder,
                "model_boundary": signature,
                "initial_noise": {
                    "kind": args.noise_kind,
                    "seed": args.noise_seed if args.noise_kind == "seeded-normal" else None,
                    "shape": list(initial_noise.shape),
                    "sha256": array_sha256(initial_noise),
                },
                "deterministic_across_repeated_eval_calls": True,
                "timing_boundary": "VLAStepData through processor and collator",
                "warmup": args.processor_warmup,
                "iterations": args.iterations,
                "latency_ms": processor_latency,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    script_dir = Path(__file__).resolve().parent
    fixture_dir = args.output_dir / "fixture"
    run_checked(
        [
            sys.executable,
            str(script_dir / "prepare_gr00t_n1d7_fixture.py"),
            "--input",
            str(processor_npz),
            "--output-dir",
            str(fixture_dir),
            "--fixture",
            fixture_name,
            "--processor-report",
            str(processor_report),
            "--action-horizon",
            str(action_horizon),
            "--action-dim",
            str(action_dim),
            "--noise-kind",
            args.noise_kind,
        ]
        + (
            ["--noise-seed", str(args.noise_seed)]
            if args.noise_kind == "seeded-normal"
            else []
        )
    )

    report: dict[str, Any] = {
        "schema": "apxinf.gr00t-n1.7.dataset-to-action-validation.v1",
        "source_revision": revision,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_config_sha256": sha256(args.checkpoint / "config.json"),
        "fixture": fixture_name,
        "fixture_dir": str(fixture_dir),
        "raw_input": raw_input,
        "video_decoder": video_decoder,
        "model_boundary": signature,
        "processor_latency_ms": processor_latency,
        "processor_input_npz": str(processor_npz.resolve()),
        "processor_input_npz_sha256": sha256(processor_npz),
        "initial_noise": {
            "kind": args.noise_kind,
            "seed": args.noise_seed if args.noise_kind == "seeded-normal" else None,
            "shape": list(initial_noise.shape),
            "sha256": array_sha256(initial_noise),
        },
        "timing_note": (
            "processor and ApxInf Model Core are measured in separate loops and "
            "processes. The independent-median sum is the stable headline component "
            "summary. Index-paired sample sums reproduce NVIDIA's published benchmark "
            "calculation but are illustrative, not a joint wall-clock distribution. "
            "Neither value is a contiguous system E2E measurement"
        ),
    }
    report_path = args.output_dir / "report.json"
    if args.prepare_only:
        report["status"] = "processor-and-fixture-only"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"wrote {report_path}")
        return

    assert args.engine is not None
    apxinf_report = args.output_dir / "apxinf-action.json"
    engine_env = os.environ.copy()
    if args.tactics is not None:
        engine_env["APXINF_GR00T_BF16_TACTICS"] = str(args.tactics.resolve())
    run_checked(
        [
            str(args.engine),
            str(args.checkpoint),
            str(args.backbone),
            str(fixture_dir),
            str(args.device),
            str(args.model_warmup),
            str(args.iterations),
            "graph",
            str(apxinf_report),
        ],
        env=engine_env,
    )

    report["apxinf_report"] = str(apxinf_report)
    apxinf = json.loads(apxinf_report.read_text())
    model_core_samples = [
        float(value) for value in apxinf["model_core_latency_ms"].get("samples", [])
    ]
    if model_core_samples:
        model_core_latency = summarize(model_core_samples)
    else:
        model_core_latency = dict(apxinf["model_core_latency_ms"])
    report["model_core_latency_ms"] = model_core_latency
    report["e2e_component_median_sum_ms"] = (
        processor_latency["median"]
        + model_core_latency["median"]
    )
    if model_core_samples:
        if len(model_core_samples) != len(processor_samples):
            raise RuntimeError(
                "processor and Model Core sample counts differ: "
                f"{len(processor_samples)} versus {len(model_core_samples)}"
            )
        report["e2e_component_sample_sum_latency_ms"] = summarize(
            [
                processor_ms + model_core_ms
                for processor_ms, model_core_ms in zip(
                    processor_samples, model_core_samples, strict=True
                )
            ]
        )

    if args.run_nvidia_reference:
        reference = args.output_dir / "nvidia-reference.npz"
        parity = args.output_dir / "parity.json"
        run_checked(
            [
                sys.executable,
                str(script_dir / "gr00t_n1d7_reference_dump.py"),
                "--checkpoint",
                str(args.checkpoint),
                "--backbone",
                str(args.backbone),
                "--source-dir",
                str(args.source_dir),
                "--input-npz",
                str(processor_npz),
                "--output",
                str(reference),
                "--device",
                f"cuda:{args.device}",
                "--fixture",
                fixture_name,
            ]
        )
        comparison = subprocess.run(
            [
                sys.executable,
                str(script_dir / "compare_gr00t_n1d7.py"),
                "--reference",
                str(reference),
                "--apxinf",
                str(apxinf_report),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        parity.write_text(comparison.stdout)
        report["nvidia_reference"] = str(reference)
        report["parity_report"] = str(parity)
        report["parity"] = json.loads(parity.read_text())

    if args.run_nvidia_benchmark:
        nvidia_report = args.output_dir / "nvidia-benchmark.json"
        command = [
            sys.executable,
            str(script_dir / "benchmark_nvidia_gr00t_n1d7.py"),
            "--checkpoint",
            str(args.checkpoint),
            "--backbone",
            str(args.backbone),
            "--source-dir",
            str(args.source_dir),
            "--input-npz",
            str(processor_npz),
            "--output",
            str(nvidia_report),
            "--device",
            f"cuda:{args.device}",
            "--warmup",
            str(args.model_warmup),
            "--iterations",
            str(args.iterations),
            "--processor-median-ms",
            str(processor_latency["median"]),
            "--processor-report",
            str(processor_report),
            "--modes",
            *args.nvidia_modes,
        ]
        if args.trt_engine_dir is not None:
            command.extend(["--trt-engine-dir", str(args.trt_engine_dir)])
        if args.run_nvidia_reference:
            command.extend(["--reference-npz", str(reference)])
        run_checked(command)
        report["nvidia_benchmark"] = str(nvidia_report)
        report["nvidia_results"] = json.loads(nvidia_report.read_text())

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
