#!/usr/bin/env python3
"""Run GR00T N1.7 ApxInf inference in NVIDIA's LIBERO rollout harness.

The NVIDIA checkpoint processor and action decoder are used unchanged. Only
the model-core call is replaced by ``apxinf_py.Gr00tModel`` so the resulting
success rate measures the deployed ApxInf policy rather than a reimplemented
pre/post-processing approximation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch


def _round_to_bf16(value: np.ndarray) -> np.ndarray:
    """Round f32 to BF16 (round-to-nearest-even), matching Gr00tPolicy.

    Mirrors ``apxinf.policies.impls.gr00t._round_to_bf16`` byte-for-byte so the
    LIBERO rollout consumes the exact noise the deployed policy produces.
    """
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_artifact(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": sha256(resolved)}


def array_signature(value: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("utf-8"))
    digest.update(
        json.dumps(list(contiguous.shape), separators=(",", ":")).encode("utf-8")
    )
    digest.update(contiguous.tobytes(order="C"))
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "sha256": digest.hexdigest(),
    }


def git_revision(path: Path) -> dict[str, object]:
    revision = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"revision": revision, "dirty": dirty}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--backbone", required=True, type=Path)
    parser.add_argument(
        "--precision", choices=("bf16", "fp8", "int8"), required=True
    )
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--tactics", type=Path)
    parser.add_argument("--env-name", required=True)
    parser.add_argument(
        "--views",
        type=int,
        choices=(1, 2),
        default=2,
        help="Number of physical LIBERO camera streams presented to the policy.",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="parallel simulator environments; model calls remain serialized at batch 1",
    )
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--noise-mode",
        choices=("fixed", "stream"),
        default="stream",
        help=(
            "stream draws the seeded sequence used for task evaluation; fixed "
            "reuses one BF16 noise tensor and is diagnostic only"
        ),
    )
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument(
        "--record-first-normalized-action",
        action="store_true",
        help="store the first normalized [1,H,D] action for paired precision parity",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.precision == "fp8" and args.calibration is None:
        parser.error("--precision fp8 requires --calibration")
    return args


def main() -> None:
    args = parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    if args.calibration is not None:
        os.environ["APXINF_GR00T_FP8_CALIBRATION"] = str(args.calibration.resolve())
    if args.tactics is not None:
        document = json.loads(args.tactics.read_text())
        records = document.get("records", [])
        os.environ["APXINF_FP8_BF16_CUBLASLT_TACTICS"] = ";".join(
            f"{record['key']['m']},{record['key']['n']},{record['key']['k']}="
            f"{record['tactic']['id']}"
            for record in records
        )

    sys.path.insert(0, str(args.source_dir.resolve()))
    import gr00t.model  # noqa: F401
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType, VLAStepData
    from gr00t.eval._horizon_contract import PolicyHorizonSpec
    from gr00t.eval.rollout_policy import (
        MultiStepConfig,
        VideoConfig,
        WrapperConfigs,
        run_rollout_gymnasium_policy,
    )
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from gr00t.policy.policy import BasePolicy
    from transformers import AutoProcessor
    import apxinf_py

    # Maturin editable installs expose a Python package that re-exports the
    # native submodule.  Hash the loaded shared object, not its tiny __init__.py.
    apxinf_extension = getattr(apxinf_py, "apxinf_py", apxinf_py)

    class ApxInfGr00tPolicy(Gr00tPolicy):
        def __init__(self) -> None:
            BasePolicy.__init__(self, strict=True)
            processor_dir = (
                args.checkpoint / "processor"
                if (args.checkpoint / "processor").is_dir()
                and not (args.checkpoint / "processor_config.json").exists()
                else args.checkpoint
            )
            self.processor = AutoProcessor.from_pretrained(
                processor_dir,
                model_name=str(args.backbone.resolve()),
                local_files_only=True,
                trust_remote_code=True,
                transformers_loading_kwargs={
                    "local_files_only": True,
                    "trust_remote_code": True,
                },
            )
            self.processor.eval()
            self.embodiment_tag = EmbodimentTag.resolve("libero_sim")
            all_configs = self.processor.get_modality_configs()
            self.modality_configs = {
                key: value
                for key, value in all_configs[self.embodiment_tag.value].items()
                if key != "rl_info"
            }
            video_config = self.modality_configs["video"]
            available_video_keys = list(video_config.modality_keys)
            if args.views > len(available_video_keys):
                raise ValueError(
                    f"requested {args.views} views, but checkpoint exposes only "
                    f"{len(available_video_keys)}: {available_video_keys}"
                )
            required_video_keys = ["image"] if args.views == 1 else ["image", "wrist_image"]
            missing_video_keys = [
                key for key in required_video_keys if key not in available_video_keys
            ]
            if missing_video_keys:
                raise ValueError(
                    f"checkpoint is missing required LIBERO camera keys {missing_video_keys}; "
                    f"available keys: {available_video_keys}"
                )
            video_config.modality_keys = required_video_keys
            self.video_keys = list(video_config.modality_keys)
            self.collate_fn = self.processor.collator
            self.language_key = self.modality_configs["language"].modality_keys[0]
            self.model = apxinf_py.Gr00tModel.load(
                args.checkpoint,
                args.backbone,
                "cuda:0",
                args.precision,
                args.calibration,
            )
            # Match the shipped Gr00tPolicy noise path exactly: numpy
            # default_rng draws (not torch) rounded through BF16 with the same
            # round-to-nearest-even helper, so this harness validates the noise
            # sequence the deployed policy actually consumes.
            self._rng = np.random.default_rng(args.seed)
            self.fixed_noise = _round_to_bf16(
                self._rng.standard_normal(
                    (1, self.model.action_horizon, self.model.action_dim),
                    dtype=np.float32,
                )
            )
            # Keep stream mode's first draw identical to the historical seeded
            # sequence; constructing the fixed tensor must not advance it.
            self._rng = np.random.default_rng(args.seed)
            self.inference_count = 0
            self.inference_seconds = 0.0
            self.first_normalized_action = None
            self.first_model_input_signatures = None

        def _get_action(self, observation, options=None):
            del options
            unbatched = self._unbatch_observation(observation)
            action_batches = {}
            for obs in unbatched:
                step = VLAStepData(
                    images=obs["video"],
                    states=obs["state"],
                    actions={},
                    text=obs["language"][self.language_key][0],
                    embodiment=self.embodiment_tag,
                )
                processed = self.processor(
                    [{"type": MessageType.EPISODE_STEP.value, "content": step}]
                )
                inputs = self.collate_fn([processed])["inputs"]
                noise = self.fixed_noise.copy()
                if args.noise_mode == "stream":
                    noise = _round_to_bf16(
                        self._rng.standard_normal(
                            (1, self.model.action_horizon, self.model.action_dim),
                            dtype=np.float32,
                        )
                    )
                pixel_values = inputs["pixel_values"].float().cpu().numpy()
                image_grid_thw = inputs["image_grid_thw"].to(torch.uint32).cpu().numpy()
                token_ids = inputs["input_ids"].to(torch.uint32).cpu().numpy().reshape(-1)
                attention_mask = (
                    inputs["attention_mask"].to(torch.uint8).cpu().numpy().reshape(-1)
                )
                normalized_state = inputs["state"].float().cpu().numpy()
                embodiment_id = int(inputs["embodiment_id"].reshape(-1)[0])
                if self.first_model_input_signatures is None:
                    self.first_model_input_signatures = {
                        "pixel_values": array_signature(pixel_values),
                        "image_grid_thw": array_signature(image_grid_thw),
                        "token_ids": array_signature(token_ids),
                        "attention_mask": array_signature(attention_mask),
                        "state": array_signature(normalized_state),
                        "noise": array_signature(noise),
                        "embodiment_id": embodiment_id,
                    }
                started = time.perf_counter()
                normalized = np.asarray(
                    self.model.infer(
                        pixel_values,
                        image_grid_thw,
                        token_ids,
                        attention_mask,
                        normalized_state,
                        embodiment_id,
                        noise,
                    ),
                    dtype=np.float32,
                )[None]
                if self.first_normalized_action is None:
                    self.first_normalized_action = normalized.copy()
                self.inference_seconds += time.perf_counter() - started
                self.inference_count += 1
                batched_states = {
                    key: np.expand_dims(step.states[key], axis=0)
                    for key in self.modality_configs["state"].modality_keys
                }
                actions = self.processor.decode_action(
                    normalized, self.embodiment_tag, batched_states
                )
                for key, value in actions.items():
                    action_batches.setdefault(key, []).append(value.astype(np.float32))
            return {
                key: np.concatenate(values, axis=0)
                for key, values in action_batches.items()
            }, {"apxinf_model_seconds": self.inference_seconds}

        def reset(self, options=None):
            del options
            return {}

    policy = Gr00tSimPolicyWrapper(ApxInfGr00tPolicy())
    contract = PolicyHorizonSpec.from_policy(policy, n_action_steps=args.n_action_steps)
    wrappers = WrapperConfigs(
        multistep=MultiStepConfig(
            contract=contract,
            max_episode_steps=args.max_episode_steps,
            terminate_on_success=True,
        ),
        video=VideoConfig(
            video_dir=str(args.video_dir) if args.video_dir else None,
            max_episode_steps=args.max_episode_steps,
        ),
    )
    env_name, successes, info = run_rollout_gymnasium_policy(
        env_name=args.env_name,
        policy=policy,
        wrapper_configs=wrappers,
        n_episodes=args.episodes,
        n_envs=min(args.n_envs, args.episodes),
        seed=args.seed,
    )
    # Some vector-environment implementations finish more than one environment
    # on the final step.  Keep the requested prefix so the receipt has exactly
    # the declared sample count and the campaign cannot silently over-count.
    successes = list(successes)[: args.episodes]
    info = {
        key: value[: args.episodes] if isinstance(value, list) else value
        for key, value in info.items()
    }
    result = {
        "schema": "apxinf.gr00t-n1.7.libero-rollout.v1",
        "env_name": env_name,
        "precision": args.precision,
        "views": args.views,
        "video_keys": policy.policy.video_keys,
        "seed": args.seed,
        "episodes": len(successes),
        "successes": int(sum(bool(value) for value in successes)),
        "success_rate": float(np.mean(successes)),
        "episode_successes": [bool(value) for value in successes],
        "episode_info": info,
        "model_calls": policy.policy.inference_count,
        "model_seconds": policy.policy.inference_seconds,
        "runtime_contract": {
            "action_horizon": int(policy.policy.model.action_horizon),
            "action_dim": int(policy.policy.model.action_dim),
            "n_action_steps": args.n_action_steps,
            "max_episode_steps": args.max_episode_steps,
            "n_envs": min(args.n_envs, args.episodes),
            "noise_mode": args.noise_mode,
            "hf_hub_offline": os.environ["HF_HUB_OFFLINE"],
            "mujoco_gl": os.environ["MUJOCO_GL"],
            "pyopengl_platform": os.environ["PYOPENGL_PLATFORM"],
        },
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "artifacts": {
            "apxinf_source_git": git_revision(Path(__file__).resolve().parents[1]),
            "source_dir": str(args.source_dir.resolve()),
            "source_git": git_revision(args.source_dir),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_config": optional_artifact(args.checkpoint / "config.json"),
            "backbone": str(args.backbone.resolve()),
            "backbone_config": optional_artifact(args.backbone / "config.json"),
            "apxinf_python_extension": optional_artifact(
                Path(apxinf_extension.__file__)
            ),
            "calibration": optional_artifact(args.calibration),
            "tactics": optional_artifact(args.tactics),
        },
    }
    if args.record_first_normalized_action:
        first_action = policy.policy.first_normalized_action
        if first_action is None:
            raise RuntimeError("rollout completed without producing a normalized action")
        result["first_normalized_action"] = first_action.reshape(-1).tolist()
        result["first_normalized_action_shape"] = list(first_action.shape)
        result["first_model_input_signatures"] = policy.policy.first_model_input_signatures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
