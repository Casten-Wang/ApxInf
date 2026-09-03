#!/usr/bin/env python3
"""Audit GR00T N1.7 LIBERO rollout receipts for selected precisions.

The evaluator writes one JSON receipt per task/seed/precision.  This tool
fails closed on missing pairs, duplicate pairs, mixed view topologies, malformed
results, non-finite accounting fields, or a task set that changes across seeds.
It intentionally reports observed success rates without prescribing a minimum;
the PR must disclose both numerator and denominator.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "apxinf.gr00t-n1.7.libero-rollout.v1"
SUPPORTED_PRECISIONS = ("bf16", "fp8", "int8", "w8a8")


def canonical_precision(precision: str) -> str:
    """Expose one public INT8 name while accepting historical W8A8 receipts."""
    return "int8" if precision == "w8a8" else precision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        action="append",
        required=True,
        type=Path,
        help="Directory containing task*-{bf16,fp8,int8}-seed*.json; repeatable.",
    )
    parser.add_argument("--views", required=True, type=int, choices=(1, 2))
    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=SUPPORTED_PRECISIONS,
        default=("bf16", "fp8"),
        help="precisions required in every task/seed; defaults to paired BF16/FP8",
    )
    parser.add_argument("--expected-tasks", type=int, default=10)
    parser.add_argument("--expected-seeds", type=int, default=None)
    parser.add_argument(
        "--expected-episodes-per-precision",
        type=int,
        default=None,
        help="Fail unless each precision has exactly this many completed episodes.",
    )
    parser.add_argument(
        "--require-provenance",
        action="store_true",
        help="Require hash-bound calibration/tactics and a recorded runtime contract.",
    )
    parser.add_argument(
        "--require-parity",
        action="store_true",
        help="require and compare paired BF16/quantized first normalized actions",
    )
    parser.add_argument("--min-cosine", type=float, default=0.997)
    parser.add_argument("--max-relative-l2", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--campaign-manifest",
        type=Path,
        default=None,
        help="bind receipts to runner/evaluator and artifact hashes recorded at launch",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_receipts(paths: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    receipts: list[tuple[Path, dict[str, Any]]] = []
    for directory in paths:
        if not directory.is_dir():
            raise ValueError(f"input directory does not exist: {directory}")
        for path in sorted(directory.glob("task*-*-seed*.json")):
            document = json.loads(path.read_text())
            receipts.append((path, document))
    if not receipts:
        raise ValueError("no rollout receipts found")
    return receipts


def require_receipt(
    path: Path,
    document: dict[str, Any],
    views: int,
    require_provenance: bool,
    require_parity: bool,
) -> None:
    if document.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unexpected schema {document.get('schema')!r}")
    precision = document.get("precision")
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(f"{path}: unexpected precision {precision!r}")
    if document.get("views") != views:
        raise ValueError(f"{path}: views={document.get('views')}, expected {views}")
    expected_keys = ["image"] if views == 1 else ["image", "wrist_image"]
    if document.get("video_keys") != expected_keys:
        raise ValueError(
            f"{path}: video_keys={document.get('video_keys')!r}, "
            f"expected {expected_keys!r}"
        )
    env_name = document.get("env_name")
    if not isinstance(env_name, str) or not env_name.startswith("libero_sim/"):
        raise ValueError(f"{path}: invalid LIBERO env_name {env_name!r}")
    episodes = document.get("episodes")
    successes = document.get("successes")
    episode_successes = document.get("episode_successes")
    if not isinstance(episodes, int) or episodes <= 0:
        raise ValueError(f"{path}: invalid episodes {episodes!r}")
    if not isinstance(successes, int) or not 0 <= successes <= episodes:
        raise ValueError(f"{path}: invalid successes {successes!r}")
    if not isinstance(episode_successes, list) or len(episode_successes) != episodes:
        raise ValueError(f"{path}: episode_successes length mismatch")
    if sum(bool(value) for value in episode_successes) != successes:
        raise ValueError(f"{path}: success counter mismatch")
    expected_rate = successes / episodes
    rate = document.get("success_rate")
    if not isinstance(rate, (int, float)) or not math.isclose(
        float(rate), expected_rate, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{path}: success_rate mismatch")
    for key in ("model_calls", "model_seconds", "seed"):
        value = document.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{path}: invalid {key}={value!r}")
    if document["model_calls"] <= 0 or document["model_seconds"] <= 0:
        raise ValueError(f"{path}: non-positive model accounting")
    if require_provenance:
        runtime = document.get("runtime_contract")
        if not isinstance(runtime, dict):
            raise ValueError(f"{path}: missing runtime_contract")
        for key in (
            "action_horizon",
            "action_dim",
            "n_action_steps",
            "max_episode_steps",
            "n_envs",
        ):
            if not isinstance(runtime.get(key), int) or runtime[key] <= 0:
                raise ValueError(f"{path}: invalid runtime_contract.{key}")
        if runtime.get("noise_mode") != "stream":
            raise ValueError(f"{path}: formal task evaluation requires stream noise mode")
        expected_environment = {
            "hf_hub_offline": "1",
            "mujoco_gl": "egl",
            "pyopengl_platform": "egl",
        }
        for key, expected in expected_environment.items():
            if runtime.get(key) != expected:
                raise ValueError(f"{path}: runtime_contract.{key} must be {expected!r}")
        artifacts = document.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError(f"{path}: missing artifacts provenance")
        for key in ("source_dir", "checkpoint", "backbone"):
            if not isinstance(artifacts.get(key), str) or not artifacts[key]:
                raise ValueError(f"{path}: invalid artifacts.{key}")
        source_git = artifacts.get("source_git")
        if (
            not isinstance(source_git, dict)
            or not isinstance(source_git.get("revision"), str)
            or len(source_git["revision"]) != 40
            or not isinstance(source_git.get("dirty"), bool)
        ):
            raise ValueError(f"{path}: invalid artifacts.source_git")
        if source_git["dirty"]:
            raise ValueError(f"{path}: Isaac-GR00T source checkout is dirty")
        apxinf_source_git = artifacts.get("apxinf_source_git")
        if (
            not isinstance(apxinf_source_git, dict)
            or not isinstance(apxinf_source_git.get("revision"), str)
            or len(apxinf_source_git["revision"]) != 40
            or not isinstance(apxinf_source_git.get("dirty"), bool)
        ):
            raise ValueError(f"{path}: invalid artifacts.apxinf_source_git")
        if apxinf_source_git["dirty"]:
            raise ValueError(f"{path}: ApxInf source checkout is dirty")
        software = document.get("software")
        if not isinstance(software, dict):
            raise ValueError(f"{path}: missing software provenance")
        for key in ("python", "numpy", "torch", "torch_cuda"):
            if not isinstance(software.get(key), str) or not software[key]:
                raise ValueError(f"{path}: invalid software.{key}")
        for key in ("checkpoint_config", "backbone_config"):
            artifact = artifacts.get(key)
            if not isinstance(artifact, dict):
                raise ValueError(f"{path}: missing artifacts.{key}")
            digest = artifact.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"{path}: invalid artifacts.{key}.sha256")
        required_artifacts = ["tactics", "apxinf_python_extension"]
        if document["precision"] == "fp8":
            required_artifacts.append("calibration")
        for key in required_artifacts:
            artifact = artifacts.get(key)
            if not isinstance(artifact, dict):
                raise ValueError(f"{path}: missing artifacts.{key}")
            digest = artifact.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"{path}: invalid artifacts.{key}.sha256")
    if require_parity:
        shape = document.get("first_normalized_action_shape")
        values = document.get("first_normalized_action")
        if shape != [1, 40, 132]:
            raise ValueError(f"{path}: invalid first action shape {shape!r}")
        if not isinstance(values, list) or len(values) != 40 * 132:
            raise ValueError(f"{path}: invalid first action element count")
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in values
        ):
            raise ValueError(f"{path}: first normalized action is not finite")
        signatures = document.get("first_model_input_signatures")
        required_inputs = {
            "pixel_values",
            "image_grid_thw",
            "token_ids",
            "attention_mask",
            "state",
            "noise",
            "embodiment_id",
        }
        if not isinstance(signatures, dict) or set(signatures) != required_inputs:
            raise ValueError(f"{path}: incomplete first model-input signatures")
        for key in required_inputs - {"embodiment_id"}:
            signature = signatures[key]
            if not isinstance(signature, dict):
                raise ValueError(f"{path}: invalid input signature {key}")
            if not isinstance(signature.get("shape"), list):
                raise ValueError(f"{path}: missing input shape for {key}")
            digest = signature.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"{path}: invalid input hash for {key}")


def parity_metrics(reference: list[float], candidate: list[float]) -> dict[str, float]:
    if len(reference) != len(candidate):
        raise ValueError(
            f"parity vectors differ in length: {len(reference)} != {len(candidate)}"
        )
    reference_l2 = math.sqrt(sum(value * value for value in reference))
    candidate_l2 = math.sqrt(sum(value * value for value in candidate))
    differences = [
        candidate_value - reference_value
        for reference_value, candidate_value in zip(reference, candidate)
    ]
    error_l2 = math.sqrt(sum(value * value for value in differences))
    dot = sum(
        reference_value * candidate_value
        for reference_value, candidate_value in zip(reference, candidate)
    )
    if reference_l2 == 0.0 or candidate_l2 == 0.0:
        cosine = 1.0 if reference_l2 == candidate_l2 == 0.0 else 0.0
    else:
        cosine = dot / (reference_l2 * candidate_l2)
    return {
        "cosine": cosine,
        "relative_l2": (
            error_l2 / reference_l2
            if reference_l2
            else (0.0 if error_l2 == 0.0 else math.inf)
        ),
        "max_abs": max(abs(value) for value in differences),
        "mean_abs": sum(abs(value) for value in differences) / len(differences),
    }


def main() -> None:
    args = parse_args()
    precisions = tuple(
        dict.fromkeys(canonical_precision(value) for value in args.precisions)
    )
    if args.campaign_manifest is not None:
        args.require_provenance = True
    if args.expected_episodes_per_precision is None and args.campaign_manifest is None:
        raise ValueError(
            "provide --expected-episodes-per-precision or --campaign-manifest "
            "to prevent incomplete campaigns from passing"
        )
    if "bf16" in precisions and len(precisions) == 2:
        args.require_parity = True
    if args.require_parity and ("bf16" not in precisions or len(precisions) != 2):
        raise ValueError(
            "--require-parity requires exactly bf16 and one quantized precision"
        )
    receipts = load_receipts(args.input_dir)
    by_key: dict[tuple[int, str, str], tuple[Path, dict[str, Any]]] = {}
    tasks_by_seed: dict[int, set[str]] = defaultdict(set)
    for path, document in receipts:
        require_receipt(
            path, document, args.views, args.require_provenance, args.require_parity
        )
        seed = int(document["seed"])
        precision = canonical_precision(str(document["precision"]))
        document["precision"] = precision
        if precision not in precisions:
            raise ValueError(
                f"{path}: precision {precision!r} was not requested; "
                f"expected only {list(precisions)!r}"
            )
        env_name = str(document["env_name"])
        key = (seed, env_name, precision)
        if key in by_key:
            raise ValueError(f"duplicate receipt for {key}: {by_key[key][0]} and {path}")
        by_key[key] = (path, document)
        tasks_by_seed[seed].add(env_name)

    seeds = sorted(tasks_by_seed)
    if args.expected_seeds is not None and len(seeds) != args.expected_seeds:
        raise ValueError(f"found {len(seeds)} seeds, expected {args.expected_seeds}")
    canonical_tasks = tasks_by_seed[seeds[0]]
    if len(canonical_tasks) != args.expected_tasks:
        raise ValueError(
            f"seed {seeds[0]} has {len(canonical_tasks)} tasks, "
            f"expected {args.expected_tasks}"
        )
    for seed in seeds:
        if tasks_by_seed[seed] != canonical_tasks:
            missing = sorted(canonical_tasks - tasks_by_seed[seed])
            extra = sorted(tasks_by_seed[seed] - canonical_tasks)
            raise ValueError(f"seed {seed}: task mismatch; missing={missing}, extra={extra}")
        for env_name in canonical_tasks:
            for precision in precisions:
                if (seed, env_name, precision) not in by_key:
                    raise ValueError(
                        f"missing paired receipt: seed={seed}, env={env_name}, "
                        f"precision={precision}"
                    )

    summary: dict[str, Any] = {
        "schema": "apxinf.gr00t-n1.7.libero-campaign-audit.v1",
        "views": args.views,
        "video_keys": ["image"] if args.views == 1 else ["image", "wrist_image"],
        "seeds": seeds,
        "tasks_per_seed": len(canonical_tasks),
        "paired_task_seeds": len(seeds) * len(canonical_tasks),
        "precisions": list(precisions),
        "precision": {},
    }
    if args.require_provenance:
        provenance = {}
        for precision in precisions:
            documents = [
                document
                for (_, _, candidate), (_, document) in by_key.items()
                if candidate == precision
            ]
            for key in ("source_dir", "checkpoint", "backbone"):
                values = {document["artifacts"][key] for document in documents}
                if len(values) != 1:
                    raise ValueError(f"{precision} uses multiple {key} values: {values}")
            source_revisions = {
                json.dumps(document["artifacts"]["source_git"], sort_keys=True)
                for document in documents
            }
            apxinf_source_revisions = {
                json.dumps(document["artifacts"]["apxinf_source_git"], sort_keys=True)
                for document in documents
            }
            software_versions = {
                json.dumps(document["software"], sort_keys=True)
                for document in documents
            }
            checkpoint_configs = {
                document["artifacts"]["checkpoint_config"]["sha256"]
                for document in documents
            }
            backbone_configs = {
                document["artifacts"]["backbone_config"]["sha256"]
                for document in documents
            }
            if len(source_revisions) != 1:
                raise ValueError(f"{precision} uses multiple source revisions")
            if len(apxinf_source_revisions) != 1:
                raise ValueError(f"{precision} uses multiple ApxInf source revisions")
            if len(software_versions) != 1:
                raise ValueError(f"{precision} uses multiple software environments")
            if len(checkpoint_configs) != 1 or len(backbone_configs) != 1:
                raise ValueError(f"{precision} uses multiple model config hashes")
            tactic_hashes = {
                document["artifacts"]["tactics"]["sha256"] for document in documents
            }
            if len(tactic_hashes) != 1:
                raise ValueError(f"{precision} uses multiple tactic hashes")
            calibration_hashes = {
                document["artifacts"]["calibration"]["sha256"]
                for document in documents
                if document["artifacts"]["calibration"] is not None
            }
            if precision == "fp8" and len(calibration_hashes) != 1:
                raise ValueError("FP8 uses multiple calibration hashes")
            extension_hashes = {
                document["artifacts"]["apxinf_python_extension"]["sha256"]
                for document in documents
            }
            if len(extension_hashes) != 1:
                raise ValueError(f"{precision} uses multiple Python extension hashes")
            runtime_contracts = {
                json.dumps(document["runtime_contract"], sort_keys=True)
                for document in documents
            }
            if len(runtime_contracts) != 1:
                raise ValueError(f"{precision} uses multiple runtime contracts")
            provenance[precision] = {
                "source_dir": documents[0]["artifacts"]["source_dir"],
                "checkpoint": documents[0]["artifacts"]["checkpoint"],
                "backbone": documents[0]["artifacts"]["backbone"],
                "tactics_sha256": next(iter(tactic_hashes)),
                "calibration_sha256": (
                    next(iter(calibration_hashes)) if calibration_hashes else None
                ),
                "apxinf_python_extension_sha256": next(iter(extension_hashes)),
                "runtime_contract": documents[0]["runtime_contract"],
                "source_git": documents[0]["artifacts"]["source_git"],
                "apxinf_source_git": documents[0]["artifacts"]["apxinf_source_git"],
                "software": documents[0]["software"],
                "checkpoint_config_sha256": next(iter(checkpoint_configs)),
                "backbone_config_sha256": next(iter(backbone_configs)),
            }
        reference_precision = precisions[0]
        common_keys = (
            "checkpoint",
            "backbone",
            "source_dir",
            "source_git",
            "apxinf_source_git",
            "software",
            "checkpoint_config_sha256",
            "backbone_config_sha256",
            "tactics_sha256",
            "apxinf_python_extension_sha256",
            "runtime_contract",
        )
        for precision in precisions[1:]:
            for key in common_keys:
                if provenance[reference_precision][key] != provenance[precision][key]:
                    raise ValueError(
                        f"{reference_precision} and {precision} {key} values differ"
                    )
        summary["provenance"] = provenance
        if args.campaign_manifest is not None:
            manifest = json.loads(args.campaign_manifest.read_text())
            if manifest.get("schema") != "apxinf.gr00t-n1.7.libero-campaign-manifest.v1":
                raise ValueError("invalid campaign manifest schema")
            if manifest.get("views") != args.views:
                raise ValueError("campaign manifest view count differs from receipts")
            if manifest.get("source_git") != provenance[reference_precision]["source_git"]:
                raise ValueError("campaign manifest source revision differs from receipts")
            if (
                manifest.get("apxinf_source_git")
                != provenance[reference_precision]["apxinf_source_git"]
            ):
                raise ValueError(
                    "campaign manifest ApxInf source revision differs from receipts"
                )
            if set(manifest.get("tasks", [])) != canonical_tasks:
                raise ValueError("campaign manifest task set differs from receipts")
            manifest_precisions = [
                canonical_precision(str(value)) for value in manifest.get("precisions", [])
            ]
            if manifest_precisions != list(precisions):
                raise ValueError("campaign manifest precision list differs from receipts")
            if len(seeds) != 1 or manifest.get("seed") != seeds[0]:
                raise ValueError("campaign manifest seed differs from receipts")
            episodes_per_task = manifest.get("episodes_per_task")
            if not isinstance(episodes_per_task, int) or episodes_per_task <= 0:
                raise ValueError("campaign manifest has invalid episodes_per_task")
            if any(
                document["episodes"] != episodes_per_task
                for _, document in by_key.values()
            ):
                raise ValueError("campaign manifest episode count differs from receipts")
            if manifest.get("episodes_per_precision") != (
                episodes_per_task * len(canonical_tasks)
            ):
                raise ValueError("campaign manifest episodes_per_precision is inconsistent")
            runtime = provenance[reference_precision]["runtime_contract"]
            for manifest_key, runtime_key in (
                ("n_envs", "n_envs"),
                ("n_action_steps", "n_action_steps"),
                ("max_episode_steps", "max_episode_steps"),
                ("noise_mode", "noise_mode"),
            ):
                if manifest.get(manifest_key) != runtime.get(runtime_key):
                    raise ValueError(
                        f"campaign manifest {manifest_key} differs from receipts"
                    )
            paths = manifest.get("paths")
            if not isinstance(paths, dict):
                raise ValueError("campaign manifest is missing paths")
            expected_hashes = {
                "calibration_sha256": (
                    provenance["fp8"]["calibration_sha256"]
                    if "fp8" in provenance
                    else None
                ),
                "tactics_sha256": provenance[reference_precision]["tactics_sha256"],
                "evaluator_sha256": sha256(
                    Path(__file__).resolve().with_name("eval_gr00t_n1d7_libero.py")
                ),
                "auditor_sha256": sha256(Path(__file__).resolve()),
                "campaign_runner_sha256": sha256(
                    Path(__file__).resolve().with_name(
                        "run_gr00t_n1d7_libero_campaign.py"
                    )
                ),
            }
            for key, expected in expected_hashes.items():
                if paths.get(key) != expected:
                    raise ValueError(
                        f"campaign manifest {key}={paths.get(key)!r}, "
                        f"expected {expected!r}"
                    )
            expected_paths = {
                "source_dir": provenance[reference_precision]["source_dir"],
                "checkpoint": provenance[reference_precision]["checkpoint"],
                "backbone": provenance[reference_precision]["backbone"],
            }
            for key, expected in expected_paths.items():
                if paths.get(key) != expected:
                    raise ValueError(
                        f"campaign manifest {key}={paths.get(key)!r}, "
                        f"expected {expected!r}"
                    )
            summary["campaign_manifest"] = {
                "path": str(args.campaign_manifest.resolve()),
                "sha256": sha256(args.campaign_manifest),
            }
    if args.require_parity:
        quantized_precision = next(
            precision for precision in precisions if precision != "bf16"
        )
        parity = []
        for seed in seeds:
            for env_name in sorted(canonical_tasks):
                bf16 = by_key[(seed, env_name, "bf16")][1]
                quantized = by_key[(seed, env_name, quantized_precision)][1]
                if (
                    bf16["first_model_input_signatures"]
                    != quantized["first_model_input_signatures"]
                ):
                    raise ValueError(
                        f"BF16/{quantized_precision.upper()} model inputs differ for "
                        f"seed={seed}, env={env_name}"
                    )
                metrics = parity_metrics(
                    bf16["first_normalized_action"],
                    quantized["first_normalized_action"],
                )
                metrics.update(
                    {
                        "seed": seed,
                        "env_name": env_name,
                        "passed": metrics["cosine"] >= args.min_cosine
                        and metrics["relative_l2"] <= args.max_relative_l2,
                    }
                )
                parity.append(metrics)
        summary["parity"] = {
            "thresholds": {
                "min_cosine": args.min_cosine,
                "max_relative_l2": args.max_relative_l2,
                "note": "max_abs and mean_abs are diagnostics only",
            },
            "pairs": len(parity),
            "minimum_cosine": min(row["cosine"] for row in parity),
            "maximum_relative_l2": max(row["relative_l2"] for row in parity),
            "maximum_absolute_error": max(row["max_abs"] for row in parity),
            "maximum_mean_absolute_error": max(row["mean_abs"] for row in parity),
            "passed": all(row["passed"] for row in parity),
            "results": parity,
        }
        if not summary["parity"]["passed"]:
            raise ValueError("at least one paired BF16/FP8 first action failed parity")
    for precision in precisions:
        documents = [
            document
            for (seed, env_name, candidate), (_, document) in by_key.items()
            if candidate == precision
        ]
        episodes = sum(int(document["episodes"]) for document in documents)
        successes = sum(int(document["successes"]) for document in documents)
        summary["precision"][precision] = {
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes,
            "model_calls": sum(int(document["model_calls"]) for document in documents),
            "model_seconds": sum(float(document["model_seconds"]) for document in documents),
        }
        if (
            args.expected_episodes_per_precision is not None
            and episodes != args.expected_episodes_per_precision
        ):
            raise ValueError(
                f"{precision} has {episodes} episodes, expected "
                f"{args.expected_episodes_per_precision}"
            )

    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
