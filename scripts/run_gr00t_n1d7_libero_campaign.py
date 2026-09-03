#!/usr/bin/env python3
"""Run a resumable GR00T N1.7 LIBERO-10 precision campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


LIBERO_10_ENVS = (
    "libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "libero_sim/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "libero_sim/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "libero_sim/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "libero_sim/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "libero_sim/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "libero_sim/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "libero_sim/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "libero_sim/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--backbone", required=True, type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--tactics", required=True, type=Path)
    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=("bf16", "fp8", "int8"),
        default=("bf16", "fp8"),
        help="precisions to evaluate; defaults to the historical paired BF16/FP8 run",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=10,
        help="parallel simulator environments; ApxInf inference remains batch-1 serialized",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    precisions = tuple(dict.fromkeys(args.precisions))
    if "fp8" in precisions and args.calibration is None:
        raise ValueError("--calibration is required when --precisions includes fp8")
    if "fp8" not in precisions and args.calibration is not None:
        raise ValueError("--calibration is only valid when --precisions includes fp8")
    if args.episodes_per_task <= 0:
        raise ValueError("--episodes-per-task must be positive")
    for path in (
        args.python,
        args.source_dir,
        args.checkpoint,
        args.backbone,
        args.tactics,
    ):
        if not path.exists():
            raise ValueError(f"required path does not exist: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Path(__file__).resolve().with_name("eval_gr00t_n1d7_libero.py")
    auditor = Path(__file__).resolve().with_name("audit_gr00t_n1d7_libero_campaign.py")
    if args.calibration is not None and not args.calibration.exists():
        raise ValueError(f"required path does not exist: {args.calibration}")
    source_git = git_revision(args.source_dir)
    apxinf_source_git = git_revision(Path(__file__).resolve().parents[1])
    if source_git["dirty"]:
        raise ValueError("Isaac-GR00T source checkout must be clean for a formal campaign")
    if apxinf_source_git["dirty"]:
        raise ValueError("ApxInf source checkout must be clean for a formal campaign")
    manifest = {
        "schema": "apxinf.gr00t-n1.7.libero-campaign-manifest.v1",
        "tasks": list(LIBERO_10_ENVS),
        "views": 2,
        "video_keys": ["image", "wrist_image"],
        "precisions": list(precisions),
        "episodes_per_task": args.episodes_per_task,
        "episodes_per_precision": args.episodes_per_task * len(LIBERO_10_ENVS),
        "seed": args.seed,
        "max_episode_steps": args.max_episode_steps,
        "n_action_steps": args.n_action_steps,
        "n_envs": min(args.n_envs, args.episodes_per_task),
        "noise_mode": "stream",
        "source_git": source_git,
        "apxinf_source_git": apxinf_source_git,
        "paths": {
            # Preserve virtual-environment interpreter symlinks. Path.resolve()
            # can collapse them to /usr/bin/python and silently drop the venv.
            "python": str(args.python.absolute()),
            "source_dir": str(args.source_dir.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "backbone": str(args.backbone.resolve()),
            "calibration": (
                str(args.calibration.resolve()) if args.calibration is not None else None
            ),
            "calibration_sha256": (
                sha256(args.calibration) if args.calibration is not None else None
            ),
            "tactics": str(args.tactics.resolve()),
            "tactics_sha256": sha256(args.tactics),
            "campaign_runner_sha256": sha256(Path(__file__).resolve()),
            "evaluator_sha256": sha256(evaluator),
            "auditor_sha256": sha256(auditor),
        },
    }
    manifest_path = args.output_dir / "campaign-manifest.json"
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(encoded)
    temporary.replace(manifest_path)

    for task_id, env_name in enumerate(LIBERO_10_ENVS):
        for precision in precisions:
            output = args.output_dir / f"task{task_id}-{precision}-seed{args.seed}.json"
            log = output.with_suffix(".log")
            if args.resume and output.exists():
                document = json.loads(output.read_text())
                if (
                    document.get("episodes") == args.episodes_per_task
                    and document.get("precision") == precision
                    and document.get("views") == 2
                    and document.get("env_name") == env_name
                    and document.get("seed") == args.seed
                ):
                    print(f"skip complete receipt: {output}", flush=True)
                    continue
                raise RuntimeError(f"refusing to resume from incompatible receipt: {output}")

            command = [
                str(args.python.absolute()),
                str(evaluator),
                "--source-dir",
                str(args.source_dir.resolve()),
                "--checkpoint",
                str(args.checkpoint.resolve()),
                "--backbone",
                str(args.backbone.resolve()),
                "--precision",
                precision,
                "--tactics",
                str(args.tactics.resolve()),
                "--env-name",
                env_name,
                "--views",
                "2",
                "--episodes",
                str(args.episodes_per_task),
                "--max-episode-steps",
                str(args.max_episode_steps),
                "--n-action-steps",
                str(args.n_action_steps),
                "--n-envs",
                str(min(args.n_envs, args.episodes_per_task)),
                "--seed",
                str(args.seed),
                "--noise-mode",
                "stream",
                "--record-first-normalized-action",
                "--output",
                str(output),
            ]
            if precision == "fp8":
                assert args.calibration is not None
                command.extend(["--calibration", str(args.calibration.resolve())])
            print(f"run task={task_id} precision={precision}", flush=True)
            with log.open("w", encoding="utf-8") as stream:
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)

    audit_command = [
            sys.executable,
            str(auditor),
            "--input-dir",
            str(args.output_dir),
            "--views",
            "2",
            "--expected-tasks",
            str(len(LIBERO_10_ENVS)),
            "--expected-seeds",
            "1",
            "--expected-episodes-per-precision",
            str(args.episodes_per_task * len(LIBERO_10_ENVS)),
            "--precisions",
            *precisions,
            "--require-provenance",
            "--campaign-manifest",
            str(manifest_path),
            "--output",
            str(args.output_dir / "audit.json"),
        ]
    if "bf16" in precisions and len(precisions) == 2:
        audit_command.append("--require-parity")
    subprocess.run(audit_command, check=True)


if __name__ == "__main__":
    main()
