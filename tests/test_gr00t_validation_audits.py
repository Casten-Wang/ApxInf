"""Unit tests for GR00T parity and closed-loop campaign audit helpers."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


def load_script(name: str):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


campaign = load_script("audit_gr00t_n1d7_libero_campaign.py")


class Gr00tCampaignAuditTests(unittest.TestCase):
    @staticmethod
    def valid_receipt() -> dict:
        digest = "0" * 64
        signature = {"dtype": "<f4", "shape": [1], "sha256": digest}
        return {
            "schema": campaign.SCHEMA,
            "precision": "fp8",
            "views": 2,
            "video_keys": ["image", "wrist_image"],
            "env_name": "libero_sim/test",
            "episodes": 1,
            "successes": 1,
            "success_rate": 1.0,
            "episode_successes": [True],
            "model_calls": 1,
            "model_seconds": 0.1,
            "seed": 7,
            "runtime_contract": {
                "action_horizon": 40,
                "action_dim": 132,
                "n_action_steps": 8,
                "max_episode_steps": 720,
                "n_envs": 1,
                "noise_mode": "stream",
                "hf_hub_offline": "1",
                "mujoco_gl": "egl",
                "pyopengl_platform": "egl",
            },
            "software": {
                "python": "3.12",
                "numpy": "2.0",
                "torch": "2.8",
                "torch_cuda": "13.0",
            },
            "artifacts": {
                "apxinf_source_git": {"revision": "2" * 40, "dirty": False},
                "source_dir": "/source",
                "source_git": {"revision": "1" * 40, "dirty": False},
                "checkpoint": "/checkpoint",
                "checkpoint_config": {"path": "/config", "sha256": digest},
                "backbone": "/backbone",
                "backbone_config": {"path": "/backbone/config", "sha256": digest},
                "calibration": {"path": "/calibration", "sha256": digest},
                "tactics": {"path": "/tactics", "sha256": digest},
                "apxinf_python_extension": {"path": "/apxinf.so", "sha256": digest},
            },
            "first_normalized_action": [0.0] * (40 * 132),
            "first_normalized_action_shape": [1, 40, 132],
            "first_model_input_signatures": {
                "pixel_values": signature,
                "image_grid_thw": signature,
                "token_ids": signature,
                "attention_mask": signature,
                "state": signature,
                "noise": signature,
                "embodiment_id": 2,
            },
        }

    def test_identical_parity(self) -> None:
        metrics = campaign.parity_metrics([1.0, -2.0, 0.5], [1.0, -2.0, 0.5])
        self.assertAlmostEqual(metrics["cosine"], 1.0)
        self.assertEqual(metrics["relative_l2"], 0.0)
        self.assertEqual(metrics["max_abs"], 0.0)

    def test_historical_w8a8_receipts_use_public_int8_name(self) -> None:
        self.assertEqual(campaign.canonical_precision("w8a8"), "int8")
        self.assertEqual(campaign.canonical_precision("int8"), "int8")

    def test_zero_vectors_match(self) -> None:
        metrics = campaign.parity_metrics([0.0, 0.0], [0.0, 0.0])
        self.assertEqual(metrics["cosine"], 1.0)
        self.assertEqual(metrics["relative_l2"], 0.0)

    def test_one_zero_vector_fails_relative_metrics(self) -> None:
        metrics = campaign.parity_metrics([0.0, 0.0], [1.0, 0.0])
        self.assertEqual(metrics["cosine"], 0.0)
        self.assertEqual(metrics["relative_l2"], float("inf"))

    def test_parity_rejects_different_lengths(self) -> None:
        with self.assertRaises(ValueError):
            campaign.parity_metrics([1.0], [1.0, 2.0])

    def test_strict_receipt_accepts_complete_provenance(self) -> None:
        campaign.require_receipt(
            Path("receipt.json"), self.valid_receipt(), 2, True, True
        )

    def test_strict_receipt_rejects_wrong_camera_order(self) -> None:
        receipt = self.valid_receipt()
        receipt["video_keys"] = ["wrist_image", "image"]
        with self.assertRaises(ValueError):
            campaign.require_receipt(Path("receipt.json"), receipt, 2, True, True)

    def test_strict_receipt_rejects_missing_input_hash(self) -> None:
        receipt = self.valid_receipt()
        del receipt["first_model_input_signatures"]["noise"]
        with self.assertRaises(ValueError):
            campaign.require_receipt(Path("receipt.json"), receipt, 2, True, True)

    def test_strict_receipt_rejects_fixed_noise(self) -> None:
        receipt = self.valid_receipt()
        receipt["runtime_contract"]["noise_mode"] = "fixed"
        with self.assertRaisesRegex(ValueError, "stream noise"):
            campaign.require_receipt(Path("receipt.json"), receipt, 2, True, True)

    def test_strict_campaign_accepts_one_complete_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for precision in ("bf16", "fp8"):
                receipt = copy.deepcopy(self.valid_receipt())
                receipt["precision"] = precision
                if precision == "bf16":
                    receipt["artifacts"]["calibration"] = None
                path = directory / f"task0-{precision}-seed7.json"
                path.write_text(json.dumps(receipt))
            output = directory / "audit.json"
            script_dir = Path(campaign.__file__).parent
            manifest = {
                "schema": "apxinf.gr00t-n1.7.libero-campaign-manifest.v1",
                "tasks": ["libero_sim/test"],
                "views": 2,
                "precisions": ["bf16", "fp8"],
                "seed": 7,
                "episodes_per_task": 1,
                "episodes_per_precision": 1,
                "n_envs": 1,
                "n_action_steps": 8,
                "max_episode_steps": 720,
                "noise_mode": "stream",
                "source_git": {"revision": "1" * 40, "dirty": False},
                "apxinf_source_git": {"revision": "2" * 40, "dirty": False},
                "paths": {
                    "source_dir": "/source",
                    "checkpoint": "/checkpoint",
                    "backbone": "/backbone",
                    "calibration_sha256": "0" * 64,
                    "tactics_sha256": "0" * 64,
                    "evaluator_sha256": campaign.sha256(
                        script_dir / "eval_gr00t_n1d7_libero.py"
                    ),
                    "auditor_sha256": campaign.sha256(Path(campaign.__file__)),
                    "campaign_runner_sha256": campaign.sha256(
                        script_dir / "run_gr00t_n1d7_libero_campaign.py"
                    ),
                },
            }
            manifest_path = directory / "campaign-manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(campaign.__file__)),
                    "--input-dir",
                    str(directory),
                    "--views",
                    "2",
                    "--expected-tasks",
                    "1",
                    "--expected-seeds",
                    "1",
                    "--expected-episodes-per-precision",
                    "1",
                    "--require-provenance",
                    "--require-parity",
                    "--campaign-manifest",
                    str(manifest_path),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(json.loads(output.read_text())["parity"]["passed"])

    def test_strict_campaign_rejects_input_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for precision in ("bf16", "fp8"):
                receipt = copy.deepcopy(self.valid_receipt())
                receipt["precision"] = precision
                if precision == "bf16":
                    receipt["artifacts"]["calibration"] = None
                else:
                    receipt["first_model_input_signatures"]["noise"]["sha256"] = "f" * 64
                path = directory / f"task0-{precision}-seed7.json"
                path.write_text(json.dumps(receipt))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(campaign.__file__)),
                    "--input-dir",
                    str(directory),
                    "--views",
                    "2",
                    "--expected-tasks",
                    "1",
                    "--expected-seeds",
                    "1",
                    "--expected-episodes-per-precision",
                    "1",
                    "--require-provenance",
                    "--require-parity",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("model inputs differ", completed.stderr)

    def test_strict_receipt_rejects_dirty_apxinf_source(self) -> None:
        receipt = self.valid_receipt()
        receipt["artifacts"]["apxinf_source_git"]["dirty"] = True
        with self.assertRaisesRegex(ValueError, "ApxInf source checkout is dirty"):
            campaign.require_receipt(Path("receipt.json"), receipt, 2, True, True)


if __name__ == "__main__":
    unittest.main()
