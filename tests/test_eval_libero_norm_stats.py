"""Exercise the evaluator CLI-to-checkpoint path without CUDA or a simulator."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from apxinf import AutoPolicy
from apxinf.checkpoints import detect_checkpoint
from scripts import eval_libero


def parse(monkeypatch, tmp_path, *extra, backend="in-process"):
    monkeypatch.setattr(sys, "argv", [
        "eval_libero.py", "--backend", backend, "--precision", "bf16",
        "--model-dir", str(tmp_path),
        "--results-jsonl", str(tmp_path / "results.jsonl"),
        "--summary-json", str(tmp_path / "summary.json"), *extra,
    ])
    return eval_libero.parse_args()


def test_explicit_norm_stats_reaches_checkpoint_loader(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"type": "pi05"}))
    stats = tmp_path / "external-norms.json"
    stats.write_text(json.dumps({"norm_stats": {
        "state": {"q01": [0.0] * 7, "q99": [1.0] * 7},
        "actions": {"q01": [2.0] * 7, "q99": [4.0] * 7},
    }}))
    loaded = []

    def load(model_dir, **options):
        loaded.append(detect_checkpoint(model_dir, norm_stats=options["norm_stats"]))
        return SimpleNamespace(metadata={})

    monkeypatch.setattr(AutoPolicy, "from_pretrained", load)
    args = parse(monkeypatch, tmp_path, "--norm-stats", str(stats))
    eval_libero.InProcessBackend(args, eval_libero.resolve_wire_keys(args))
    assert loaded[0].norm_stats == stats
    assert loaded[0].normalization.action.values["q01"] == (2.0,) * 7


def test_omitted_norm_stats_preserves_checkpoint_defaults(monkeypatch, tmp_path):
    options = {}

    def load(model_dir, **kwargs):
        options.update(kwargs)
        return SimpleNamespace(metadata={})

    monkeypatch.setattr(AutoPolicy, "from_pretrained", load)
    args = parse(monkeypatch, tmp_path)
    eval_libero.InProcessBackend(args, eval_libero.resolve_wire_keys(args))
    assert "norm_stats" not in options


def test_gr00t_backbone_and_two_joint_state_reach_policy(monkeypatch, tmp_path):
    options = {}

    def load(model_dir, **kwargs):
        options.update(kwargs)
        return SimpleNamespace(metadata={"model_type": "gr00t"})

    monkeypatch.setattr(AutoPolicy, "from_pretrained", load)
    backbone = tmp_path / "backbone"
    args = parse(monkeypatch, tmp_path, "--model-type", "gr00t", "--backbone", str(backbone))
    backend = eval_libero.InProcessBackend(args, eval_libero.resolve_wire_keys(args))
    state = backend.state_from_observation(
        {
            "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.array([0.04, -0.04]),
        }
    )

    assert options["backbone"] == backbone
    np.testing.assert_array_equal(
        state["gripper"], np.array([0.04, -0.04], dtype=np.float32)
    )


def test_gr00t_decoded_gripper_is_adapted_before_libero_step(monkeypatch, tmp_path):
    class FakePolicy:
        metadata = {"model_type": "gr00t"}

        def infer(self, observation, *, noise=None):
            actions = np.zeros((2, 7), dtype=np.float32)
            actions[:, -1] = [0.0, 1.0]
            return {
                "actions": actions,
                "normalized_actions": np.zeros((2, 132), dtype=np.float32),
                "timing": {},
            }

    monkeypatch.setattr(AutoPolicy, "from_pretrained", lambda *args, **kwargs: FakePolicy())
    args = parse(monkeypatch, tmp_path, "--model-type", "gr00t")
    backend = eval_libero.InProcessBackend(args, eval_libero.resolve_wire_keys(args))

    actions, _, _ = backend.infer(None, None, None, "prompt")

    np.testing.assert_array_equal(actions[:, -1], np.array([1.0, -1.0], dtype=np.float32))


def test_websocket_norm_stats_is_rejected(monkeypatch, tmp_path, capsys):
    stats = tmp_path / "norm_stats.json"
    stats.write_text("{}")
    with pytest.raises(SystemExit) as exc:
        parse(monkeypatch, tmp_path, "--norm-stats", str(stats), backend="websocket")
    assert exc.value.code == 2
    assert "pass it to pi05_openpi_websocket_server.py" in capsys.readouterr().err


def test_missing_norm_stats_is_rejected_before_rollout(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        parse(monkeypatch, tmp_path, "--norm-stats", str(tmp_path / "missing.json"))
    assert exc.value.code == 2
    assert "--norm-stats must name an existing file" in capsys.readouterr().err
