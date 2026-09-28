# GR00T N1.7

ApxInf serves the NVIDIA GR00T N1.7 model core through the same public path as
the other VLA families:

```text
AutoPolicy -> Gr00tPolicy -> apxinf_py.ModelRunner -> AutoModel -> VlaRuntime
```

`Gr00tPolicy` deliberately keeps NVIDIA's official checkpoint processor around
the native model core. The processor owns raw RGB/state/prompt conversion,
normalization, embodiment selection, token and patch-grid construction, and
action decode. Rust owns validated tensor contracts, weights, CUDA execution,
graph lifetime, and the flow schedule.

## Supported deployments

| Platform | Precision |
| --- | --- |
| Jetson AGX Thor (SM110) | BF16, calibrated FP8 |
| Jetson AGX Orin (SM87) | BF16, INT8/W8A8 |

`precision="auto"` selects BF16 on both platforms. Unsupported combinations
fail at load time; they never silently fall back to another precision.

The loader selects the precision exactly once and constructs a concrete
runtime/executor pair:

```text
BF16 -> bf16_runtime -> bf16_executor -> BF16 device weights
FP8  -> fp8_runtime  -> fp8_executor  -> calibrated FP8 device weights
INT8 -> int8_runtime -> int8_executor -> mixed BF16/W8A8 typed device weights
```

The three executors reuse a generic GR00T topology, but Rust monomorphizes that
topology for their distinct weight types. There is no per-linear precision
enum or BF16/FP8/INT8 dispatch in the inference hot path. The INT8 execution
plan explicitly keeps attention, adapters, encoders, and decoders in BF16 and
uses W8A8 only for the validated FFN and eligible fused self-QKV matrices.

## Loading

**Supply both `model_dir` and `backbone` as local paths**, including when using
`AutoPolicy`. The GR00T checkpoint already packages the backbone and action-head
weights together; all inference weight tensors are loaded from `model_dir`.
The separate `backbone` argument supplies the compatible Cosmos-Reason2-2B
architecture configuration and official processor resources.

| Argument | Required local assets |
| --- | --- |
| `model_dir` | GR00T `config.json` and SafeTensors weights (including the index and referenced shards for a sharded checkpoint); NVIDIA `processor_config.json`, `statistics.json`, and `embodiment_id.json`. The processor files may be at the checkpoint root or together in its `processor/` subdirectory. |
| `backbone` | A compatible Cosmos-Reason2-2B asset directory containing `config.json` and its Qwen3-VL tokenizer, chat template, image processor, and video processor resources. Preserve the snapshot's processor/tokenizer files and any files they reference. |

The Cosmos resources commonly include `tokenizer.json`, `tokenizer_config.json`,
`chat_template.json`, `preprocessor_config.json`, and
`video_preprocessor_config.json`, together with the tokenizer's vocabulary and
merges files. Rust reads the Cosmos architecture config; NVIDIA's processor
loads its own resources from the same directory. Install the compatible
Isaac-GR00T and Transformers environment before loading a policy. Processor
loading uses `local_files_only=True`.

```python
from apxinf import AutoPolicy

policy = AutoPolicy.from_pretrained(
    "/models/GR00T-N1.7-LIBERO/libero_10",
    backbone="/models/nvidia/Cosmos-Reason2-2B",
    precision="bf16",
)

result = policy.infer({
    "observation/image": base_rgb,
    "observation/wrist_image": wrist_rgb,
    "observation/state": {
        "x": eef_position[0:1],
        "y": eef_position[1:2],
        "z": eef_position[2:3],
        "roll": eef_axis_angle[0:1],
        "pitch": eef_axis_angle[1:2],
        "yaw": eef_axis_angle[2:3],
        "gripper": gripper_qpos,  # both mirrored finger joints
    },
    "prompt": "put the moka pot on the stove",
})
actions = result["actions"]
```

The Python policy forwards `backbone` as `ModelRunner.load(...,
assets={"backbone": ...})`; native Rust callers set
`LoadOptions.assets["backbone"]`. ApxInf does not resolve this local directory
from an environment variable or checkpoint metadata, and does not download it.
The checkpoint's `model_name` may name a Hub repository or a path from its
training/export machine; it does not establish an available, compatible local
snapshot. Omitting `backbone=` therefore raises an error.

BF16 and INT8 inference do not read the Cosmos SafeTensors files. **Current FP8
calibration identity validation additionally hashes the weight shards in both
supplied directories.** For FP8, retain the complete Cosmos snapshot used to
produce the calibration JSON, including those shards. They participate in the
identity check; the model's inference weights still come from the GR00T
checkpoint. Pass the calibration JSON explicitly with `calibration=` and a
device-specific tactics database with `tactics=` when needed.

The generic command-line example accepts the same second path through its
existing policy options:

```bash
python python/apxinf/examples/autopolicy_infer.py \
  --model-dir /models/GR00T-N1.7-LIBERO/libero_10 --precision bf16 \
  --policy-options '{"backbone":"/models/nvidia/Cosmos-Reason2-2B"}'
```

For LIBERO, GR00T's official state contract has eight values: XYZ, axis-angle
rotation, and both mirrored gripper joint positions. Do not pass the seven-value
OpenPI state convention, which intentionally keeps only one gripper value.
The shared evaluator also applies NVIDIA's official decoded-gripper conversion
(`0=closed, 1=open` in the dataset to `+1=closed, -1=open` in robosuite) only
for the GR00T in-process backend; existing policy and websocket conventions are
unchanged.

Importing `apxinf` does not import Torch, Transformers, Isaac-GR00T, or the CUDA
binding. Those optional dependencies are loaded lazily by
`AutoPolicy.from_pretrained` when it dispatches to GR00T, or by the equivalent
model-specific `Gr00tPolicy.from_pretrained` entry point.

## FP8 calibration

GR00T uses the project-wide `CalibrationRunner`; it does not define a private
calibration format or scan module names. `Gr00tPolicy.fp8_execution_plan()`
publishes the actual quantized linear consumers, `calibration_plan()` maps each
consumer to its stable `<consumer>.input` capture site, and
`collect_calibration(observation, context)` runs the normal official processor
before collecting BF16 activation maxima. A calibration job therefore follows
the same model-neutral workflow as a new VLA:

```python
from apxinf import CalibrationRunner, Gr00tPolicy

plan = policy.calibration_plan()
profile = CalibrationRunner(
    policy,
    plan,
    checkpoint=Gr00tPolicy.checkpoint_identity(checkpoint, backbone),
    data_identity=representative_dataset_identity,
    source_revision=source_revision,
    device={"requested": "cuda:0", "host": host_identity},
    margin=1.1,
    seed=0,
).run(public_observations)
```

The current identity hashes the weight shards in the combined GR00T checkpoint
and the supplied Cosmos snapshot, as described under [Loading](#loading).
The emitted artifact uses
`apxinf.fp8-calibration.v1`. Runtime loading rejects
the wrong model family, checkpoint identity, scale formula, missing or unknown
consumer/site, incomplete provenance, and non-production data labels. The
profile is an external deployment artifact passed through `calibration=`; it is
not copied into the read-only checkpoint and is not committed to this tree.

## Performance optimizations

- Optimize Thor BF16/FP8 and Orin BF16/W8A8 inference.
- Fuse quantization, normalization, RoPE, attention, and projection epilogues to reduce kernel launches and intermediate tensors.
- GR00T explicitly selects optimized operations; shared default paths and interfaces remain compatible.

## Best measured performance

| Device | Precision | 1-view P50 | 2-view P50 |
| --- | --- | ---: | ---: |
| Thor | BF16 | 51.834 ms | 54.216 ms |
| Thor | FP8 | 32.557 ms | 35.436 ms |
| Orin | BF16 | 75.778 ms | 84.864 ms |
| Orin | W8A8 | 56.711 ms | 64.924 ms |

## LIBERO-10 task accuracy

| Device | Precision | Episodes | Successes | Success rate |
| --- | --- | ---: | ---: | ---: |
| Thor | BF16 | 100 | 94 | 94.0% |
| Thor | FP8 | 100 | 92 | 92.0% |
| Orin | BF16 | 100 | 93 | 93.0% |
| Orin | W8A8 | 100 | 93 | 93.0% |

Two views, 10 episodes for each of the 10 tasks, a 720-step limit, and eight
executed actions per chunk.

## Fixed-input benchmark

The maintained runner uses tensors produced by the official NVIDIA processor.
Its timing boundary starts at those host tensors and ends after model-core
action D2H; it excludes simulator and raw-observation preprocessing time.

```bash
python scripts/bench_gr00t.py \
  --checkpoint /models/GR00T-N1.7-LIBERO/libero_10 \
  --backbone /models/nvidia/Cosmos-Reason2-2B \
  --fixture devlocal/gr00t-n1d7/fixtures/libero-two-view \
  --precision bf16 \
  --warmup 10 \
  --iterations 50
```

Use `--calibration` for FP8, `--tactics` for an explicit tactic database, and
`--reference` for a numerical parity gate. When the installed CUDA/cuBLAS or
kernel implementation version does not match a stored database, pass an
explicit new `--tactics` path together with `--autotune`; never rewrite the
bundled database merely to bypass provenance validation. The report records
P50/P95 and the complete normalized model-core output. Generated fixtures,
reference dumps, logs, tactic databases, and result JSON belong under
`devlocal/gr00t-n1d7/` and are not committed.

The repository's shared LIBERO evaluator selects the GR00T state adapter while
leaving the existing OpenPI state wire format unchanged for other policies:

```bash
python scripts/eval_libero.py \
  --backend in-process \
  --model-dir /models/GR00T-N1.7-LIBERO/libero_10 \
  --backbone /models/nvidia/Cosmos-Reason2-2B \
  --precision bf16 \
  --suite libero_10 \
  --trials-per-task 10 \
  --max-steps 720 \
  --replan-steps 8 \
  --results-jsonl devlocal/gr00t-n1d7/results/libero-bf16.jsonl \
  --summary-json devlocal/gr00t-n1d7/results/libero-bf16-summary.json
```

## Validation contract

Correctness comparison uses the same official processor output, embodiment ID,
and initial noise on both implementations. The native model-core output is
`[40, 132]`; the processor decodes and trims the LIBERO action to `[16, 7]`.
The closed-loop evaluator executes eight actions before replanning.

Minimum release gates are:

- BF16: cosine similarity at least `0.999`, relative L2 at most `0.05`.
- FP8: cosine similarity at least `0.997`, relative L2 at most `0.10`.
- INT8: cosine similarity at least `0.995`, relative L2 at most `0.10`.
- Every output must be finite and have the exact expected shape.

One-view fixtures are used only for fixed-input numerical accuracy and
performance; they are never used for LIBERO closed-loop task evaluation. The
two-view release campaign uses 10 episodes for each of the 10 LIBERO-10 tasks
and each supported platform/precision pair. GR00T LIBERO rollouts explicitly
use the NVIDIA N1.7 evaluation protocol of 720 maximum simulator steps and 8
executed actions per predicted chunk; the evaluator's 520/5 defaults remain
unchanged for existing PI0.5 and WallOSS callers.

PI0.5 regression acceptance uses a same-host upstream-versus-candidate A/B
with the same checkpoint, evaluator, trials, frozen noise, and calibration.
Unmatched historical aggregate success rates are useful context, but are not
used to attribute a regression to this change.
