//! End-to-end CUDA smoke test for the released NVIDIA GR00T N1.7 checkpoint.
//!
//! The input is deliberately synthetic and minimal. Together with
//! `scripts/gr00t_n1d7_reference_dump.py`, it forms a deterministic numerical
//! parity fixture at the already-preprocessed model boundary.

use std::path::{Path, PathBuf};

use apxinf_core::{Backend, DType, Device, Tensor};
use apxinf_model::gr00t::{
    load_backbone_config, Gr00tConfig, Gr00tLoadOptions, Gr00tObservation, Gr00tVlaRuntime,
};
use apxinf_model::ModelPrecision;
use half::bf16;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = std::env::args().collect::<Vec<_>>();
    if !(3..=6).contains(&arguments.len()) {
        return Err(format!(
            "usage: {} <gr00t-checkpoint-dir> <cosmos-backbone-dir-or-config> [cuda-device] [output-json] [graph|graph-update|eager|eager-update]",
            arguments
                .first()
                .map(String::as_str)
                .unwrap_or("gr00t_synthetic_smoke")
        )
        .into());
    }
    let checkpoint = Path::new(&arguments[1]);
    let backbone_path = PathBuf::from(&arguments[2]);
    let device_id = arguments
        .get(3)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(0);
    let output_path = arguments.get(4).map(Path::new);
    let execution = arguments.get(5).map(String::as_str).unwrap_or("graph");
    if !matches!(
        execution,
        "graph" | "graph-update" | "eager" | "eager-update"
    ) {
        return Err(format!("invalid execution mode {execution:?}").into());
    }

    let config = Gr00tConfig::from_json_file(&checkpoint.join("config.json"))?;
    let backbone = load_backbone_config(&config, &backbone_path)?;
    let patch_width = backbone
        .vision
        .in_channels
        .checked_mul(backbone.vision.temporal_patch_size)
        .and_then(|value| value.checked_mul(backbone.vision.patch_size))
        .and_then(|value| value.checked_mul(backbone.vision.patch_size))
        .ok_or("GR00T synthetic pixel width overflow")?;
    let image_token_id = backbone.image_token_id;
    let state_history_length = config.state_history_length;
    let max_state_dim = config.max_state_dim;
    let action_horizon = config.action_horizon;
    let max_action_dim = config.max_action_dim;

    // Two 2x2 camera grids become two visual tokens after Qwen's 2x2 merger.
    // Using two grids also exercises the multi-view segmented-attention path.
    let mut observation = Gr00tObservation {
        pixel_values: Tensor::zeros(vec![8, patch_width], DType::BF16),
        image_grid_thw: vec![[1, 2, 2], [1, 2, 2]],
        token_ids: vec![1, image_token_id, image_token_id, 2],
        attention_mask: vec![1, 1, 1, 1],
        state: Tensor::zeros(
            vec![1, config.state_history_length, config.max_state_dim],
            DType::BF16,
        ),
        embodiment_id: 0,
        noise: Tensor::zeros(
            vec![1, config.action_horizon, config.max_action_dim],
            DType::BF16,
        ),
    };
    if matches!(execution, "graph-update" | "eager-update") {
        observation = patterned_observation(observation, image_token_id)?;
    }
    let fixture = if matches!(execution, "graph-update" | "eager-update") {
        "two-view-patterned-boundary-input-v1"
    } else {
        "two-view-zero-boundary-input-v1"
    };
    let eager_backend = if matches!(execution, "eager" | "eager-update") {
        let backend = apxinf_cuda::CudaBackend::new(device_id)?;
        observation.pixel_values = backend.to_device(&observation.pixel_values)?;
        observation.state = backend.to_device(&observation.state)?;
        observation.noise = backend.to_device(&observation.noise)?;
        Some(backend)
    } else {
        None
    };
    let options = Gr00tLoadOptions {
        config: Some(config),
        precision: ModelPrecision::Bf16,
        backbone_path: Some(backbone_path),
        fp8_calibration_path: None,
        tuning_path: std::env::var_os("APXINF_GR00T_BF16_TACTICS").map(PathBuf::from),
    };

    let mut runtime = Gr00tVlaRuntime::from_dir(checkpoint, options, Device::Cuda(device_id))?;
    if execution == "graph-update" {
        let zero_observation = Gr00tObservation {
            pixel_values: Tensor::zeros(vec![8, patch_width], DType::BF16),
            image_grid_thw: vec![[1, 2, 2], [1, 2, 2]],
            token_ids: vec![1, image_token_id, image_token_id, 2],
            attention_mask: vec![1, 1, 1, 1],
            state: Tensor::zeros(vec![1, state_history_length, max_state_dim], DType::BF16),
            embodiment_id: 0,
            noise: Tensor::zeros(vec![1, action_horizon, max_action_dim], DType::BF16),
        };
        let _ = runtime.infer(&zero_observation)?;
    }
    let output = runtime.infer(&observation)?;
    drop(eager_backend);
    let values = output.to_f32_vec()?;
    let non_finite = values.iter().filter(|value| !value.is_finite()).count();
    if non_finite != 0 {
        return Err(format!("GR00T output contains {non_finite} non-finite values").into());
    }
    let sum = values.iter().map(|value| f64::from(*value)).sum::<f64>();
    let max_abs = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let report = serde_json::json!({
        "schema": "apxinf.gr00t-n1.7.synthetic-smoke.v1",
        "fixture": fixture,
        "execution": execution,
        "device": device_id,
        "output_shape": output.shape().dims(),
        "output_dtype": output.dtype().to_string(),
        "non_finite": non_finite,
        "sum": sum,
        "max_abs": max_abs,
        "output": values,
    });
    let rendered = serde_json::to_string_pretty(&report)?;
    if let Some(path) = output_path {
        std::fs::write(path, format!("{rendered}\n"))?;
        println!("wrote {}", path.display());
    } else {
        println!("{rendered}");
    }
    Ok(())
}

fn patterned_observation(
    mut observation: Gr00tObservation,
    image_token_id: u32,
) -> Result<Gr00tObservation, Box<dyn std::error::Error>> {
    let pixel_shape = observation.pixel_values.shape().clone();
    let pixel_values = (0..observation.pixel_values.numel())
        .map(|index| bf16::from_f32(((index % 31) as f32 - 15.0) / 128.0))
        .collect::<Vec<_>>();
    observation.pixel_values = Tensor::from_bf16(pixel_shape, &pixel_values)?;

    let state_shape = observation.state.shape().clone();
    let state = (0..observation.state.numel())
        .map(|index| bf16::from_f32(((index % 17) as f32 - 8.0) / 32.0))
        .collect::<Vec<_>>();
    observation.state = Tensor::from_bf16(state_shape, &state)?;

    let noise_shape = observation.noise.shape().clone();
    let noise = (0..observation.noise.numel())
        .map(|index| bf16::from_f32((((index * 7) % 29) as f32 - 14.0) / 32.0))
        .collect::<Vec<_>>();
    observation.noise = Tensor::from_bf16(noise_shape, &noise)?;

    for (index, token) in observation.token_ids.iter_mut().enumerate() {
        if *token != image_token_id {
            *token = 3 + index as u32;
        }
    }
    Ok(observation)
}
