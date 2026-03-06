#!/usr/bin/env python3

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_squared_error

from heal_swin.data import data
from heal_swin.evaluation.custom_metrics import get_non_inf_non_nan_idxs
from heal_swin.evaluation.evaluate_config import EvaluateConfig
from heal_swin.models_lightning import models_lightning
from heal_swin.training.train_config import PLConfig
from heal_swin.utils import depth_utils, get_paths, serialize, utils


def calculate_metrics(label: np.ndarray, pred: np.ndarray, epsilon: float = 1e-6) -> dict:
    label = np.asarray(label, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    rmse = float(np.sqrt(mean_squared_error(label, pred)))

    label = label + epsilon
    pred = pred + epsilon

    abs_rel = float(np.mean(np.abs(label - pred) / label))
    sq_rel = float(np.mean(((label - pred) ** 2) / label))

    ratio = np.maximum(label / pred, pred / label)
    delta1 = float(np.mean(ratio < 1.25))
    delta2 = float(np.mean(ratio < 1.25**2))
    delta3 = float(np.mean(ratio < 1.25**3))

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
    }


def _extract_sample_id(dataset, idx: int) -> str:
    if hasattr(dataset, "paths"):
        name = os.path.basename(dataset.paths[idx])
        match = re.search(r"pos(\d+)\.npz", name)
        if match:
            return match.group(1)
    if hasattr(dataset, "depth_masks_dataset") and hasattr(dataset.depth_masks_dataset, "file_names"):
        file_name = dataset.depth_masks_dataset.file_names[idx]
        match = re.search(r"(\d+)", file_name)
        if match:
            return match.group(1)
    return str(idx)


def _prepare_input(sample, model_config_name: str):
    if not isinstance(sample, (tuple, list)) or len(sample) < 2:
        raise ValueError("Dataset sample must be tuple/list with at least (image, depth).")

    img = torch.as_tensor(sample[0])
    depth = torch.as_tensor(sample[1])

    if model_config_name == "WoodscapeDepthSwinHPSobelConfig":
        if len(sample) < 3:
            raise ValueError("Sobel model expects (image, depth, sobel) samples.")
        sobel = torch.as_tensor(sample[2])
        if sobel.ndim == img.ndim - 1:
            sobel = sobel.unsqueeze(0)
        img = torch.cat([img, sobel], dim=0)

    return img, depth


def _apply_flat_circle_mask(idxs: torch.Tensor, prediction: torch.Tensor, model_config_name: str):
    if model_config_name != "WoodscapeDepthSwinConfig":
        return idxs
    if prediction.ndim != 2:
        return idxs

    h, w = prediction.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=prediction.device),
        torch.arange(w, device=prediction.device),
    )
    radius = min(h, w) // 2
    mask = (yy - h // 2) ** 2 + (xx - w // 2) ** 2 <= radius**2
    return idxs & mask


def evaluate_model(eval_config: EvaluateConfig, pl_config: PLConfig, config_path: str):
    del pl_config

    ckpt_path, artifact_path, _ = utils.check_and_get_ckpt_paths(
        run_identifier=eval_config.path,
        epoch=eval_config.epoch,
        epoch_number=eval_config.epoch_number,
    )

    artifact_path = Path(artifact_path)
    run_id = artifact_path.parent.name

    serialize.save(eval_config, artifact_path / eval_config.eval_config_name)

    abs_config_path = get_paths.get_abs_path_from_config_path(config_path)
    config_name = os.path.basename(abs_config_path)
    if config_name == "slurm_script":
        config_name = f"eval_config_{eval_config.eval_config_name}.py"
    shutil.copyfile(abs_config_path, artifact_path / config_name)

    model_config = serialize.load(artifact_path / "model_config")
    model_config_name = type(model_config).__name__

    datamodule, data_spec = data.get_data_module(eval_config.data_config)
    model_cls = models_lightning.MODEL_FROM_NAME[
        models_lightning.MODEL_NAME_FROM_CONFIG_NAME[model_config_name]
    ]

    print("-" * 40)
    print(f"Model class: {model_cls.__name__}")
    print(f"Checkpoint: {ckpt_path}")
    print("-" * 40)

    model = model_cls.load_from_checkpoint(
        ckpt_path,
        config=model_config,
        data_spec=data_spec,
        data_config=eval_config.data_config,
        strict=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    val_loader = datamodule.val_dataloader()
    dataset = val_loader.dataset
    val_size = len(dataset)
    print(f"Validation samples: {val_size}")

    metrics_by_sample = {}
    totals = {
        "mse": 0.0,
        "abs_rel": 0.0,
        "sq_rel": 0.0,
        "rmse": 0.0,
        "delta1": 0.0,
        "delta2": 0.0,
        "delta3": 0.0,
    }

    with torch.no_grad():
        for idx in range(val_size):
            sample_id = _extract_sample_id(dataset, idx)
            sample = dataset[idx]

            image, depth = _prepare_input(sample, model_config_name)
            pred = model(image.unsqueeze(0).to(device)).squeeze(0)

            depth = depth_utils.unnormalize_and_retransform(
                data=depth.to(device),
                normalization=model.normalize_data,
                data_stats=model.depth_data_statistics,
                data_transform=model.data_transform,
            )

            pred = pred.squeeze()
            depth = depth.squeeze()

            idxs = get_non_inf_non_nan_idxs(pred, depth)
            idxs = _apply_flat_circle_mask(idxs, pred, model_config_name)

            pred_np = pred[idxs].detach().cpu().numpy()
            depth_np = depth[idxs].detach().cpu().numpy()

            if pred_np.size == 0:
                continue

            metrics = calculate_metrics(depth_np, pred_np)
            metrics["mse"] = float(mean_squared_error(depth_np, pred_np))
            metrics_by_sample[sample_id] = metrics

            for key in totals:
                totals[key] += metrics[key]

            print(
                f"{idx:04d} sample={sample_id} mse={metrics['mse']:.4f} "
                f"abs_rel={metrics['abs_rel']:.4f} sq_rel={metrics['sq_rel']:.4f} "
                f"rmse={metrics['rmse']:.4f} d1={metrics['delta1']:.4f} "
                f"d2={metrics['delta2']:.4f} d3={metrics['delta3']:.4f}"
            )

    sample_count = max(len(metrics_by_sample), 1)
    average = {key: float(value / sample_count) for key, value in totals.items()}
    metrics_by_sample["average"] = average

    metrics_dir = Path(__file__).resolve().parent / "runtime_assets" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_path = metrics_dir / f"{run_id}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_by_sample, f, indent=2)

    print(f"Saved metrics to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="heal_swin/run_configs/depth_estimation/evaluate_all_depth_config.py",
        help="Path to evaluation config file.",
    )
    args = parser.parse_args()

    eval_config = utils.get_config_from_config_path(args.config_path, "get_eval_run_config")
    pl_config = utils.get_config_from_config_path(args.config_path, "get_pl_config")
    evaluate_model(eval_config, pl_config, args.config_path)


if __name__ == "__main__":
    main()
