"""Train / predict / evaluate — headless port of the QuantiFly density-count pipeline.

Density regression: per-pixel shape features -> ExtraTreesRegressor -> density map whose
sum/255 is the egg count (matching v2_functions.evaluate_forest's prediction convention,
but with ground truth counted by connected-component blobs rather than a raw pixel sum).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from .features import stack_channels, FEATURE_CHANNELS
from .io import load_image, dots_to_points, build_density

MODEL_VERSION = 1


@dataclass
class TrainParams:
    feature_type: str = "fine"
    feature_scale: float = 0.8
    sigma: float = 1.0
    channels: tuple = (0, 1, 2)
    samples_per_image: int = 20000
    n_trees: int = 20
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: int = 7
    seed: int = 0


def _resolve_channels(image: np.ndarray, channels):
    """Guard grayscale (2-D) inputs (blocker #5): only channel 0 exists."""
    if image.ndim == 2:
        return (0,)
    return tuple(c for c in channels if c < image.shape[2])


def extract_training_data(pairs, p: TrainParams):
    rng = np.random.default_rng(p.seed)
    X_parts, y_parts = [], []
    for img_path, dots_path in pairs:
        image = load_image(img_path).astype(np.float32)
        chans = _resolve_channels(image, p.channels)
        feats = stack_channels(image, chans, p.feature_scale, p.feature_type)
        pts, _ = dots_to_points(load_image(dots_path))
        dens = build_density(pts, image.shape, p.sigma)
        h, w = dens.shape
        flat_f = feats.reshape(h * w, -1)
        flat_d = dens.reshape(h * w)
        k = min(p.samples_per_image, h * w)
        idx = rng.choice(h * w, size=k, replace=False)
        X_parts.append(flat_f[idx])
        y_parts.append(flat_d[idx])
    return np.concatenate(X_parts), np.concatenate(y_parts)


def train(pairs, p: TrainParams) -> dict:
    X, y = extract_training_data(pairs, p)
    rf = ExtraTreesRegressor(
        n_estimators=p.n_trees,
        max_depth=p.max_depth,
        min_samples_split=p.min_samples_split,
        min_samples_leaf=p.min_samples_leaf,
        max_features=min(p.max_features, X.shape[1]),
        bootstrap=True,
        n_jobs=-1,
        random_state=p.seed,  # blocker #6: reproducibility
    )
    rf.fit(X, y)
    return {
        "version": MODEL_VERSION,
        "rf": rf,
        "feature_type": p.feature_type,
        "feature_scale": p.feature_scale,
        "sigma": p.sigma,
        "channels": p.channels,
        "n_features": int(X.shape[1]),
    }


def predict_image(image: np.ndarray, model: dict):
    """Return (density_map, predicted_count)."""
    image = image.astype(np.float32)
    chans = _resolve_channels(image, model["channels"])
    feats = stack_channels(image, chans, model["feature_scale"], model["feature_type"])
    h, w, c = feats.shape
    pred = model["rf"].predict(feats.reshape(h * w, c)).reshape(h, w)
    pred = np.clip(pred, 0, None)  # density is non-negative
    return pred, float(pred.sum() / 255.0)


def evaluate(pairs, model: dict):
    """Return (rows, summary). Each row: name, predicted, ground_truth, abs_err, pct_err."""
    rows = []
    abs_errs = []
    for img_path, dots_path in pairs:
        image = load_image(img_path)
        _, count = predict_image(image, model)
        _, gt = dots_to_points(load_image(dots_path))
        abs_err = abs(count - gt)
        pct_err = (abs_err * 100.0 / gt) if gt else float("nan")
        rows.append({
            "name": img_path.name,
            "predicted": round(count, 2),
            "ground_truth": gt,
            "abs_err": round(abs_err, 2),
            "pct_err": round(pct_err, 1),
        })
        abs_errs.append(abs_err)
    summary = {
        "n_images": len(rows),
        "MAE": round(float(np.mean(abs_errs)), 3) if abs_errs else float("nan"),
    }
    return rows, summary
