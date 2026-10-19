"""Command-line interface: train / predict / evaluate."""
from __future__ import annotations

import csv
import pickle
from pathlib import Path

import click
import numpy as np

from .io import find_pairs, load_image
from .core import (TrainParams, train as _train, evaluate as _evaluate, predict_image,
                   MODEL_VERSION)


def _parse_channels(spec: str) -> tuple:
    return tuple(int(x) for x in spec.split(",") if x.strip() != "")


def _load_model(path) -> dict:
    with open(path, "rb") as fh:
        model = pickle.load(fh)
    if model.get("version") != MODEL_VERSION:
        raise click.ClickException(
            f"Model format version {model.get('version')} != expected {MODEL_VERSION}; "
            f"please retrain with this version of quantifly.")
    return model


@click.group()
@click.version_option(package_name="quantifly")
def main():
    """QuantiFly v2 — headless Drosophila egg counting (density estimation)."""


@main.command()
@click.option("--images", required=True, type=click.Path(exists=True, file_okay=False),
              help="Directory of image/dots pairs.")
@click.option("--out", required=True, type=click.Path(dir_okay=False),
              help="Output model.pkl path.")
@click.option("--feature", "feature_type", default="fine",
              type=click.Choice(["basic", "fine"]), show_default=True)
# NOTE: "fineSpatial" exists in features.py but is intentionally not exposed here — its
# image-index/absolute-position channels only generalize under a fixed capture geometry
# (see REVISION-01 and the fineSpatial discussion). Kept in code for a future fixed-rig variant.
@click.option("--scale", "feature_scale", default=0.8, show_default=True, type=float)
@click.option("--sigma", default=1.0, show_default=True, type=float,
              help="Gaussian sigma for the density map.")
@click.option("--channels", default="0,1,2", show_default=True,
              help="Comma-separated colour channels to use (ignored for grayscale).")
@click.option("--samples", "samples_per_image", default=20000, show_default=True, type=int)
@click.option("--trees", "n_trees", default=20, show_default=True, type=int)
@click.option("--seed", default=0, show_default=True, type=int)
def train(images, out, feature_type, feature_scale, sigma, channels, samples_per_image,
          n_trees, seed):
    """Train a model from images + ground-truth dots."""
    pairs = find_pairs(images, require_dots=True)
    if not pairs:
        raise click.ClickException(f"No image/dots pairs found in {images}")
    click.echo(f"Training on {len(pairs)} image(s)...")
    params = TrainParams(
        feature_type=feature_type, feature_scale=feature_scale, sigma=sigma,
        channels=_parse_channels(channels), samples_per_image=samples_per_image,
        n_trees=n_trees, seed=seed,
    )
    model = _train(pairs, params)
    with open(out, "wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
    click.echo(f"Saved model -> {out}  ({model['n_features']} features)")


@main.command()
@click.option("--images", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--model", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--out", required=True, type=click.Path(dir_okay=False),
              help="Output counts CSV.")
@click.option("--density-maps", type=click.Path(file_okay=False),
              help="Optional directory to write per-image density maps (float32 TIFF).")
def predict(images, model, out, density_maps):
    """Predict egg counts for images (no labels needed)."""
    mdl = _load_model(model)
    pairs = find_pairs(images, require_dots=False)
    if not pairs:
        raise click.ClickException(f"No images found in {images}")
    if density_maps:
        Path(density_maps).mkdir(parents=True, exist_ok=True)
        import tifffile
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "predicted_count"])
        for img_path, _ in pairs:
            dens, count = predict_image(load_image(img_path), mdl)
            w.writerow([img_path.name, round(count, 2)])
            click.echo(f"{img_path.name}: {count:.2f}")
            if density_maps:
                tifffile.imwrite(str(Path(density_maps) / f"{img_path.stem}_density.tif"),
                                 dens.astype(np.float32))
    click.echo(f"Wrote counts -> {out}")


@main.command()
@click.option("--images", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--model", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--out", type=click.Path(dir_okay=False), help="Optional report CSV.")
def evaluate(images, model, out):
    """Evaluate a model against ground-truth dots (reports MAE / % error)."""
    mdl = _load_model(model)
    pairs = find_pairs(images, require_dots=True)
    if not pairs:
        raise click.ClickException(f"No image/dots pairs found in {images}")
    rows, summary = _evaluate(pairs, mdl)
    hdr = ["name", "predicted", "ground_truth", "abs_err", "pct_err"]
    for r in rows:
        click.echo("  ".join(f"{r[k]}" for k in hdr))
    click.echo(f"\nImages: {summary['n_images']}   MAE: {summary['MAE']}")
    if out:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=hdr)
            w.writeheader()
            w.writerows(rows)
        click.echo(f"Wrote report -> {out}")


if __name__ == "__main__":
    main()
