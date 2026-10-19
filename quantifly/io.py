"""Image and annotation I/O.

Key fixes vs. the original (see PROPOSED-CHANGES.md §8):
 - Load pixels at their true dtype; do NOT multiply PNGs by 255 (that overflowed uint8
   and scrambled the data — the original's `np.array(Image.open(png)) * 255`).
 - One reader (Pillow) at one scale, instead of PIL*255 for images and pylab.imread
   (0-1 floats) for ground truth.
 - Ground-truth dots are counted by connected components (one blob = one egg), which
   reproduces the published single-pixel dots AND handles multi-pixel hand-painted dots.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label, center_of_mass

DOTS_SUFFIXES = ("_dots", "dots")  # name_dots.png  and  NNNdots.png


def load_image(path) -> np.ndarray:
    """Load an image as a numpy array at its native dtype (no scaling)."""
    return np.array(Image.open(str(path)))


def dots_to_points(dots_img: np.ndarray, thresh_frac: float = 0.5):
    """Connected-components centroids of a dots annotation image.

    Returns (points, count) where points is a list of integer (row, col) tuples.
    Robust to multi-pixel / anti-aliased brush dots: each distinct blob counts once.
    """
    ch0 = dots_img[..., 0] if dots_img.ndim == 3 else dots_img
    peak = float(ch0.max())
    if peak <= 0:
        return [], 0
    mask = ch0 > (peak * thresh_frac)
    lab, n = label(mask)
    if n == 0:
        return [], 0
    centroids = center_of_mass(mask, lab, range(1, n + 1))
    points = [(int(round(r)), int(round(c))) for (r, c) in centroids]
    return points, n


def build_density(points, shape, sigma: float) -> np.ndarray:
    """Place 255 at each egg location and Gaussian-blur (sum-preserving).

    sum(density) / 255 == number of eggs, matching the predicted-count convention.
    Mirrors v2_functions.update_density_fn but driven by centroid points.
    """
    from scipy.ndimage import gaussian_filter
    dots_im = np.zeros(shape[:2], dtype=np.float32)
    for (r, c) in points:
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            dots_im[r, c] = 255.0
    return gaussian_filter(dots_im, float(sigma))


def _strip_dots(stem: str) -> str | None:
    """Map a dots-file stem to its image stem, or None if it isn't a dots file."""
    if stem.endswith("_dots"):
        return stem[:-5]
    m = re.fullmatch(r"(.*?)dots", stem)  # e.g. 001dots -> 001  (paired with 001cell)
    if m:
        return m.group(1)
    return None


def find_pairs(image_dir, require_dots: bool = True):
    """Find (image_path, dots_path_or_None) pairs in a directory.

    Supports two conventions:
      - name.png  +  name_dots.png
      - NNNcell.png + NNNdots.png   (the published datasets)
    """
    image_dir = Path(image_dir)
    pngs = {p.stem: p for p in image_dir.glob("*.png")}
    pairs = []
    used_dots = set()
    for stem, path in sorted(pngs.items()):
        if _strip_dots(stem) is not None and stem != _strip_dots(stem):
            continue  # this is a dots file, skip as a primary image
        dots = None
        # name_dots.png
        if f"{stem}_dots" in pngs:
            dots = pngs[f"{stem}_dots"]
        # NNNcell.png -> NNNdots.png
        elif stem.endswith("cell") and f"{stem[:-4]}dots" in pngs:
            dots = pngs[f"{stem[:-4]}dots"]
        if dots is not None:
            used_dots.add(dots.stem)
        if require_dots and dots is None:
            continue
        pairs.append((path, dots))
    return pairs
