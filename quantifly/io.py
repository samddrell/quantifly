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

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label, center_of_mass


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
            dots_im[r, c] += 255.0  # accumulate: two eggs rounding to one pixel still count twice
    return gaussian_filter(dots_im, float(sigma))


def find_pairs(image_dir, require_dots: bool = True):
    """Find (image_path, dots_path_or_None) pairs in a directory.

    Supports two conventions (case-insensitive):
      - name.png     +  name_dots.png
      - NNNcell.png  +  NNNdots.png   (the published datasets)

    A ``*dots`` file is treated as an annotation only when its matching base image
    exists, so an ordinary image whose name happens to end in "dots" is not dropped.
    """
    image_dir = Path(image_dir)
    pngs = {p.stem: p for p in image_dir.glob("*.png")}
    by_lower = {s.lower(): s for s in pngs}  # case-insensitive stem lookup

    def get(name):
        actual = by_lower.get(name.lower())
        return pngs[actual] if actual is not None else None

    def is_dots(stem: str) -> bool:
        sl = stem.lower()
        if sl.endswith("_dots"):
            return True
        if sl.endswith("dots"):
            return get(stem[:-4] + "cell") is not None  # NNNdots is dots iff NNNcell exists
        return False

    pairs = []
    for stem in sorted(pngs):
        if is_dots(stem):
            continue
        dots = get(stem + "_dots")
        if dots is None and stem.lower().endswith("cell"):
            dots = get(stem[:-4] + "dots")
        if require_dots and dots is None:
            continue
        pairs.append((pngs[stem], dots))
    return pairs
