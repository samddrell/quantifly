"""Local shape features — Python 3 port of the vigra-based feature builders.

Ported from ``v2_functions.local_shape_features_{basic,fine,fine_spatial}``.

Backend swap (see PROPOSED-CHANGES.md §2a):
    vigra.filters.structureTensorEigenvalues(im, inner, outer)
        -> pre-smooth with a Gaussian of ``inner``, then
           skimage.feature.structure_tensor(sigma=outer) + structure_tensor_eigenvalues
    vigra.filters.gaussianGradientMagnitude(im, s) -> scipy.ndimage.gaussian_gradient_magnitude
    vigra.filters.laplacianOfGaussian(im, s)       -> scipy.ndimage.gaussian_laplace

vigra's structure tensor takes TWO scales: ``inner`` is the derivative/gradient scale,
``outer`` the integration scale. skimage's ``structure_tensor`` only takes the integration
sigma and computes gradients by finite difference, so we approximate the inner scale by
pre-smoothing the image. Eigenvalues come back in decreasing order from both libraries,
matching the original's ``[:, :, 0]`` = larger, ``[:, :, 1]`` = smaller.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_gradient_magnitude, gaussian_laplace, gaussian_filter
from skimage.feature import structure_tensor, structure_tensor_eigenvalues

FEATURE_CHANNELS = {"basic": 13, "fine": 21, "fineSpatial": 23}


def _structure_tensor_eigs(im: np.ndarray, inner: float, outer: float) -> np.ndarray:
    """vigra structureTensorEigenvalues(im, inner, outer) analogue.

    Returns an (H, W, 2) array: [..., 0] = larger eigenvalue, [..., 1] = smaller.
    """
    smoothed = gaussian_filter(im, inner)
    a_elems = structure_tensor(smoothed, sigma=outer, order="rc")
    eigs = structure_tensor_eigenvalues(a_elems)  # (2, H, W), decreasing
    return np.stack([eigs[0], eigs[1]], axis=-1)


def local_shape_features(im: np.ndarray, scale_start: float, feature_type: str = "fine",
                         im_num: int = 0) -> np.ndarray:
    """Compute the per-pixel feature stack for a single-channel image.

    ``im`` is a 2-D float array (one colour channel). Returns (H, W, C) where C is
    13 (basic), 21 (fine) or 23 (fineSpatial).
    """
    s = float(scale_start)
    im = im.astype(np.float32, copy=False)
    h, w = im.shape
    n = FEATURE_CHANNELS[feature_type]
    f = np.zeros((h, w, n), dtype=np.float32)

    # Scale octaves used by each feature type.
    if feature_type == "basic":
        octaves = [1, 2, 4]
    else:  # fine / fineSpatial
        octaves = [1, 2, 4, 8, 16]

    # Channel 0 is the raw intensity (fineSpatial replaces it with the image index).
    if feature_type == "fineSpatial":
        f[:, :, 0] = float(im_num)
    else:
        f[:, :, 0] = im

    idx = 1
    for k in octaves:
        f[:, :, idx] = gaussian_gradient_magnitude(im, s * k); idx += 1
        st = _structure_tensor_eigs(im, s * k, s * 2 * k)
        f[:, :, idx] = st[:, :, 0]; idx += 1
        f[:, :, idx] = st[:, :, 1]; idx += 1
        f[:, :, idx] = gaussian_laplace(im, s * k); idx += 1

    if feature_type == "fineSpatial":
        # (row, col) spatial coordinates — note the original had a transposed meshgrid
        # bug (v2_functions.py:529); this uses correct (H, W) ordering.
        rr, cc = np.mgrid[0:h, 0:w]
        f[:, :, 21] = rr
        f[:, :, 22] = cc

    return f


def stack_channels(image: np.ndarray, channels, scale_start: float,
                   feature_type: str = "fine", im_num: int = 0) -> np.ndarray:
    """Build features for each selected colour channel and concatenate.

    ``image`` is (H, W) or (H, W, C_colour). ``channels`` is an iterable of channel
    indices. Returns (H, W, C_features * len(channels)).
    """
    if image.ndim == 2:
        # Grayscale: only channel 0 is meaningful (blocker #5).
        planes = [image]
    else:
        planes = [image[:, :, c] for c in channels]
    feats = [local_shape_features(p, scale_start, feature_type, im_num) for p in planes]
    return np.concatenate(feats, axis=-1)
