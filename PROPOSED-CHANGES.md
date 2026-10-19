# Proposed changes — QuantiFly Python 3 port (Track A)

Status: **proposal, not yet implemented.** This document is the plan we agreed on
before touching any code. Baseline = commit for the unmodified Python 2 source plus the
commit adding the legacy conda `environment.yml`. Reviewed once by an independent pass;
its findings are folded in below (see §8 for the correctness blockers to clear first).

## Goal

Stand up a **headless, Python 3, scriptable** version of QuantiFly's core counting
algorithm, exposed as a CLI. No GUI. The point is to be able to train/predict/evaluate
from the command line and run parameter sweeps (MAE vs. egg density), so the tool can
be compared against detection models later.

## Scope

**In scope (Track A):**
- Port the core algorithm in `v2_functions.py` to Python 3.
- Replace the two blocker dependencies (`vigra`, `PyQt4`).
- Add a CLI with `train`, `predict`, `evaluate` commands.
- Define a simple annotation format (painted "dots" PNGs).
- Modernize the dependency/runtime story (replace the legacy conda env).

**Out of scope (deferred — "Track B"):**
- Porting the three PyQt4 GUIs (`v2_release.py`, `v2_releaseAuto.py`, `v2_evaluate.py`).
  They stay in the repo untouched as reference. Interactive ROI/dot labelling will be
  handled by the external GIMP workflow instead (see Annotation format). NOTE: the headless
  batch logic the CLI needs is lifted out of `v2_releaseAuto.py`, so that file is *read* for
  logic even though it isn't ported wholesale — its Py2-isms (below) travel with the lift.

## 1. Dependency changes

| Legacy (remove) | Modern (add) | Why |
|---|---|---|
| `PyQt4`, `QtWebKit` | — (no GUI in Track A) | Unshippable; GUI deferred |
| `vigra` | `scikit-image`, `scipy.ndimage` | No pip wheels; the real swap (see §2) |
| `from scipy.ndimage import filters` | `from scipy.ndimage import gaussian_filter` | **`scipy.ndimage.filters` deprecated (SciPy 1.8), removal in 2.0.** On the headless critical path (density kernel) — see §2b |
| `cPickle` | stdlib `pickle` | Removed in Py3 |
| `from scipy.special import _ufuncs_cxx` | *(delete)* | Was only a PyInstaller bundling hack |
| `import sklearn.utils.lgamma` | *(delete)* | Same; private module gone in modern sklearn |
| pinned `scipy=0.17`, `scikit-learn=0.17`, `numpy=1.11` | current `numpy` / `scipy` / `scikit-learn` | No reason to pin old once vigra is gone |
| `PIL` (Py2 build) | `Pillow` (+ `tifffile` for TIFF stacks) | Maintained fork |
| `pylab.imread` (GUI lift source) | `Pillow` / `tifffile` | Avoid a matplotlib dep for I/O; also fixes a scale bug (see §8) |
| — | `click` (or `argparse`) | CLI entry point |

> TIFF caveat: `tifffile` is pure-Python and reads uncompressed/basic stacks fine, but
> **compressed** TIFFs (LZW/Deflate/etc.) need the optional `imagecodecs` package (Windows
> wheels exist). Add it only if compressed TIFFs show up; QuantiFly inputs are mostly PNG.

### Environment file (the one we committed)
`environment.yml` is **not** preserved as a historical artifact — it can be replaced or
deleted freely. The modern runtime will be:
- A `requirements.txt` pinning the modern stack above, installed with a plain
  `python -m venv .venv` + `pip install -r requirements.txt` — **no conda**.
- `environment.yml` is then **removed**, superseded by `requirements.txt`.

## 2. `v2_functions.py` — the core port

### 2a. vigra → scikit-image (the one substantive change)
vigra is used **only** in the three feature builders:
`local_shape_features_basic` (13 feats, ~L565), `local_shape_features_fine` (21, ~L482),
`local_shape_features_fine_spatial` (23, ~L521). Three operations to replace:

| vigra call | Replacement |
|---|---|
| `vigra.filters.structureTensorEigenvalues(im, inner, outer)` | `skimage.feature.structure_tensor` + `structure_tensor_eigenvalues` |
| `vigra.filters.gaussianGradientMagnitude(im, s)` | `scipy.ndimage.gaussian_gradient_magnitude` |
| `vigra.filters.laplacianOfGaussian(im, s)` | `scipy.ndimage.gaussian_laplace` |

⚠️ **Structure-tensor scale mapping is the real subtlety.** vigra's
`structureTensorEigenvalues(im, inner, outer)` takes **two** scales: `inner` = the
Gaussian-derivative scale for computing the gradient, `outer` = the integration/smoothing
scale (QuantiFly calls it at `inner=s·k`, `outer=2·s·k` across 5 octaves). skimage's
`structure_tensor(image, sigma=...)` takes **one** scale (the integration sigma) and computes
the gradient with a plain finite-difference `np.gradient` — it has **no inner/derivative
scale at all**. So this isn't merely "parameterised differently"; the inner scale is absent.
To approximate vigra: **pre-smooth the image with a Gaussian of the inner scale**, then call
`structure_tensor` with `sigma = outer`. Both skimage functions exist and return eigenvalues
in decreasing order (matches vigra's `[:,:,0]` = larger, `[:,:,1]` = smaller).
`gaussian_gradient_magnitude` and `gaussian_laplace` are correct 1:1 substitutes.
Because of this, **models are not transferable** across backends and old `.p` caches can't be
validated byte-for-byte — we validate on **egg-count accuracy**, not numerical identity.

### 2b. Mechanical Python 2 → 3 fixes (scoped to the ported core)
- **5** bare `print x` statements in `v2_functions.py` (L238, L239, L242, L383, L384) →
  `print(x)`. (The earlier "19" counted across the deferred GUI files; ~15 more live in
  `v2_releaseAuto.py` and will be inherited by the lifted batch logic — convert those too.)
- `from scipy.ndimage import filters` (L11) / `filters.gaussian_filter` (L265) →
  `from scipy.ndimage import gaussian_filter`. **This is on the headless critical path.**
- `cPickle` → `pickle`; add `encoding="latin1"` when loading any legacy `.p` cache
  (or just regenerate features).
- Integer division: only **L207** `(win.p_size-1)/2` is a genuine unguarded Py2 int-div
  (and it sits on the already-broken patch path, §2c). L409/L432 are already wrapped in
  `int(...)` and behave identically in Py3 — no change needed there.
- `par_obj.im_num_range = range(...)` (L752) and the batch lift's
  `left_2_calc = range(...)` + `.pop(...)` (`v2_releaseAuto.py:1408`) → wrap in `list(...)`;
  Py3 `range` is not a list.
- Delete the `_ufuncs_cxx` / `sklearn.utils.lgamma` imports (GUI files only; not reachable
  from the ported core).

### 2c. Latent/broken paths to excise or fix (not caught by `2to3`)
- **The `p_size > 1` patch path in `update_training_samples_fn` is already broken** in the
  original: it references module globals `win.` (L207, L221–224, L228–229) and `v2.` (L226)
  that don't exist inside `v2_functions.py`, and calls `v2.extractPatch` / `v2.regenerateImg`
  when the functions are actually named `extract_patch` (L429) / `regenerate_img` (L407);
  `regenerate_img` itself references an undefined `evalImWin` (L408/L410). Default `p_size=1`
  dodges all of it. **Decision: excise patch mode for the CLI (or repair it deliberately) —
  do not assume it runs.**
- **`v2.bootstrap` is called but never defined** (`v2_release.py`/`v2_releaseAuto.py` bias
  path). If we port the CI/bias-correction logic we hit a missing function — see §8 decision.
- `!= None` / `== None` on arrays (`denseRegion` ~L437/L469; `gt_vec` ~L78) → `is (not) None`.
- `Tiff_Controller` (~L802): `.next()` (L835) → `__next__`, **and** note `maxFrames` ends as
  the max frame *index* (N−1), not the count — downstream mixes `maxFrames` and
  `maxFrames+1`. Replace the class with `tifffile`/`PIL.ImageSequence` and enumerate frames
  `0..N−1` consistently.
- `local_shape_features_fine_spatial` meshgrid bug (L529, L558–559):
  `np.meshgrid(range(imSizeC), range(imSizeR))` yields shape `(imSizeR, imSizeC)` but is
  assigned into `f[:,:,21/22]` of shape `(imSizeC, imSizeR)` — only works for square images.
  Fix the meshgrid order if `fineSpatial` is exposed (it's not the default).
- `exec('a = int_obj.CH_cbx'+str(c)+'...')` (L623) is GUI-coupled and relies on Py2 `exec`
  scope rules (broken in Py3). Not needed headless — channel selection becomes a CLI arg.
- Deferred-file notes (Track B, but watch if logic is lifted): `xrange`
  (`v2_release.py:371`, in `hyphen_range` — relevant if the CLI reuses TIFF frame-range
  parsing), `import thread` / `QtCore.QString` / `scipy.sparse.csgraph._validation`
  (`v2_evaluate.py`).

### 2d. Feature cache — see §9 explainer

### 2e. Image size limit — see §9 explainer

## 3. New CLI

New module (`quantifly/` package + `cli.py`), reusing the ported core. The batch logic
already exists scattered in `v2_releaseAuto.py`; it gets lifted out of Qt. Prefer keying
feature/prediction arrays by **explicit image id/path** rather than reproducing the original's
fragile running-`count` offset bookkeeping (`im_pred_inline_fn`/`evaluate_forest` L288/L312/L360,
and the `count = test_im_start - 1` hacks in `v2_releaseAuto.py`).

```
quantifly train    --images DIR --out model.pkl   [--feature fine|basic|fineSpatial --sigma N --trees N --seed N ...]
quantifly predict  --images DIR --model model.pkl --out counts.csv [--density-maps DIR]
quantifly evaluate --images DIR --model model.pkl --out report.csv [--sweep param=a,b,c --seed N]
```

- **`predict`**: model + images → predicted counts (+ optional density maps). No labels needed.
- **`evaluate`**: same, but requires ground-truth dots and reports per-image absolute/%
  error and MAE across the set; supports parameter sweeps.
- **`train`**: images + ground-truth dots → `model.pkl`.

`ExtraTreesRegressor` (L232) is kept as-is: it passes an **integer** `max_features` (=7, from
`par_obj.max_features`), which is still valid in current sklearn — only the *default* changed
(`'auto'`→`1.0`), so no edit is required there. The signature is
`ExtraTreesRegressor(n_estimators=100, *, ...)`, so the positional first arg plus all-keyword
rest is valid. **Add `random_state`** (see §8 reproducibility).

### Model schema (`model.pkl`)
Pin an explicit schema so `predict`/`evaluate` can reconstruct feature settings. At minimum:
the fitted RF, `feature_type`, `feature_scale`, `sigma_data`, `ch_active`, and — if bias
correction is kept — `M`/`c` (and whatever `evaluate` needs for CI). The GUI's save dict
(`v2_release.py:1106`) is the reference for the full field list.

## 4. Annotation format (the GIMP "dots" workflow)

One supported format. In GIMP: open image → new transparent layer → dot each egg →
export **just the dots layer** as PNG. Rules:

1. **One dot per egg.**
2. **Dots for different eggs must not touch** (small gap, small brush ~3–6 px).
3. **Dots clearly brighter than background** (white is simplest).
4. **Same width × height as the source image**, exported 1:1.

Ingestion: threshold the dots PNG to a mask → `scipy.ndimage.label` connected components →
**one centroid per blob** = one egg. Robust to brush size and anti-aliasing, and replaces the
original's fragile `sum(gt)/255` counting (L394).

**Ground-truth counting must be made self-consistent end-to-end** (the original uses three
different scales: `evaluate_forest` sums channel-0 pixels with *no* `/255` at L394–396;
`make_correction` uses `/255` at L111; the auto script reads GT via `pylab.imread` as 0–1
floats). The port standardizes on:
- **evaluate** GT count = **blob count** (not a pixel sum);
- **train/predict** density = place **one 255-valued pixel per centroid**, then Gaussian-blur
  (`update_density_fn`, L243), so predicted `sum/255` and GT blob-count share one scale;
- keep the `(row, col)` convention aligned: `update_density_fn` indexes `dots_im[row, col]`
  (L260–263) and `ndimage` centroids are `(row, col)` — no transpose.

Pairing convention: `name.png` ↔ `name_dots.png` (default), and also accept the published
datasets' `NNNcell.png` / `NNNdots.png`. No `image,x,y` CSV — you never record coordinates by hand.

## 5. New files
- `requirements.txt` — modern deps, installed into a `python -m venv .venv` virtualenv (no conda).
- `quantifly/` — ported core + CLI.
- `.gitignore` — `__pycache__/`, `*.pyc`, `*.p` feature caches, `.venv/`, model/output artifacts.
- README update — new install + CLI usage (legacy README kept or moved).
- Remove `environment.yml` (superseded by `requirements.txt`).

## 6. Validation plan
Run `evaluate` against `data04-20140331-DM` (8 image/dot pairs) and report MAE / % error as
the first real accuracy anchor. **Blocked on §8.1 and §8.2** (the input-dtype resolution) —
until those are settled the accuracy numbers are meaningless.

## 7. Open questions / decisions pending
- Datasets (~16 MB zipped total) are **not** committed here — download script or a
  gitignored `data/` folder?
- Bias correction / CI: report raw or corrected counts? (see §8).
- Expose `fineSpatial` at all? (has the meshgrid bug; not the default).

## 8. Correctness blockers to clear BEFORE implementation (from review)
1. **Input dtype / `*255` overflow.** `np.array(PIL.Image.open(png)) * 255` (L294, L615,
   L720) overflows `uint8` mod 256 for ordinary 8-bit PNGs (e.g. 10→246) → garbage features.
   This is not a Py2/Py3 issue; it "works" only if the `cell.png` inputs are effectively
   binary/normalized. **Action: inspect the actual dtype/value range of the published
   `cell.png` files first**, then replace with dtype-safe handling (`.astype(np.float32)`
   without the `×255`, scaling only genuinely-normalized float inputs).
2. **Reader/scale mismatch.** Features read via `PIL*255` while the auto script reads GT via
   `pylab.imread` (0–1 float). Standardize on one reader (Pillow/tifffile) and one scale.
3. **GT counting consistency** — implement the end-to-end scheme in §4.
4. **Structure-tensor inner/outer scale mapping** — §2a (pre-smooth by inner scale).
5. **Grayscale (2-D) inputs.** `feature_create` indexes `imRGB[:,:,ch]` (L334–341) and needs
   3-D; a grayscale PNG loads 2-D (`import_data_fn` sets `numCH=2`). The `--channel` arg must
   guard the 2-D case.
6. **Reproducibility.** `ExtraTreesRegressor` sets no `random_state`; sampling uses unseeded
   `np.random.choice` (L197, L457) / `random`. Add a `--seed` feeding sklearn + numpy + random,
   so MAE-vs-density sweeps are reproducible and comparable.
7. **Cleanup in the same pass:** `list(range(...))` for `.pop`; pin `model.pkl` schema;
   excise/repair the broken `p_size>1` path and missing `v2.bootstrap`; fix the `fineSpatial`
   meshgrid if exposed.

Items 1–4 gate the validation run; 5–7 land in the same implementation pass.

## 9. Explainer — feature cache (§2d) and image size limit (§2e)

**Feature cache.** "Features" are the per-pixel stacks of filter responses the model trains on
(13/21/23 channels × active colour channels). Computing them is the slow step, so the original
**pickles each image's feature array to disk** next to the image (`feature_create`, L343) and
reloads it on the next run instead of recomputing (L299). Those arrays are large — a ~700×700
image with 21 "fine" channels in float64 is ~80 MB *per image*, so a dataset leaves hundreds of
MB of `.p` files lying around. For a batch CLI that touches each image once per run, this mostly
just clutters disk. **Change:** compute features **in memory**, write **no cache by default**;
offer an **opt-in** `--cache` for repeated sweeps over the same images, stored as **float32**
(half the size).

**Image size limit.** The original **rejects any image larger than 756×756**
(`import_data_fn`, L703–705 and L725–727: "Please reduce to less than 756x756"). That cap is a
2015 GUI performance guard — the method runs per-pixel feature extraction + a forest prediction
for every pixel, and a big image would freeze the UI / exhaust RAM. It is **not** an algorithmic
requirement. **Change:** make it a configurable flag, **default off**, so full-resolution vial
photos run without forced downscaling. Caveat: bigger images = more pixels = slower + more RAM,
so an optional downscale/tiling switch may follow — but no hard 756 wall.
