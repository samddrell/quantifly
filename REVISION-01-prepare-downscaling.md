# Revision 01 — image downscaling via a separate `prepare` step

Status: **proposal, not yet implemented.** Revises PROPOSED-CHANGES.md §2e (image size
limit). Baseline = commit `cc71039` (working Python 3 port + CLI).

## Problem

Target datasets are iPhone photos at **3024×4032** (~12.2 MP), ~50 images and growing, with
the egg-bearing plate filling most of the frame (so cropping can't meaningfully shrink them).
At largest, **eggs are ~50 px** across. Two issues with running these at full resolution:

1. **Scale mismatch (correctness).** The feature pyramid uses fixed Gaussian scales
   (`feature_scale` ≈ 0.8 px up through the 16× octave) and the density blur assumes eggs are
   a few pixels wide. At 3024×4032 an egg is ~50 px — far outside what those scales model, so
   the forest would see egg-interior texture rather than egg-sized blobs.
2. **Compute/memory.** Per-pixel features at 12.2 MP × 63 (3-channel "fine") × 4 bytes ≈ 3 GB
   per image for the feature array alone, plus intermediate filtered arrays and a forest
   prediction over 12.2 M rows. Impractical across 50+ images.

## Decision

Add a **separate `prepare` (downscale) step**. The user downscales full-res images **once**,
then **annotates the prepared (downscaled) images** in GIMP. This is preferred over scaling
dot coordinates at load time because:

- The scaling transform (the "seam") lives in **one place, applied to images only**, and is
  visually verifiable (you annotate the exact image you'll train on).
- Dots are **created at the working resolution**, so there is **no dot-coordinate transform
  anywhere** in the pipeline.
- A dots mask is **never resampled** (resampling would blur/merge dots); only the photographic
  cell image is resampled.
- The existing core already handles "same-dimension image + dots → centroids," so
  **`features.py` and the dot/centroid logic are unchanged.**

### Tradeoff and mitigation
Prepare-first **binds annotations to the chosen working resolution** (changing it later means
re-annotating). Mitigation: **lock the resolution now.** With eggs ~50 px on a 4032 long side,
target eggs ~10–12 px (the scale-matched range):

- **`--max-dim 1000`** (factor ≈ 4): 4032 → 1008, egg ≈ 12 px, ~760×1010 — the regime the
  original data lived in; seconds-per-image compute.

Commit to max-dim 1000 once; every batch (now and future) goes through the same `prepare`.

## Spec — `quantifly prepare`

```
quantifly prepare --in <raw dir> --out <working dir> [--max-dim 1000]
```

- Read `W, H` from `PIL.Image.size`, which is `(width, height)` — **not** numpy's
  `(H, W)` from `np.array(img).shape`. Deriving W/H from the array shape transposes the output
  (B3).
- For each image file in `--in`: compute `scale = max_dim / max(W, H)`.
  - If `scale >= 1` (image already within the target), **copy unchanged — never upscale.**
  - Else resize to `(round(W*scale), round(H*scale))` (width, height order) using
    **anti-aliased Lanczos** (`PIL.Image.LANCZOS`), preserving aspect ratio.
- Write each output as PNG with the **same filename** into `--out` (so the pairing conventions
  in `io.find_pairs` keep working).
- **Input homogeneity (B1):** all raws are assumed to share capture resolution/zoom (true for a
  single iPhone capture scheme). The goal is a consistent egg **pixel size**, not a uniform
  output dimension. If input `max(W, H)` varies by more than ~10% across the batch, **warn** —
  mixed capture resolutions would make some images downscale and others copy-unchanged, yielding
  inconsistent egg scales and defeating the purpose.
- **Dots safety (B4):** skip any input whose name matches the dots convention — case-insensitive
  `*_dots`, or `NNNdots` when a matching `NNNcell` exists (the same rule as the hardened
  `find_pairs`) — with a warning. Dots are drawn on the prepared output, never resampled.
- Write a **manifest** `prepare_manifest.json` into `--out` recording `max_dim` and, per image,
  the source `(W, H)` and applied `scale`, for traceability and the model guard below.
- Create `--out` if missing. Never overwrite the originals in `--in`.

### Resampling note
Downscaling a photo without anti-aliasing causes aliasing that corrupts the gradient/structure
features. Lanczos (or a box/area filter) with anti-aliasing is required; nearest-neighbor is
not acceptable for the cell image.

## Model guard (cheap safety net)

The mistake this design allows is feeding an **unprepared** (full-res) or differently-prepared
image to `predict`/`evaluate`. Guard on the **intended** `max_dim`, not an empirical per-corpus
maximum (B2 — an empirical max is inflated by one accidentally-large training image and only
catches gross errors in one direction):

- `prepare` records `max_dim` in `prepare_manifest.json`.
- `train` reads that manifest from the images directory and stamps `prepared_max_dim` into
  `model.pkl` (falling back to the observed training `max(W, H)` **with a note** if no manifest
  is present).
- `predict`/`evaluate` **warn** if a test image's `max(W, H)` falls outside
  `[0.9, 1.1] × prepared_max_dim` — too large (unprepared) *or* too small (over-shrunk), either
  of which shifts egg pixel size and silently skews counts.

Advisory only; it does not resample or reject.

## Code impact
- **New:** `quantifly/prepare.py` — `prepare_images(in_dir, out_dir, max_dim)` + manifest write.
- **`cli.py`:** add the `prepare` command; emit the size-mismatch warning in predict/evaluate.
- **`core.py`:** `train()` reads `prepare_manifest.json` and stamps `prepared_max_dim`.
- **Unchanged:** `features.py`, dot ingestion, density build, the ExtraTrees training.

## Workflow after this revision
```
quantifly prepare  --in raw/ --out work/ --max-dim 1000     # downscale once
#   ... annotate work/*.png in GIMP -> work/*_dots.png ...
quantifly train    --images work/ --out model.pkl
quantifly evaluate --images work/ --model model.pkl
quantifly predict  --images work_new/ --model model.pkl --out counts.csv
```

## Open items
- Confirm max-dim 1000 after measuring a real sample egg (estimate only; ~50 px assumed).
- 16-bit / HEIC iPhone originals: confirm inputs are 8-bit RGB (JPEG/PNG) before `prepare`;
  HEIC would need conversion first (out of scope here).

## Review notes folded in (REVISION-01 v2)
Independent review found the resize math, same-filename PNG output, and anti-aliased-Lanczos
requirement sound, and confirmed the "annotate at working resolution, never transform dot
coords" decision keeps `features.py`/dot logic untouched. Resolved before coding: **B1**
(input-homogeneity warning + egg-scale, not dimension, is the goal), **B2** (guard on the
manifest's intended `max_dim`, warn both directions), **B3** (W=width from `Image.size`), **B4**
(case-insensitive dots-skip matching the hardened `find_pairs`). Manifest adopted (was an open
item).
