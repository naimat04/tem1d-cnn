# Kashid TEM Inversion

A 1D CNN that inverts transient electromagnetic (TEM) voltage-decay
measurements into layered resistivity/depth profiles, developed for a
submarine groundwater discharge (SGD) survey at Kashid Beach, Raigad
District, Maharashtra.

## What it does

1. **Train** a CNN (`model.py`) on synthetic/forward-modeled TEM data:
   voltage decay curve in → 20-layer resistivity + depth profile out.
2. **Evaluate** the trained model on a held-out test split, with a
   Monte-Carlo-dropout uncertainty plot for a single station.
3. **Predict** resistivity/depth profiles for a single field survey
   (`Voltage` only, no ground truth).
4. **Visualize** many field surveys together as an interactive 3D "fence
   diagram" (Plotly), stitched together with GPS coordinates.

## Repository layout

```
.
├── dataset.py        # .mat loading, train/valid/test split, scaling
├── model.py           # CNN architecture (build_model)
├── utils.py           # MC-dropout, step-plot, UTM, profile-split, DOI mask
├── train.py            # trains the model, saves it + scalers + metrics
├── evaluate.py         # MC-dropout uncertainty plot on the test split
├── predict_field.py    # runs the model on one field .mat file
├── fence_diagram.py    # 3D fence diagram across multiple field surveys
├── requirements.txt
└── README.md
```

There are no subfolders for data/outputs in the repo itself — scripts
create their output directory (default `outputs/`) on demand, and expect
you to point `--mat-path` / `--data-dir` at wherever your `.mat`/`.csv`
files actually live (e.g. a local `data/` folder, untracked).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**1. Train**

```bash
python train.py --mat-path data/Kashidtrainingdata1perc2.mat --output-dir outputs --n-layers 20
```

Saves to `outputs/`: `tem_inversion_model.keras`, `scaler_X.joblib`,
`scaler_y.joblib`, `training_history.csv`, `training_curves.png`,
`final_metrics.json`.

**2. Evaluate (uncertainty plot on the test split)**

```bash
python evaluate.py --mat-path data/Kashidtrainingdata1perc2.mat --model-dir outputs --station 11
```

**3. Predict on a single field survey**

```bash
python predict_field.py --mat-path data/Kashid_221225.mat --model-dir outputs --output predictions.csv
```

**4. Build the 3D fence diagram across all field surveys**

```bash
python fence_diagram.py --data-dir data --model-dir outputs --output outputs/TEM_3D_Fence_Diagram.html
```

`--data-dir` should contain matching pairs of `Kashid_<id>.mat` (voltage
data) and `kashid<id>.csv` (station `X_Longitude`/`Y_Latitude`) files.

## Assumptions and changes made while refactoring

The notebook had a few rough edges that I either preserved-but-documented
or fixed outright. Listed here so nothing is a silent surprise:

- **Decoupled the model's output size from `im_height`.** This is the
  most consequential fix: the original `Net_modified` reused `im_height`
  (the input sequence length, i.e. number of time gates) for the output
  `Dense`/`Reshape` size too. But the target `y` array is actually
  `2 * n_layers` wide (resistivity + depth concatenated), a different
  number from the time-gate count in general — and indeed, running the
  original architecture as written raises a shape-mismatch error from
  Keras as soon as `model.fit` is called, unless a dataset happens to
  have exactly `im_height / 2` layers. `model.build_model` now takes
  `output_dim` as its own explicit argument (`train.py` passes
  `output_dim = 2 * n_layers` via the new `--n-layers` flag, default 20),
  so the two no longer have to coincide. I verified the full pipeline
  (`train.py` → `evaluate.py` / `predict_field.py` / `fence_diagram.py`)
  end-to-end on a synthetic dataset with this fix in place.
- **Removed a duplicate model definition.** `Net_modified` was defined
  twice, identically apart from one unused default argument. Only one
  copy (`model.build_model`) remains.
- **Removed the `enable_dropout` / `dp_coeff` parameters.** They existed
  in both copies of the original function signature but were never
  actually wired into the network body — no `Dropout` layer was ever
  added. Because of this, the notebook's `predict_with_uncertainty`
  (Monte-Carlo dropout) likely produced a near-zero standard deviation in
  practice, since there are no stochastic layers for `training=True` to
  affect. This limitation is preserved as-is and called out in
  `utils.predict_with_uncertainty`'s docstring — if you want genuine
  MC-dropout uncertainty, add `Dropout` layers to `model.build_model`.
- **Fixed an undefined-variable bug in the training loop.** The original
  training cell referenced a variable `epoch` (`if epoch % 1 == 0`) and
  indexed `hist.history[...][0]` after a single `model.fit(..., epochs=200, ...)`
  call — i.e. it only ever recorded the *first* epoch's metrics, and
  `epoch` itself was never assigned (this would raise `NameError` if it
  ever reached that line). This looks like leftover code from an earlier
  manual per-epoch training loop that was later replaced by one
  `model.fit()` call. `train.py` now uses the full per-epoch history
  that `model.fit()` already returns for the loss/MAE curves, and
  computes R²/MSE/MAE once on the final trained model — which preserves
  the original intent (track curves, report final scores) without the
  undefined variable.
- **Persisted the scalers and model to disk.** The notebook kept
  `scaler_X`, `scaler_y`, and `model` alive in memory across cells.
  Since this is now several independent scripts, `train.py` saves both
  scalers (via `joblib`) and the model (`.keras`) so `evaluate.py`,
  `predict_field.py`, and `fence_diagram.py` can load the exact same
  fitted objects later.
- **Standardized the fence-diagram gap threshold.** The original cell
  used a 100 m gap to split survey profiles when computing the global
  color scale (first pass) but 150 m when actually building the surfaces
  (second pass) — almost certainly an inconsistency rather than an
  intentional difference. `fence_diagram.py` uses one configurable
  `--gap-threshold` (default 150 m) for both.
- **De-duplicated the DOI-mask and UTM-projection logic** that was
  copy-pasted across the two passes of the original fence-diagram cell
  into `utils.compute_doi_mask` / `utils.latlon_to_utm`, reused by both
  passes in `fence_diagram.py`.
- **The std-dev inverse-scaling is naive, by design (matches original).**
  `evaluate.py` applies `StandardScaler.inverse_transform` directly to a
  standard-deviation array, which isn't strictly correct error
  propagation (the scaler also re-adds its mean offset). This mirrors
  what the notebook did; treat the resulting band as a rough qualitative
  indicator, not a calibrated confidence interval.
- **Hardcoded Kaggle paths (`/kaggle/input/...`, `/kaggle/working/...`)
  were replaced with CLI arguments** (`--mat-path`, `--data-dir`,
  `--output`, etc.) so the code runs anywhere.
- **`--im-height` defaults to 28**, matching the time-gate count the
  original model was actually built and trained with. (A comment
  elsewhere in the notebook referenced 40 `dBzdt_i` columns from an
  unused/commented-out alternate data path — that comment doesn't
  reflect what the `.mat`-based pipeline actually used, so it was
  dropped rather than followed.) Pass `--im-height` explicitly if your
  data has a different number of time gates.
- **Trimmed unused imports.** The original first cell imported many
  names that were never referenced anywhere in the notebook (e.g.
  `tqdm`, `Timer`, `Path`, `precision_score`, `skimage.transform`,
  `scipy.ndimage`, `savgol_filter`, `tukey`, `Axes3D`,
  `FormatStrFormatter`, `repmat`, Keras `backend`/`optimizers`,
  `Dropout`/`BatchNormalization`/`Conv2D`/etc.). Each file here only
  imports what it actually uses.
- **Number of layers (20) and number of time gates (28) are passed as
  CLI arguments** (`--n-layers`, `--im-height`) rather than hardcoded, so
  you can reuse these scripts if your survey design changes.
