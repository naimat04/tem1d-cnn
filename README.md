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
