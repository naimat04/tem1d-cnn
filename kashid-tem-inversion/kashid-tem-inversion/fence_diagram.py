"""Build an interactive 3D 'fence diagram' of predicted resistivity
sections across multiple field surveys, stitched together using
GPS-derived UTM coordinates, and save it as a standalone HTML file.

Each survey needs a matching pair of files in the same folder:

* ``Kashid_<id>.mat``  -- the Voltage decay curves (model input)
* ``kashid<id>.csv``   -- station coordinates with X_Longitude/Y_Latitude

Usage:
    python fence_diagram.py --data-dir data --model-dir outputs \\
                             --output outputs/TEM_3D_Fence_Diagram.html
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.io import loadmat
from tensorflow.keras.models import load_model

from dataset import preprocess_field_input
from utils import compute_doi_mask, latlon_to_utm, split_into_profiles


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True, help="Folder with Kashid_*.mat and kashid*.csv files")
    p.add_argument("--model-dir", default="outputs", help="Directory with the trained model + scalers")
    p.add_argument("--n-layers", type=int, default=20, help="Number of resistivity/depth layers")
    p.add_argument(
        "--gap-threshold",
        type=float,
        default=150.0,
        help="Distance (m) between consecutive stations beyond which a new profile starts",
    )
    p.add_argument("--doi-threshold-frac", type=float, default=0.1, help="Depth-of-investigation cutoff fraction")
    p.add_argument("--output", default="TEM_3D_Fence_Diagram.html")
    p.add_argument("--npz-dir", default=None, help="Optional folder to also dump per-profile .npz arrays")
    return p.parse_args()


def find_survey_pairs(data_dir):
    mat_files = sorted(
        f for f in os.listdir(data_dir) if f.endswith(".mat") and f.startswith("Kashid_") and f[7:13].isdigit()
    )
    csv_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv") and f.startswith("kashid"))
    return list(zip(mat_files, csv_files))


def predict_survey(mat_path, model, scaler_X, scaler_y, n_layers):
    voltage = loadmat(mat_path)["Voltage"]
    X_new = preprocess_field_input(voltage, scaler_X)
    y_pred = model.predict(X_new, verbose=0).squeeze()
    y_pred = 10 ** scaler_y.inverse_transform(y_pred)
    return y_pred[:, :n_layers], y_pred[:, n_layers:]


def main():
    args = parse_args()
    if args.npz_dir:
        os.makedirs(args.npz_dir, exist_ok=True)

    model = load_model(os.path.join(args.model_dir, "tem_inversion_model.keras"))
    scaler_X = joblib.load(os.path.join(args.model_dir, "scaler_X.joblib"))
    scaler_y = joblib.load(os.path.join(args.model_dir, "scaler_y.joblib"))

    pairs = find_survey_pairs(args.data_dir)
    if not pairs:
        raise SystemExit(f"No matching Kashid_*.mat / kashid*.csv pairs found in {args.data_dir}")

    # First pass: predict every survey once, derive profile splits, and
    # track the global log-resistivity range so every trace can share one
    # colour scale.
    surveys = []
    global_min, global_max = np.inf, -np.inf

    for mat_file, csv_file in pairs:
        res_pred, depth_pred = predict_survey(
            os.path.join(args.data_dir, mat_file), model, scaler_X, scaler_y, args.n_layers
        )

        coords = pd.read_csv(os.path.join(args.data_dir, csv_file))
        x_utm, y_utm = latlon_to_utm(coords["X_Longitude"].values, coords["Y_Latitude"].values)

        dist = np.sqrt(np.diff(x_utm) ** 2 + np.diff(y_utm) ** 2)
        dist = np.insert(np.cumsum(dist), 0, 0)
        starts, ends = split_into_profiles(dist, gap_threshold=args.gap_threshold)

        surveys.append((mat_file, res_pred, depth_pred, x_utm, y_utm, dist, starts, ends))

        for ps, pe in zip(starts, ends):
            res_sub = res_pred[ps : pe + 1, :]
            depth_sub = depth_pred[ps : pe + 1, :]
            mask = compute_doi_mask(res_sub, depth_sub, args.doi_threshold_frac)
            res_masked = np.where(mask, res_sub, np.nan)
            global_min = min(global_min, np.nanmin(np.log10(res_masked)))
            global_max = max(global_max, np.nanmax(np.log10(res_masked)))

    # Second pass: build one Plotly Surface trace per profile.
    fig = go.Figure()
    trace_idx = 0
    for mat_file, res_pred, depth_pred, x_utm, y_utm, dist, starts, ends in surveys:
        for ps, pe in zip(starts, ends):
            res_sub = res_pred[ps : pe + 1, :]
            depth_sub = depth_pred[ps : pe + 1, :]
            mask = compute_doi_mask(res_sub, depth_sub, args.doi_threshold_frac)
            res_masked = np.where(mask, res_sub, np.nan)

            Xf = np.tile(x_utm[ps : pe + 1], (depth_sub.shape[1], 1))
            Yf = np.tile(y_utm[ps : pe + 1], (depth_sub.shape[1], 1))
            Zf = depth_sub.T
            Rf = res_masked.T

            fig.add_trace(
                go.Surface(
                    x=Xf,
                    y=Yf,
                    z=Zf,
                    surfacecolor=np.log10(Rf),
                    colorscale="Jet",
                    cmin=global_min,
                    cmax=global_max,
                    showscale=(trace_idx == 0),
                    colorbar=dict(
                        title=dict(text="Resistivity (Ohm.m)", side="right"),
                        tickvals=np.linspace(global_min, global_max, 6),
                        ticktext=[f"{10 ** v:.1f}" for v in np.linspace(global_min, global_max, 6)],
                    ),
                )
            )

            if args.npz_dir:
                np.savez(
                    os.path.join(args.npz_dir, f"{mat_file.replace('.mat', '')}_stations{ps}_{pe}.npz"),
                    res_pred=res_sub,
                    depth_pred=depth_sub,
                    dist=dist[ps : pe + 1],
                )

            print(f"Processed: {mat_file} stations {ps}-{pe}")
            trace_idx += 1

    fig.update_layout(
        title="TEM 3D Fence Diagram (Interactive)",
        width=1400,
        height=900,
        scene=dict(
            xaxis_title="UTM Easting (m)",
            yaxis_title="UTM Northing (m)",
            zaxis_title="Depth (m)",
            zaxis=dict(autorange="reversed"),
        ),
    )

    fig.write_html(args.output)
    print(f"Interactive 3D figure saved: {args.output}")


if __name__ == "__main__":
    main()
