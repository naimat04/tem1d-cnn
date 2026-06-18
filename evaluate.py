"""Run MC-dropout uncertainty estimation on the held-out test split and
plot a true-vs-predicted resistivity/depth profile for one station.

Usage:
    python evaluate.py --mat-path data/Kashidtrainingdata1perc2.mat \\
                        --model-dir outputs --station 11
"""

import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

from dataset import build_xy, load_training_mat, scale_and_reshape, split_dataset
from utils import make_step, predict_with_uncertainty


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mat-path", required=True, help="Path to the training .mat file (used to rebuild the test split)")
    p.add_argument("--model-dir", default="outputs", help="Directory with the trained model + scalers")
    p.add_argument("--station", type=int, default=11, help="Index within the test split to plot")
    p.add_argument("--n-iter", type=int, default=50, help="Number of MC-dropout forward passes")
    p.add_argument("--n-layers", type=int, default=20, help="Number of resistivity/depth layers")
    p.add_argument("--output", default=None, help="Output PNG path (default: <model-dir>/station_<i>.png)")
    return p.parse_args()


def main():
    args = parse_args()

    # Rebuild the exact same train/valid/test split as train.py (same
    # random_state), then keep only the test portion.
    raw = load_training_mat(args.mat_path)
    X, y = build_xy(raw["response"], raw["all_resistivity"], raw["all_depth"])
    _, _, X_test, _, _, y_test = split_dataset(X, y)

    scaler_X = joblib.load(os.path.join(args.model_dir, "scaler_X.joblib"))
    scaler_y = joblib.load(os.path.join(args.model_dir, "scaler_y.joblib"))

    # Reuse the saved (already-fitted) scalers so test-time scaling exactly
    # matches what the model was trained on.
    _, _, X_test, _, _, y_test, _, _ = scale_and_reshape(
        X_test, X_test, X_test, y_test, y_test, y_test, scaler_X=scaler_X, scaler_y=scaler_y
    )

    model = load_model(os.path.join(args.model_dir, "tem_inversion_model.keras"))

    mean_pred, std_pred = predict_with_uncertainty(model, X_test, n_iter=args.n_iter)
    mean_pred = mean_pred.squeeze(-1)
    std_pred = std_pred.squeeze(-1)

    mean_pred = 10 ** scaler_y.inverse_transform(mean_pred)
    # NOTE: applying StandardScaler.inverse_transform to a standard
    # deviation is not strictly correct error propagation (it re-adds the
    # mean offset), but this mirrors the original notebook's approach and
    # is kept as-is -- treat the resulting band as a qualitative indicator
    # rather than a calibrated confidence interval.
    std_pred = scaler_y.inverse_transform(std_pred)

    y_test_lin = 10 ** scaler_y.inverse_transform(y_test.squeeze(-1))

    n = args.n_layers
    res_true, depth_true = y_test_lin[:, :n], y_test_lin[:, n:]
    res_pred, depth_pred = mean_pred[:, :n], mean_pred[:, n:]
    uncertainty = std_pred[:, :n]

    i = args.station
    dep_p, res_p, unc = depth_pred[i], res_pred[i], uncertainty[i]
    dep_t, res_t = depth_true[i], res_true[i]

    order_p = np.argsort(dep_p)
    dep_p, res_p, unc = dep_p[order_p], res_p[order_p], unc[order_p]
    order_t = np.argsort(dep_t)
    dep_t, res_t = dep_t[order_t], res_t[order_t]

    res_lower = np.maximum(res_p - unc, 1e-3)
    res_upper = res_p + unc

    res_p_step, dep_p_step = make_step(res_p, dep_p)
    res_lower_step, _ = make_step(res_lower, dep_p)
    res_upper_step, _ = make_step(res_upper, dep_p)

    plt.figure(figsize=(5, 7))
    plt.step(res_t, dep_t, where="post", color="black", linewidth=2, label="True")
    plt.step(res_p, dep_p, where="post", color="red", linestyle="--", linewidth=2, label="Predicted")
    plt.fill_betweenx(dep_p_step, res_lower_step, res_upper_step, color="#4a90e2", alpha=0.3, label="Uncertainty")

    plt.gca().invert_yaxis()
    plt.xscale("log")
    plt.gca().set_facecolor("#e6e0ff")
    plt.xlabel("Resistivity (Ohm.m)")
    plt.ylabel("Depth (m)")
    plt.title(f"Station {i}")
    plt.legend()
    plt.grid(True, color="white")

    output = args.output or os.path.join(args.model_dir, f"station_{i}.png")
    plt.savefig(output, dpi=150)
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    main()
