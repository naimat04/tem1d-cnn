"""Run a trained TEM inversion model on a single field-survey ``.mat`` file
(containing only a ``Voltage`` array) and save the predicted
resistivity/depth profiles to CSV.

Usage:
    python predict_field.py --mat-path data/Kashid_221225.mat \\
                             --model-dir outputs --output predictions.csv
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from dataset import load_field_mat, preprocess_field_input


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mat-path", required=True, help="Field .mat file with a Voltage array")
    p.add_argument("--model-dir", default="outputs", help="Directory with the trained model + scalers")
    p.add_argument("--n-layers", type=int, default=20, help="Number of resistivity/depth layers")
    p.add_argument("--output", default="predictions.csv")
    return p.parse_args()


def main():
    args = parse_args()

    model = load_model(os.path.join(args.model_dir, "tem_inversion_model.keras"))
    scaler_X = joblib.load(os.path.join(args.model_dir, "scaler_X.joblib"))
    scaler_y = joblib.load(os.path.join(args.model_dir, "scaler_y.joblib"))

    voltage = load_field_mat(args.mat_path)
    X_new = preprocess_field_input(voltage, scaler_X)

    y_pred = model.predict(X_new).squeeze()
    y_pred = 10 ** scaler_y.inverse_transform(y_pred)

    n = args.n_layers
    res_cols = [f"resistivity_{i + 1}" for i in range(n)]
    depth_cols = [f"depth_{i + 1}" for i in range(n)]
    out_df = pd.DataFrame(np.hstack([y_pred[:, :n], y_pred[:, n:]]), columns=res_cols + depth_cols)
    out_df.to_csv(args.output, index=False)
    print(f"Saved {len(out_df)} station predictions to {args.output}")


if __name__ == "__main__":
    main()
