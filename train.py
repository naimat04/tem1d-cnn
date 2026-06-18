"""Train the TEM inversion CNN on synthetic/training data and save the
trained model, fitted scalers, training curves, and metrics.

Usage:
    python train.py --mat-path data/Kashidtrainingdata1perc2.mat \\
                     --output-dir outputs
"""

import argparse
import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from dataset import build_xy, load_training_mat, scale_and_reshape, split_dataset
from model import build_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mat-path", required=True, help="Path to the training .mat file")
    p.add_argument("--output-dir", default="outputs", help="Where to save the model/scalers/plots")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--im-height", type=int, default=28, help="Number of time gates in the input")
    p.add_argument("--n-layers", type=int, default=20, help="Number of resistivity (and depth) layers in the target")
    p.add_argument("--neurons", type=int, default=16)
    p.add_argument("--kern-sz", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    raw = load_training_mat(args.mat_path)
    X, y = build_xy(raw["response"], raw["all_resistivity"], raw["all_depth"])
    print(f"Data shape: X={X.shape}, y={y.shape}")

    X_train, X_valid, X_test, y_train, y_valid, y_test = split_dataset(X, y)
    X_train, X_valid, X_test, y_train, y_valid, y_test, scaler_X, scaler_y = scale_and_reshape(
        X_train, X_valid, X_test, y_train, y_valid, y_test
    )

    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Valid: X={X_valid.shape}, y={y_valid.shape}")
    print(f"Test:  X={X_test.shape}, y={y_test.shape}")

    # MirroredStrategy works fine on CPU / a single GPU too -- it simply
    # falls back to a single-replica strategy in that case.
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_model(
            im_height=args.im_height, output_dim=2 * args.n_layers, neurons=args.neurons, kern_sz=args.kern_sz
        )
        model.compile(optimizer=Adam(args.learning_rate), loss="mse", metrics=["mae"])
    model.summary()

    early_stopping = EarlyStopping(monitor="val_loss", patience=20, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1)

    hist = model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        callbacks=[reduce_lr, early_stopping],
    )

    # Final evaluation, computed once after training completes.
    y_train_pred = model.predict(X_train, verbose=0)
    y_valid_pred = model.predict(X_valid, verbose=0)
    y_test_pred = model.predict(X_test, verbose=0)

    test_mse = mean_squared_error(y_test.flatten(), y_test_pred.flatten())
    test_mae = mean_absolute_error(y_test.flatten(), y_test_pred.flatten())
    train_r2 = r2_score(y_train.flatten(), y_train_pred.flatten())
    valid_r2 = r2_score(y_valid.flatten(), y_valid_pred.flatten())
    test_r2 = r2_score(y_test.flatten(), y_test_pred.flatten())

    print(
        f"Final: Test MSE={test_mse:.6f}, Test MAE={test_mae:.6f}, "
        f"Train R2={train_r2:.6f}, Valid R2={valid_r2:.6f}, Test R2={test_r2:.6f}"
    )

    # --- Training curves ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist.history["loss"], label="Train Loss")
    plt.plot(hist.history["val_loss"], label="Val Loss")
    plt.title("MSE Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist.history["mae"], label="Train MAE")
    plt.plot(hist.history["val_mae"], label="Val MAE")
    plt.title("MAE over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_curves.png"), dpi=150)

    # --- Per-epoch history CSV ---
    history_df = pd.DataFrame(
        {
            "epoch": range(1, len(hist.history["loss"]) + 1),
            "train_loss": hist.history["loss"],
            "val_loss": hist.history["val_loss"],
            "train_mae": hist.history["mae"],
            "val_mae": hist.history["val_mae"],
        }
    )
    history_df.to_csv(os.path.join(args.output_dir, "training_history.csv"), index=False)

    # --- Final metrics + hyperparameters ---
    final_metrics = {
        "test_mse": test_mse,
        "test_mae": test_mae,
        "train_r2": train_r2,
        "valid_r2": valid_r2,
        "test_r2": test_r2,
        "params": vars(args),
    }
    with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)

    # --- Persist model + scalers for evaluate.py / predict_field.py / fence_diagram.py ---
    model.save(os.path.join(args.output_dir, "tem_inversion_model.keras"))
    joblib.dump(scaler_X, os.path.join(args.output_dir, "scaler_X.joblib"))
    joblib.dump(scaler_y, os.path.join(args.output_dir, "scaler_y.joblib"))
    print(f"Saved model, scalers, history, and metrics to {args.output_dir}/")


if __name__ == "__main__":
    main()
