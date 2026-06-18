"""Data loading and preprocessing for TEM inversion.

Two kinds of ``.mat`` inputs are handled:

* **Training/synthetic data** -- contains ``response`` (simulated voltage
  decay curves), ``all_resistivity`` / ``all_depth`` (the corresponding
  layered-earth model used to generate ``response``), plus ``time``,
  ``check_t_1`` and ``check_t_2`` time-gate metadata.
* **Field survey data** -- contains only ``Voltage``, the measured decay
  curve for each station, to be run through an already-trained model.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_training_mat(path: str) -> dict:
    """Load a training ``.mat`` file and return its relevant arrays."""
    mat = loadmat(path)
    return {
        "all_resistivity": mat["all_resistivity"],
        "all_depth": mat["all_depth"],
        "time": mat["time"].flatten(),
        "response": mat["response"],
        "check_t_1": mat["check_t_1"].flatten(),
        "check_t_2": mat["check_t_2"].flatten(),
    }


def load_field_mat(path: str) -> np.ndarray:
    """Load a field-survey ``.mat`` file and return its ``Voltage`` array."""
    mat = loadmat(path)
    return mat["Voltage"]


def build_xy(
    response: np.ndarray, all_resistivity: np.ndarray, all_depth: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Build model-ready ``(X, y)`` arrays in log10 space.

    ``X`` is ``log10(voltage + 10)``, one row per station/sample.
    ``y`` is ``log10([resistivity, depth])`` concatenated per layer; exact
    zeros are floored to a tiny value before taking the log to avoid
    ``-inf``.
    """
    X_raw = response.T
    y_raw = np.vstack((all_resistivity, all_depth)).T.copy()

    X = np.log10(X_raw + 10)
    y_raw[y_raw == 0] = 1e-10
    y = np.log10(y_raw)
    return X, y


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    valid_size: float = 0.05,
    random_state: int = 32,
):
    """Split into train / validation / test sets.

    Matches the two-stage split used in the original notebook: first carve
    off ``test_size`` of the data, then carve ``valid_size`` of *that*
    remainder off again for validation (so the final validation set is
    smaller than ``valid_size * len(X)``).
    """
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rest, y_rest, test_size=valid_size, random_state=random_state)
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def scale_and_reshape(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    scaler_X: Optional[StandardScaler] = None,
    scaler_y: Optional[StandardScaler] = None,
):
    """Standardize features/targets and add the trailing channel dimension.

    If ``scaler_X`` / ``scaler_y`` are omitted, new scalers are fit on the
    training split (as in the original notebook). Pass previously-fitted
    scalers (e.g. loaded from disk) to reproduce the *exact* same scaling
    at evaluation or inference time, without re-fitting.
    """
    if scaler_X is None:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        y_train = scaler_y.fit_transform(y_train)
    else:
        X_train = scaler_X.transform(X_train)
        y_train = scaler_y.transform(y_train)

    X_valid = scaler_X.transform(X_valid)
    X_test = scaler_X.transform(X_test)
    y_valid = scaler_y.transform(y_valid)
    y_test = scaler_y.transform(y_test)

    X_train, X_valid, X_test = (a[..., np.newaxis] for a in (X_train, X_valid, X_test))
    y_train, y_valid, y_test = (a[..., np.newaxis] for a in (y_train, y_valid, y_test))

    return X_train, X_valid, X_test, y_train, y_valid, y_test, scaler_X, scaler_y


def preprocess_field_input(voltage: np.ndarray, scaler_X: StandardScaler) -> np.ndarray:
    """Apply the same log10 + standardization pipeline used for training data."""
    X_new = np.log10(voltage + 10).T
    X_new = scaler_X.transform(X_new)
    return X_new[..., np.newaxis]
