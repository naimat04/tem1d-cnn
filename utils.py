"""Shared helpers used by the evaluation and field-visualization scripts:
Monte-Carlo dropout uncertainty estimation, step-plot formatting, lon/lat
to UTM projection, profile splitting, and depth-of-investigation (DOI)
masking.

The DOI-mask and UTM-projection logic was previously copy-pasted twice
inside the 3D fence-diagram notebook cell (once per pass over the data);
it now lives here as plain functions reused by both passes.
"""

from typing import Tuple

import numpy as np
from pyproj import Transformer


def predict_with_uncertainty(model, X: np.ndarray, n_iter: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Monte-Carlo dropout uncertainty estimate.

    Runs ``n_iter`` stochastic forward passes with dropout active
    (``training=True``) and returns the mean and standard deviation
    across runs.

    Note: this only produces a non-zero standard deviation if the model
    actually contains stochastic layers (e.g. ``Dropout``). The CNN built
    by ``model.build_model`` does not currently include any, so this will
    return ``std == 0`` until dropout layers are added -- the same
    limitation present in the original notebook (see README).
    """
    preds = np.array([model(X, training=True).numpy() for _ in range(n_iter)])
    return preds.mean(axis=0), preds.std(axis=0)


def make_step(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert ``(x, y)`` point pairs into step-plot coordinates."""
    x_step = np.repeat(x, 2)
    y_step = np.repeat(y, 2)
    return x_step[1:], y_step[:-1]


def latlon_to_utm(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project WGS84 lon/lat coordinates to the UTM zone of the first point."""
    utm_zone = int(np.floor((lon[0] + 180) / 6) + 1)
    transformer = Transformer.from_crs("epsg:4326", f"epsg:326{utm_zone:02d}")
    return transformer.transform(lat, lon)


def split_into_profiles(dist: np.ndarray, gap_threshold: float = 150.0):
    """Split a cumulative-distance array into contiguous survey profiles.

    A new profile starts wherever the gap between consecutive stations
    exceeds ``gap_threshold`` (in the same units as ``dist``, typically
    metres). Returns ``(starts, ends)`` index arrays, both inclusive.
    """
    split_indices = np.where(np.diff(dist) > gap_threshold)[0]
    starts = np.insert(split_indices + 1, 0, 0)
    ends = np.append(split_indices, len(dist) - 1)
    return starts, ends


def compute_doi_mask(res: np.ndarray, depth: np.ndarray, threshold_frac: float = 0.1) -> np.ndarray:
    """Depth-of-investigation (DOI) mask for one profile.

    For each station (row), find the deepest layer whose predicted
    resistivity is still at least ``threshold_frac`` of that station's RMS
    resistivity, and mask out everything below it as unreliable.

    Args:
        res: Resistivity array, shape ``(n_stations, n_layers)``.
        depth: Depth array, same shape as ``res``.
        threshold_frac: Fraction of the per-station RMS resistivity used
            as the cutoff.

    Returns:
        Boolean mask, same shape as ``res``, ``True`` where a layer is
        considered within the depth of investigation.
    """
    rms = np.sqrt(np.mean(res**2, axis=1))
    threshold = threshold_frac * rms.max()
    mask = np.zeros_like(res, dtype=bool)
    for i in range(res.shape[0]):
        valid_layers = depth[i, :][res[i, :] >= threshold]
        max_depth = valid_layers.max() if valid_layers.size > 0 else depth[i, :].max()
        mask[i, :] = depth[i, :] <= max_depth
    return mask
