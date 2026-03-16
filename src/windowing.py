import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Sliding Window Formulation
#
# Given a time series of shape (T, F) and a window size W:
#   X[i] = metrics[i : i+W, :]       → shape (W, F), flattened to (W*F,)
#   y[i] = label[i+W]                 → predict the label at the end of the window
#
# Why flatten rather than keep 2D?
#   XGBoost operates on flat feature vectors. Flattening preserves all temporal
#   information and lets the tree learner discover lag-based patterns.
#   For LSTM/CNN, you would keep the (W, F) shape instead.
#
# Temporal features added:
#   - Per-feature rolling mean and std over the window (summary statistics)
#   - Per-feature trend (last value minus first value in window)
#   These help the model distinguish a steady high CPU from a rapidly rising one.
#
# Train/test split:
#   We split TEMPORALLY (first 80% = train, last 20% = test).
#   Never shuffle time series data — shuffling causes data leakage because
#   nearby windows overlap. A temporal split correctly simulates deployment.
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = ["cpu_usage", "memory_usage", "error_rate"]


def make_windows(df: pd.DataFrame, W: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract sliding windows from the DataFrame.

    Returns
    -------
    X : shape (n_windows, W * n_features + 3 * n_features)
        Raw flattened window + rolling mean, std, trend per feature
    y : shape (n_windows,)
        Binary label at the end of each window
    """
    metrics = df[FEATURE_COLS].values   # (T, F)
    labels  = df["label"].values         # (T,)
    T, F = metrics.shape
    n_windows = T - W

    X_raw   = np.zeros((n_windows, W * F))
    X_stats = np.zeros((n_windows, 3 * F))  # mean, std, trend per feature
    y       = np.zeros(n_windows, dtype=int)

    for i in range(n_windows):
        window = metrics[i : i + W]           # (W, F)
        X_raw[i]   = window.flatten()
        X_stats[i] = np.concatenate([
            window.mean(axis=0),              # mean per feature
            window.std(axis=0),               # std per feature
            window[-1] - window[0],           # trend per feature
        ])
        y[i] = labels[i + W]

    X = np.concatenate([X_raw, X_stats], axis=1)
    return X, y


def temporal_train_test_split(
        X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8
) -> tuple:
    split = int(len(X) * train_ratio)
    return X[:split], X[split:], y[:split], y[split:]
