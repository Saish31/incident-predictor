"""
End-to-end training script for the incident predictor.

Design parameters:
    W = 20   : look-back window of 20 timesteps
    H = 10   : predict whether incident starts within next 10 steps

These are hyperparameters — in a real system, W and H are chosen based on:
    - W: how much history is reliably available at inference time
    - H: how much lead time is operationally useful (e.g., enough time to act)
"""

import numpy as np
from src.data_generator import generate_dataset
from src.windowing import make_windows, temporal_train_test_split
from src.model import IncidentPredictor
from src.evaluation import evaluate, plot_precision_recall, plot_predictions

# ── Hyperparameters ───────────────────────────────────────────────────────────
W = 20           # window size (look-back steps)
H = 10           # prediction horizon (steps ahead)
N_STEPS = 10_000 # total timesteps to generate
TRAIN_RATIO = 0.8


def main():
    print(f"Generating synthetic dataset: {N_STEPS} steps, W={W}, H={H}")
    df = generate_dataset(n_steps=N_STEPS, H=H, seed=42)

    incident_rate = df["label"].mean()
    print(f"Positive label rate: {incident_rate:.2%}")

    # ── Build windows ─────────────────────────────────────────────────────────
    X, y = make_windows(df, W=W)
    X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, TRAIN_RATIO)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Use last 10% of train as validation for early stopping
    val_split = int(len(X_train) * 0.9)
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]

    # ── Class imbalance weight ────────────────────────────────────────────────
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / max(pos, 1)
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    # ── Train ─────────────────────────────────────────────────────────────────
    predictor = IncidentPredictor(scale_pos_weight=scale_pos_weight)
    predictor.fit(X_train, y_train, X_val, y_val)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_prob = predictor.predict_proba(X_test)
    results = evaluate(y_test, y_prob)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_precision_recall(y_test, y_prob, save_path="pr_curve.png")
    plot_predictions(y_test, y_prob,
                     threshold=results["best_threshold"],
                     n_steps=500,
                     save_path="predictions.png")

    print("\nDone. Plots saved to pr_curve.png and predictions.png")


if __name__ == "__main__":
    main()
