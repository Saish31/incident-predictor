from xgboost import XGBClassifier
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Model Choice: XGBoost Gradient Boosted Trees
#
# Why XGBoost over LSTM/Random Forest?
#   - LSTM would be appropriate if temporal dependencies are long-range (W >> 50).
#     For short windows (W=20), flattened trees capture lag patterns well.
#   - Random Forest doesn't handle class imbalance as gracefully as XGBoost's
#     scale_pos_weight parameter (which directly adjusts the loss for the
#     minority class — incidents are rare, ~5-10% of timesteps).
#   - XGBoost is fast to train, interpretable via feature importance, and
#     produces well-calibrated probability outputs (needed for threshold tuning).
#
# Class imbalance handling:
#   Incidents are rare. Without correction, the model learns to always predict 0.
#   scale_pos_weight = count(negatives) / count(positives) reweights the loss
#   so each positive example contributes equally to the gradient as all negatives.
#
# Output:
#   predict_proba() returns P(incident within H steps | window).
#   We do NOT threshold at 0.5 by default — threshold is tuned separately
#   using the precision-recall curve (see evaluation.py).
# ──────────────────────────────────────────────────────────────────────────────


class IncidentPredictor:

    def __init__(self, scale_pos_weight: float = 1.0, seed: int = 42):
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,   # handles class imbalance
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray):
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(label=1) for each window."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
