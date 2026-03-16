import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score,
    average_precision_score, classification_report,
    confusion_matrix, f1_score,
)

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation Design
#
# Why NOT use accuracy?
#   With ~10% positive rate, a model predicting all-zeros gets 90% accuracy.
#   This is meaningless for alerting. Instead we use:
#
# Primary metrics:
#   - AUROC  : area under ROC curve. Threshold-independent. Good for ranking.
#   - AUPRC  : area under Precision-Recall curve. Better than AUROC for
#              imbalanced data because it focuses on the minority (incident) class.
#   - F1 @ threshold : harmonic mean of precision and recall at the chosen
#              operating point.
#
# Threshold selection:
#   In a real alerting system, the operator chooses a threshold that balances:
#     - Precision: fraction of alerts that are real incidents (avoid alert fatigue)
#     - Recall: fraction of real incidents that trigger alerts (avoid missed incidents)
#   We sweep thresholds and plot the precision-recall curve so the operator can
#   pick the operating point that fits their SLA.
#
#   We also report the "best F1 threshold" as a data-driven default.
#
# Lead time analysis:
#   For each true incident, we check the earliest timestep at which our model
#   would have fired an alert. The average lead time (steps before incident start)
#   is a practical metric for real alerting systems.
# ──────────────────────────────────────────────────────────────────────────────


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = None):
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    # Best F1 threshold
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if threshold is None else threshold
    best_f1 = f1_scores[best_idx]

    y_pred = (y_prob >= best_threshold).astype(int)

    print("=" * 50)
    print(f"  AUROC             : {auroc:.4f}")
    print(f"  AUPRC             : {auprc:.4f}")
    print(f"  Best F1 threshold : {best_threshold:.4f}")
    print(f"  Best F1 score     : {best_f1:.4f}")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=["no-incident", "incident"]))

    return {
        "auroc": auroc,
        "auprc": auprc,
        "best_threshold": best_threshold,
        "best_f1": best_f1,
    }


def plot_precision_recall(y_true: np.ndarray, y_prob: np.ndarray, save_path: str = None):
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(recalls, precisions, lw=2, label=f"AUPRC = {auprc:.3f}")
    plt.xlabel("Recall (fraction of incidents detected)")
    plt.ylabel("Precision (fraction of alerts that are real)")
    plt.title("Precision-Recall Curve — Incident Predictor")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_predictions(y_true: np.ndarray, y_prob: np.ndarray,
                     threshold: float, n_steps: int = 500,
                     save_path: str = None):
    """Visualise predicted probability vs true labels over time."""
    t = np.arange(n_steps)
    plt.figure(figsize=(14, 4))
    plt.plot(t, y_prob[:n_steps], label="P(incident)", alpha=0.7, lw=1.5)
    plt.fill_between(t, 0, y_true[:n_steps].astype(float) * 0.3,
                     alpha=0.3, color="red", label="True label=1")
    plt.axhline(threshold, color="orange", linestyle="--", label=f"Threshold={threshold:.2f}")
    plt.xlabel("Timestep")
    plt.ylabel("Predicted probability")
    plt.title("Model Output vs True Incident Labels")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
