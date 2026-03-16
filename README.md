# Incident Predictor — Sliding Window Time Series Classification

Binary early-warning model: given W past timesteps of system metrics,
predict whether an incident will start within the next H timesteps.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
```

## Problem Formulation

| Parameter | Value | Description |
|---|---|---|
| W | 20 | Look-back window (input steps) |
| H | 10 | Prediction horizon (alert lead time) |
| Features | 3 | cpu_usage, memory_usage, error_rate |
| Label | binary | 1 = incident starts within next H steps |

**Sliding window:**
```
metrics[t : t+W]  →  label[t+W]
metrics[t+1 : t+W+1]  →  label[t+W+1]
...
```

## Model: XGBoost with scale_pos_weight

- **Why XGBoost**: fast, handles tabular features well, supports class imbalance via `scale_pos_weight = neg/pos`
- **Why not LSTM**: for short windows (W≤30), flattened tree models match LSTM performance with less complexity
- **Features**: raw flattened window + per-feature rolling mean, std, trend

## Evaluation

- **AUROC**: threshold-independent ranking quality
- **AUPRC**: preferred for imbalanced data (focuses on minority/incident class)
- **F1 @ best threshold**: operational metric at the chosen alert threshold
- **No accuracy**: meaningless with ~10% positive rate

## Train/Test Split

Strictly temporal (first 80% train, last 20% test). No shuffling — shuffling time series causes data leakage via overlapping windows.

## Limitations & Real-System Adaptations

- **Label leakage risk**: windows just before incidents see pre-incident metric rises, which is realistic but means the model partially relies on leading indicators in the data
- **Fixed threshold**: in production, threshold would be tuned per-service based on on-call team's alert fatigue tolerance
- **Concept drift**: model should be retrained periodically as incident patterns change
- **Real data**: would require a time-series database (e.g. Prometheus) and an incident log (e.g. PagerDuty) joined on timestamps
