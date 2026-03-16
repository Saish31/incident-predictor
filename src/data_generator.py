import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic Dataset Design
#
# We simulate three metrics a real monitoring system might expose:
#   - cpu_usage    : baseline ~40%, spikes during incidents
#   - memory_usage : baseline ~50%, gradual rise before incidents
#   - error_rate   : baseline ~2%,  sharp spike at incident onset
#
# Incident definition:
#   An incident is a correlated degradation event. We generate incident
#   start times randomly (~5% of timesteps are incident starts). Each
#   incident lasts INCIDENT_DURATION steps.
#
# Label definition (the prediction target):
#   label[t] = 1  iff  an incident starts within the next H steps
#   label[t] = 0  otherwise
#
# This is a BINARY EARLY-WARNING classification problem — not anomaly detection.
# The model sees a window of W past steps and predicts whether an incident
# will START in the next H steps. This framing is directly applicable to
# real alerting systems (alert BEFORE the incident, not after it starts).
# ──────────────────────────────────────────────────────────────────────────────

INCIDENT_DURATION = 20   # each incident lasts 20 steps
INCIDENT_PROB = 0.03     # probability of a new incident starting at any step
NOISE_SCALE = 0.05       # Gaussian noise amplitude


def generate_dataset(
        n_steps: int = 10_000,
        H: int = 10,
        seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic multivariate time series with incident labels.

    Parameters
    ----------
    n_steps : total number of timesteps
    H       : prediction horizon — label[t]=1 if incident starts within H steps
    seed    : random seed

    Returns
    -------
    DataFrame with columns: [cpu_usage, memory_usage, error_rate, incident, label]
        incident : 1 during an active incident, 0 otherwise
        label    : 1 if incident starts within the next H steps
    """
    rng = np.random.default_rng(seed)

    # ── Generate incident mask ────────────────────────────────────────────────
    incident = np.zeros(n_steps, dtype=int)
    t = 0
    while t < n_steps:
        if rng.random() < INCIDENT_PROB:
            end = min(t + INCIDENT_DURATION, n_steps)
            incident[t:end] = 1
            t = end  # no overlapping incidents
        else:
            t += 1

    # ── Generate metrics ──────────────────────────────────────────────────────
    # Each metric has a baseline signal + incident-correlated spike + noise

    cpu = np.full(n_steps, 0.40)
    memory = np.full(n_steps, 0.50)
    error = np.full(n_steps, 0.02)

    # Incident effect: ramp up 5 steps before incident starts, spike at onset
    for t in range(n_steps):
        if incident[t]:
            cpu[t]    += 0.35 * min(1.0, (t - _incident_start(incident, t)) / 5 + 0.5)
            memory[t] += 0.20
            error[t]  += 0.15

    # Add noise
    cpu    += rng.normal(0, NOISE_SCALE, n_steps)
    memory += rng.normal(0, NOISE_SCALE * 0.5, n_steps)
    error  += rng.normal(0, NOISE_SCALE * 0.3, n_steps)

    # Clip to [0,1]
    cpu    = np.clip(cpu, 0, 1)
    memory = np.clip(memory, 0, 1)
    error  = np.clip(error, 0, 1)

    # ── Generate labels ───────────────────────────────────────────────────────
    # label[t] = 1 if any incident start falls within (t, t+H]
    incident_starts = np.where(np.diff(np.concatenate([[0], incident])) == 1)[0]
    label = np.zeros(n_steps, dtype=int)
    for start in incident_starts:
        # Mark the H steps leading up to this incident
        pre_start = max(0, start - H)
        label[pre_start:start] = 1

    df = pd.DataFrame({
        "cpu_usage":    cpu,
        "memory_usage": memory,
        "error_rate":   error,
        "incident":     incident,
        "label":        label,
    })
    return df


def _incident_start(incident: np.ndarray, t: int) -> int:
    """Find the start of the current incident block containing t."""
    s = t
    while s > 0 and incident[s - 1] == 1:
        s -= 1
    return s
