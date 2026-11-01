"""The training objective, as a NumPy reference: label-smoothed cross-entropy plus a
Brier term on the option probabilities, per question, averaged over questions and
examples. Cloudflare describes training Clef's head and adapters with exactly this pair
(cross-entropy with label smoothing, plus Brier "to refine probability calibration").
train.py computes the same loss in torch, and a test checks the two agree.
"""

from __future__ import annotations

import numpy as np


def log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def smoothed_targets(n_options: int, target: np.ndarray, smoothing: float) -> np.ndarray:
    """One-hot targets with `smoothing` spread evenly over the other options."""
    if not 0 <= smoothing < 1:
        raise ValueError("smoothing must be in [0, 1)")
    out = np.full((len(target), n_options), smoothing / max(1, n_options - 1))
    out[np.arange(len(target)), target] = 1 - smoothing
    return out


def cross_entropy(logits: np.ndarray, target: np.ndarray, smoothing: float = 0.0) -> float:
    q = smoothed_targets(logits.shape[-1], target, smoothing)
    return float(-(q * log_softmax(logits)).sum(axis=-1).mean())


def brier(logits: np.ndarray, target: np.ndarray) -> float:
    p = np.exp(log_softmax(logits))
    y = np.zeros_like(p)
    y[np.arange(len(target)), target] = 1
    return float(((p - y) ** 2).sum(axis=-1).mean())


def objective(logits: np.ndarray, target: np.ndarray, smoothing: float, brier_weight: float) -> float:
    return cross_entropy(logits, target, smoothing) + brier_weight * brier(logits, target)


def majority_baseline(train_targets: np.ndarray, heldout_targets: np.ndarray, n_options: int) -> dict[str, float]:
    """Brier and accuracy of always predicting the training label frequencies: the bar a tune must clear."""
    freq = np.bincount(train_targets, minlength=n_options).astype(float)
    freq /= freq.sum()
    logits = np.log(np.clip(np.tile(freq, (len(heldout_targets), 1)), 1e-12, 1))
    return {
        "brier": brier(logits, heldout_targets),
        "accuracy": float((heldout_targets == int(freq.argmax())).mean()),
    }
