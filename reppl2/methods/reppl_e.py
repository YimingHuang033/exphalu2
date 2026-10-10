"""E: nonnegative supervised calibration of propagation uncertainty ONLY.

No raw hidden state, Outer, length, confidence or correctness feature enters X.
"""
from __future__ import annotations

import numpy as np
from ..cache import content_hash

FEATURES = {"d": ("d_mean", "d_max", "d_top2"),
            "cd": ("c_mean", "c_max", "c_top2", "d_mean", "d_max", "d_top2")}


def uncertainty_features(c, d, mode="cd"):
    if mode not in FEATURES:
        raise ValueError("E mode must be d or cd")
    parts = [d] if mode == "d" else [c, d]
    values = []
    for part in parts:
        if not part or part.get("validity") != "ok":
            return None
        a = np.asarray(part["components"], dtype=np.float64)
        if a.ndim != 1 or not a.size or not np.isfinite(a).all() or (a < 0).any():
            raise ValueError("invalid uncertainty components")
        values.extend([a.mean(), a.max(), np.sort(a)[-2:].mean()])
    return np.asarray(values)


def fit_calibrator(X, y, regularization, max_iter=2000):
    from scipy.optimize import minimize
    from scipy.special import expit

    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if (X.ndim != 2 or y.shape != (len(X),) or not np.isfinite(X).all()
            or (X < 0).any() or set(np.unique(y)) != {0.0, 1.0}):
        raise ValueError("finite nonnegative features and both binary classes required")
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be positive")
    scale = np.sqrt(np.mean(X ** 2, axis=0))
    scale[scale < 1e-12] = 1.0
    Z = X / scale

    def objective(theta):
        w, b = theta[:-1], theta[-1]
        logits = Z @ w + b
        residual = expit(logits) - y
        loss = np.mean(np.logaddexp(0, logits) - y * logits)
        loss += regularization * np.dot(w, w) / 2
        grad = np.r_[Z.T @ residual / len(y) + regularization * w, residual.mean()]
        return loss, grad

    initial = np.zeros(X.shape[1] + 1)
    initial[-1] = np.log(y.mean() / (1 - y.mean()))
    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      bounds=[(0, None)] * X.shape[1] + [(None, None)],
                      options={"maxiter": max_iter})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"E optimizer failed: {result.message}")
    return {"scale": scale.tolist(), "weights": result.x[:-1].tolist(),
            "intercept": float(result.x[-1]), "regularization": regularization,
            "zero_weights": bool(np.max(result.x[:-1]) < 1e-8)}


def predict_calibrator(model, X):
    from scipy.special import expit
    X = np.asarray(X, dtype=float)
    scale, weights = np.asarray(model["scale"]), np.asarray(model["weights"])
    if (X.ndim != 2 or X.shape[1] != len(weights) or (X < 0).any()
            or not np.isfinite(X).all() or (weights < 0).any() or (scale <= 0).any()
            or scale.shape != weights.shape or not np.isfinite(weights).all()
            or not np.isfinite(scale).all() or not np.isfinite(model["intercept"])):
        raise ValueError("invalid E input/model")
    contributions = X / scale * weights
    inner = contributions.sum(axis=1)
    return inner, expit(model["intercept"] + inner), contributions


def grouped_oof(X, y, groups, sample_ids, mode, config):
    """Nested grouped CV. Every scaler/weight/hyperparameter excludes its test group."""
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import log_loss

    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    if mode not in FEATURES or len(sample_ids) != len(y):
        raise ValueError("E mode or sample IDs mismatch")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("duplicate sample_ids in E")
    if X.shape != (len(y), len(FEATURES[mode])) or len(groups) != len(y):
        raise ValueError("E feature schema mismatch")
    n_outer, n_inner, seed = config["outer_folds"], config["inner_folds"], config["seed"]
    if n_outer < 2 or n_inner < 2 or not config["regularizations"]:
        raise ValueError("E requires >=2 folds and regularization candidates")
    if len(set(groups)) < n_outer:
        raise ValueError("not enough groups for outer CV")
    outer = StratifiedGroupKFold(n_outer, shuffle=True, random_state=seed)
    predictions, models = [], []
    for fold, (train, test) in enumerate(outer.split(X, y, groups)):
        if len(set(groups[train])) < n_inner:
            raise ValueError("not enough training groups for inner CV")
        inner = StratifiedGroupKFold(n_inner, shuffle=True, random_state=seed + fold + 1)
        splits = list(inner.split(X[train], y[train], groups[train]))
        losses = []
        for reg in config["regularizations"]:
            fold_losses = []
            for tr, va in splits:
                model = fit_calibrator(X[train][tr], y[train][tr], reg, config["max_iter"])
                _, prob, _ = predict_calibrator(model, X[train][va])
                fold_losses.append(log_loss(y[train][va], prob, labels=[0, 1]))
            losses.append(float(np.mean(fold_losses)))
        reg = config["regularizations"][int(np.argmin(losses))]
        model = fit_calibrator(X[train], y[train], reg, config["max_iter"])
        model.update({"fold": fold, "feature_names": FEATURES[mode],
                      "train_ids": [sample_ids[i] for i in train],
                      "test_ids": [sample_ids[i] for i in test],
                      "train_groups": sorted(set(groups[train])),
                      "test_groups": sorted(set(groups[test])), "inner_cv_losses": losses})
        model["model_hash"] = content_hash(model)
        scores, probs, contributions = predict_calibrator(model, X[test])
        for i, score, prob, cs in zip(test, scores, probs, contributions):
            predictions.append({"sample_id": sample_ids[i], "fold": fold, "inner": float(score),
                                "calibrated_probability": float(prob), "contributions": cs.tolist(),
                                "feature_values": X[i].tolist(), "model_hash": model["model_hash"]})
        models.append(model)
    return {"mode": mode, "predictions": predictions, "fold_models": models,
            "protocol": "nested-grouped-out-of-fold; not independent held-out test"}
