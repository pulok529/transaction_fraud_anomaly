"""Training pipeline for fraud and anomaly detection models."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraud_anomaly.data import generate
from fraud_anomaly.features import FEATURE_COLUMNS, build_features


def select_threshold(
    y_true: np.ndarray,
    scores_or_probs: np.ndarray,
    target_precision: float = 0.80,
) -> float:
    """Find decision threshold on training set meeting target precision or best F1.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels (0/1).
    scores_or_probs : np.ndarray
        Model continuous risk scores or predicted probabilities.
    target_precision : float, default=0.80
        Target minimum precision on training set.

    Returns
    -------
    float
        Selected optimal decision threshold.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores_or_probs)

    if len(thresholds) == 0:
        return 0.5

    p_valid = precisions[:-1]
    r_valid = recalls[:-1]

    # Thresholds meeting target precision
    qualifying_indices = np.where(p_valid >= target_precision)[0]
    if len(qualifying_indices) > 0:
        # Choose the threshold that maximizes recall among qualifying candidates
        best_qualifying_idx = qualifying_indices[np.argmax(r_valid[qualifying_indices])]
        return float(thresholds[best_qualifying_idx])

    # Fallback to threshold that maximizes F1 score
    denom = p_valid + r_valid
    f1_scores = np.where(denom > 0, 2 * p_valid * r_valid / (denom + 1e-9), 0.0)
    best_f1_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_f1_idx])


def train(
    n: int = 50000,
    out_dir: str | Path = "artifacts",
    fraud_rate: float = 0.02,
    seed: int = 7,
) -> dict[str, Any]:
    """Execute complete end-to-end data generation, training, and evaluation pipeline.

    Parameters
    ----------
    n : int, default=50000
        Number of synthetic transactions to generate.
    out_dir : str or Path, default="artifacts"
        Directory to store trained model bundle, plots, and metrics.
    fraud_rate : float, default=0.02
        Proportion of anomalous fraudulent transactions.
    seed : int, default=7
        Random seed for reproducibility.

    Returns
    -------
    dict
        Evaluation metrics on the test partition.
    """
    artifacts_dir = Path(out_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate synthetic transactions
    df = generate(n=n, fraud_rate=fraud_rate, seed=seed)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 2. Strict time-based split: first 80% time window -> train, last 20% -> test
    min_time = df["timestamp"].min()
    max_time = df["timestamp"].max()
    split_cutoff = min_time + 0.80 * (max_time - min_time)

    train_mask = df["timestamp"] < split_cutoff
    test_mask = ~train_mask

    train_df = df[train_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    # 3. Build features
    X_train = build_features(train_df)
    y_train = train_df["label"].to_numpy(dtype=int)
    X_test = build_features(test_df)
    y_test = test_df["label"].to_numpy(dtype=int)

    # 4. Hold out the most recent 20% of the training window to choose thresholds,
    #    so the decision cut-off is never tuned on data the model was fitted on.
    n_fit = int(len(X_train) * 0.8)
    X_fit, y_fit = X_train.iloc[:n_fit], y_train[:n_fit]
    X_val, y_val = X_train.iloc[n_fit:], y_train[n_fit:]

    # 5. Unsupervised model: Isolation Forest (labels unused)
    iso_forest = IsolationForest(
        contamination=fraud_rate,
        random_state=seed,
        n_estimators=150,
        n_jobs=-1,
    )
    iso_forest.fit(X_fit)

    train_iso_scores = -iso_forest.score_samples(X_fit)
    val_iso_scores = -iso_forest.score_samples(X_val)
    test_iso_scores = -iso_forest.score_samples(X_test)
    iso_score_min = float(train_iso_scores.min())
    iso_score_max = float(train_iso_scores.max())

    # 6. Supervised model: HistGradientBoostingClassifier
    hgb = HistGradientBoostingClassifier(
        class_weight="balanced",
        random_state=seed,
        max_iter=150,
    )
    hgb.fit(X_fit, y_fit)

    val_hgb_probs = hgb.predict_proba(X_val)[:, 1]
    test_hgb_probs = hgb.predict_proba(X_test)[:, 1]

    # Thresholds: highest recall at precision >= 0.80 on validation, else best F1
    iso_threshold = select_threshold(y_val, val_iso_scores, target_precision=0.80)
    hgb_threshold = select_threshold(y_val, val_hgb_probs, target_precision=0.80)

    # 7. Evaluate on TEST partition
    test_iso_preds = (test_iso_scores >= iso_threshold).astype(int)
    test_hgb_preds = (test_hgb_probs >= hgb_threshold).astype(int)

    metrics = {
        "isolation_forest": {
            "roc_auc": round(float(roc_auc_score(y_test, test_iso_scores)), 4),
            "pr_auc": round(float(average_precision_score(y_test, test_iso_scores)), 4),
            "threshold": round(float(iso_threshold), 4),
            "precision": round(
                float(precision_score(y_test, test_iso_preds, zero_division=0)), 4
            ),
            "recall": round(float(recall_score(y_test, test_iso_preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, test_iso_preds, zero_division=0)), 4),
        },
        "hist_gradient_boosting": {
            "roc_auc": round(float(roc_auc_score(y_test, test_hgb_probs)), 4),
            "pr_auc": round(float(average_precision_score(y_test, test_hgb_probs)), 4),
            "threshold": round(float(hgb_threshold), 4),
            "precision": round(
                float(precision_score(y_test, test_hgb_preds, zero_division=0)), 4
            ),
            "recall": round(float(recall_score(y_test, test_hgb_preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, test_hgb_preds, zero_division=0)), 4),
        },
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "train_fraud_rate": round(float(np.mean(y_train)), 4),
        "test_fraud_rate": round(float(np.mean(y_test)), 4),
    }

    # 8. Save model bundle
    bundle = {
        "isolation_forest": iso_forest,
        "hgb_classifier": hgb,
        "feature_columns": FEATURE_COLUMNS,
        "iso_score_min": iso_score_min,
        "iso_score_max": iso_score_max,
        "iso_threshold": iso_threshold,
        "hgb_threshold": hgb_threshold,
        "version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_size": len(train_df),
        "test_size": len(test_df),
    }
    model_path = artifacts_dir / "model.joblib"
    joblib.dump(bundle, model_path)

    # 9. Save metrics.json
    metrics_path = artifacts_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 10. Generate visualization plots (Agg backend)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # PR Curve Plot
    plt.figure(figsize=(8, 6))
    p_hgb, r_hgb, _ = precision_recall_curve(y_test, test_hgb_probs)
    p_iso, r_iso, _ = precision_recall_curve(y_test, test_iso_scores)
    baseline = float(np.mean(y_test))

    plt.plot(
        r_hgb,
        p_hgb,
        label=f"HistGradientBoosting (PR-AUC = {metrics['hist_gradient_boosting']['pr_auc']:.3f})",
        color="#2563eb",
        linewidth=2.2,
    )
    plt.plot(
        r_iso,
        p_iso,
        label=f"Isolation Forest (PR-AUC = {metrics['isolation_forest']['pr_auc']:.3f})",
        color="#dc2626",
        linewidth=2.0,
        linestyle="--",
    )
    plt.axhline(
        baseline,
        color="#6b7280",
        linestyle=":",
        label=f"Baseline Prevalence ({baseline:.3f})",
    )
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall Curve on Test Set", fontsize=13, fontweight="bold")
    plt.legend(loc="lower left", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(artifacts_dir / "pr_curve.png", dpi=300)
    plt.close()

    # Score Histogram Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Isolation Forest distribution
    axes[0].hist(
        test_iso_scores[y_test == 0],
        bins=45,
        alpha=0.6,
        label="Legitimate (0)",
        color="#2563eb",
        density=True,
    )
    axes[0].hist(
        test_iso_scores[y_test == 1],
        bins=45,
        alpha=0.6,
        label="Fraudulent (1)",
        color="#dc2626",
        density=True,
    )
    axes[0].axvline(
        iso_threshold,
        color="black",
        linestyle="--",
        label=f"Threshold ({iso_threshold:.3f})",
    )
    axes[0].set_title("Isolation Forest Anomaly Scores", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Anomaly Score (-score_samples)")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.3)

    # HistGradientBoosting distribution
    axes[1].hist(
        test_hgb_probs[y_test == 0],
        bins=45,
        alpha=0.6,
        label="Legitimate (0)",
        color="#2563eb",
        density=True,
    )
    axes[1].hist(
        test_hgb_probs[y_test == 1],
        bins=45,
        alpha=0.6,
        label="Fraudulent (1)",
        color="#dc2626",
        density=True,
    )
    axes[1].axvline(
        hgb_threshold,
        color="black",
        linestyle="--",
        label=f"Threshold ({hgb_threshold:.3f})",
    )
    axes[1].set_title(
        "HistGradientBoosting Fraud Probability", fontsize=12, fontweight="bold"
    )
    axes[1].set_xlabel("Predicted Probability")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(artifacts_dir / "score_hist.png", dpi=300)
    plt.close()

    # 11. Print Markdown Summary Table
    iso_m = metrics["isolation_forest"]
    hgb_m = metrics["hist_gradient_boosting"]
    keys = ("roc_auc", "pr_auc", "precision", "recall", "f1", "threshold")

    def row(name: str, m: dict) -> str:
        return f"| {name} | " + " | ".join(f"{m[k]:.4f}" for k in keys) + " |"

    table = "\n".join(
        [
            "### Model Evaluation Summary (Test Set)",
            "",
            "| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Threshold (validation-tuned) |",
            "|---|---|---|---|---|---|---|",
            row("Isolation Forest (Unsupervised)", iso_m),
            row("HistGradientBoosting (Supervised)", hgb_m),
        ]
    )
    print(table)

    return metrics


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Train transaction fraud and anomaly detection models."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50000,
        help="Number of synthetic transactions to generate (default: 50000).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts/",
        help="Output directory for model artifacts and metrics (default: artifacts/).",
    )
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=0.02,
        help="Proportion of synthetic fraud cases (default: 0.02).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for data generation and training (default: 7).",
    )
    args = parser.parse_args()
    train(n=args.n, out_dir=args.out, fraud_rate=args.fraud_rate, seed=args.seed)


if __name__ == "__main__":
    main()
