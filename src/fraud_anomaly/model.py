"""Model artifact loading and inference with explainable risk reasons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def load_bundle(path: str | Path) -> dict[str, Any]:
    """Load the trained model bundle from disk.

    Parameters
    ----------
    path : str or Path
        Path to the saved joblib bundle.

    Returns
    -------
    dict
        Dictionary containing trained models, thresholds, and metadata.
    """
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found at: {model_path}")

    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise ValueError(f"Expected dict bundle from {model_path}, got {type(bundle)}")

    required_keys = {
        "isolation_forest",
        "hgb_classifier",
        "feature_columns",
        "iso_score_min",
        "iso_score_max",
        "hgb_threshold",
    }
    missing = required_keys - set(bundle.keys())
    if missing:
        raise KeyError(f"Model bundle missing required keys: {sorted(missing)}")

    return bundle


def extract_reasons(features_row: pd.Series | dict[str, Any]) -> list[str]:
    """Identify human-readable risk indicators from feature values.

    Parameters
    ----------
    features_row : pd.Series or dict
        Single row of engineered features.

    Returns
    -------
    list of str
        Human-readable reason descriptions.
    """
    reasons: list[str] = []

    # Rule 1: High amount relative to customer baseline
    amount_ratio = float(features_row.get("amount_to_customer_median", 1.0))
    if amount_ratio > 3.0:
        reasons.append(
            f"Transaction amount is {amount_ratio:.1f}x higher than customer typical spend"
        )

    # Rule 2: Night hours
    is_night = int(features_row.get("is_night", 0))
    if is_night == 1:
        reasons.append("Transaction occurred during unusual night-time window (22:00 - 05:00)")

    # Rule 3: High destination country risk
    country_risk = float(features_row.get("country_risk", 0.0))
    if country_risk > 0.7:
        reasons.append(f"High destination country risk score ({country_risk:.2f})")

    # Rule 4: High merchant category risk
    merchant_risk = float(features_row.get("merchant_risk", 0.0))
    if merchant_risk > 0.7:
        reasons.append(f"High merchant category risk score ({merchant_risk:.2f})")

    # Rule 5: Cross-border + online channel combination
    is_foreign = int(features_row.get("is_foreign", 0))
    channel_online = int(features_row.get("channel_online", 0))
    if is_foreign == 1 and channel_online == 1:
        reasons.append("Cross-border transaction conducted via digital/online channel")

    return reasons


def score(bundle: dict[str, Any], features_df: pd.DataFrame) -> dict[str, Any]:
    """Score a single transaction feature row against both models.

    Parameters
    ----------
    bundle : dict
        Loaded model bundle.
    features_df : pd.DataFrame
        Single-row feature DataFrame.

    Returns
    -------
    dict
        Scoring result containing anomaly_score, fraud_probability,
        is_fraud, reasons, and model_version.
    """
    feature_cols = bundle["feature_columns"]
    X = features_df[feature_cols]

    iso_model = bundle["isolation_forest"]
    hgb_model = bundle["hgb_classifier"]

    # Isolation Forest anomaly score: raw score inverted, then min-max normalized
    raw_iso_score = -iso_model.score_samples(X)[0]
    iso_min = float(bundle.get("iso_score_min", 0.0))
    iso_max = float(bundle.get("iso_score_max", 1.0))
    iso_range = max(iso_max - iso_min, 1e-9)
    norm_iso_score = float(np.clip((raw_iso_score - iso_min) / iso_range, 0.0, 1.0))

    # HistGradientBoosting fraud probability
    fraud_prob = float(hgb_model.predict_proba(X)[:, 1][0])

    threshold = float(bundle.get("hgb_threshold", 0.5))
    is_fraud = bool(fraud_prob >= threshold)

    reasons = extract_reasons(X.iloc[0])

    return {
        "fraud_probability": round(fraud_prob, 4),
        "anomaly_score": round(norm_iso_score, 4),
        "is_fraud": is_fraud,
        "reasons": reasons,
        "model_version": bundle.get("version", "1.0.0"),
    }


def score_batch(
    bundle: dict[str, Any],
    features_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Score multiple transaction records efficiently.

    Parameters
    ----------
    bundle : dict
        Loaded model bundle.
    features_df : pd.DataFrame
        Multi-row feature DataFrame.

    Returns
    -------
    list of dict
        List of scoring results.
    """
    feature_cols = bundle["feature_columns"]
    X = features_df[feature_cols].reset_index(drop=True)

    iso_model = bundle["isolation_forest"]
    hgb_model = bundle["hgb_classifier"]

    raw_iso_scores = -iso_model.score_samples(X)
    iso_min = float(bundle.get("iso_score_min", 0.0))
    iso_max = float(bundle.get("iso_score_max", 1.0))
    iso_range = max(iso_max - iso_min, 1e-9)
    norm_iso_scores = np.clip((raw_iso_scores - iso_min) / iso_range, 0.0, 1.0)

    fraud_probs = hgb_model.predict_proba(X)[:, 1]
    threshold = float(bundle.get("hgb_threshold", 0.5))
    version = bundle.get("version", "1.0.0")

    results: list[dict[str, Any]] = []
    for idx in range(len(X)):
        prob = float(fraud_probs[idx])
        norm_score = float(norm_iso_scores[idx])
        is_fraud = bool(prob >= threshold)
        reasons = extract_reasons(X.iloc[idx])
        results.append(
            {
                "fraud_probability": round(prob, 4),
                "anomaly_score": round(norm_score, 4),
                "is_fraud": is_fraud,
                "reasons": reasons,
                "model_version": version,
            }
        )

    return results
