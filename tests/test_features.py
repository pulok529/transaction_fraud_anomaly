"""Unit tests for feature engineering transformations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_anomaly.data import generate
from fraud_anomaly.features import FEATURE_COLUMNS, build_features, features_for_single


def test_build_features_columns_and_types():
    """Verify built features match FEATURE_COLUMNS and contain numeric values."""
    df = generate(n=500, fraud_rate=0.02, seed=42)
    features = build_features(df)

    assert list(features.columns) == FEATURE_COLUMNS
    assert len(features) == 500
    assert not features.isnull().any().any()
    assert np.all(np.isfinite(features.to_numpy()))


def test_build_features_trigonometric_and_night():
    """Verify sine/cosine and night indicator properties."""
    df = pd.DataFrame(
        {
            "amount": [50.0, 100.0, 20.0],
            "customer_id": [1, 1, 2],
            "hour": [0, 12, 23],
            "day_of_week": [0, 2, 6],
            "channel": ["pos", "online", "atm"],
            "country_risk": [0.1, 0.4, 0.9],
            "merchant_risk": [0.2, 0.5, 0.8],
            "is_foreign": [0, 1, 1],
        }
    )
    features = build_features(df)

    assert (features["hour_sin"] >= -1.0).all() and (features["hour_sin"] <= 1.0).all()
    assert (features["hour_cos"] >= -1.0).all() and (features["hour_cos"] <= 1.0).all()

    # Hour 0 and 23 should be night (1), Hour 12 should be day (0)
    assert features["is_night"].tolist() == [1, 0, 1]


def test_build_features_amount_to_customer_median():
    """Verify calculation of transaction amount relative to customer historical median."""
    df = pd.DataFrame(
        {
            "amount": [100.0, 200.0, 300.0],
            "customer_id": [999, 999, 999],  # median = 200.0
            "hour": [10, 11, 12],
            "day_of_week": [1, 1, 1],
            "channel": ["pos", "pos", "pos"],
            "country_risk": [0.1, 0.1, 0.1],
            "merchant_risk": [0.1, 0.1, 0.1],
            "is_foreign": [0, 0, 0],
        }
    )
    features = build_features(df)

    expected_ratios = [0.5, 1.0, 1.5]
    np.testing.assert_allclose(
        features["amount_to_customer_median"].tolist(),
        expected_ratios,
        rtol=1e-5,
    )


def test_features_for_single_missing_median():
    """Verify single transaction feature extraction defaults ratio to 1.0 when median missing."""
    tx = {
        "amount": 250.0,
        "hour": 3,
        "day_of_week": 4,
        "channel": "online",
        "country_risk": 0.85,
        "merchant_risk": 0.75,
        "is_foreign": True,
    }
    feats = features_for_single(tx)

    assert list(feats.columns) == FEATURE_COLUMNS
    assert len(feats) == 1
    assert feats.loc[0, "amount_to_customer_median"] == 1.0
    assert feats.loc[0, "is_night"] == 1
    assert feats.loc[0, "channel_online"] == 1
    assert feats.loc[0, "channel_pos"] == 0
    assert feats.loc[0, "is_foreign"] == 1


def test_features_for_single_with_median():
    """Verify single transaction feature extraction computes correct amount ratio."""
    tx = {
        "amount": 300.0,
        "hour": 14,
        "day_of_week": 2,
        "channel": "pos",
        "country_risk": 0.1,
        "merchant_risk": 0.2,
        "is_foreign": False,
        "customer_median_amount": 100.0,
    }
    feats = features_for_single(tx)

    assert feats.loc[0, "amount_to_customer_median"] == 3.0
    assert feats.loc[0, "is_night"] == 0
    assert feats.loc[0, "channel_pos"] == 1
    assert feats.loc[0, "is_foreign"] == 0
