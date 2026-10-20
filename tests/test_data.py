"""Unit tests for synthetic transaction generator and storage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fraud_anomaly.data import generate, save_csv


def test_generate_default_shape_and_columns():
    """Verify that generate produces the expected columns and shape."""
    df = generate(n=1000, fraud_rate=0.03, seed=42)

    expected_columns = [
        "transaction_id",
        "customer_id",
        "timestamp",
        "amount",
        "hour",
        "day_of_week",
        "channel",
        "country_risk",
        "merchant_risk",
        "is_foreign",
        "label",
    ]
    assert list(df.columns) == expected_columns
    assert len(df) == 1000
    assert not df.isnull().any().any()


def test_generate_fraud_rate_and_labels():
    """Verify that fraud rate matches the specified configuration."""
    n = 2000
    fraud_rate = 0.05
    df = generate(n=n, fraud_rate=fraud_rate, seed=123)

    num_fraud = df["label"].sum()
    expected_fraud = int(round(n * fraud_rate))
    assert num_fraud == expected_fraud
    assert set(df["label"].unique()).issubset({0, 1})


def test_generate_data_ranges_and_types():
    """Verify value ranges for synthetic transaction attributes."""
    df = generate(n=1500, fraud_rate=0.02, seed=99)

    assert (df["amount"] > 0).all()
    assert (df["hour"] >= 0).all() and (df["hour"] <= 23).all()
    assert (df["day_of_week"] >= 0).all() and (df["day_of_week"] <= 6).all()
    assert set(df["channel"].unique()).issubset({"pos", "online", "atm"})
    assert (df["country_risk"] >= 0.0).all() and (df["country_risk"] <= 1.0).all()
    assert (df["merchant_risk"] >= 0.0).all() and (df["merchant_risk"] <= 1.0).all()
    assert set(df["is_foreign"].unique()).issubset({0, 1})

    # Verify timestamps are within 90-day range
    ts = pd.to_datetime(df["timestamp"])
    time_span = (ts.max() - ts.min()).total_seconds() / 86400.0
    assert time_span <= 91.0


def test_generate_deterministic_seed():
    """Ensure identical output when using identical seed."""
    df1 = generate(n=500, fraud_rate=0.02, seed=42)
    df2 = generate(n=500, fraud_rate=0.02, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_invalid_inputs():
    """Check that invalid parameters raise ValueErrors."""
    with pytest.raises(ValueError, match="n must be a positive integer"):
        generate(n=0)

    with pytest.raises(ValueError, match="fraud_rate must be between 0.0 and 1.0"):
        generate(n=100, fraud_rate=-0.1)

    with pytest.raises(ValueError, match="fraud_rate must be between 0.0 and 1.0"):
        generate(n=100, fraud_rate=1.5)


def test_save_csv(tmp_path: Path):
    """Ensure save_csv writes valid CSV data and creates parent folders."""
    df = generate(n=100, fraud_rate=0.02, seed=1)
    target_path = tmp_path / "subfolder" / "transactions.csv"

    saved_path = save_csv(df, target_path)
    assert saved_path.exists()

    loaded = pd.read_csv(saved_path)
    assert len(loaded) == 100
    assert list(loaded.columns) == list(df.columns)
