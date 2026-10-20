"""Feature engineering for transaction fraud and anomaly detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

FEATURE_COLUMNS: list[str] = [
    "log_amount",
    "amount_to_customer_median",
    "hour_sin",
    "hour_cos",
    "is_night",
    "day_of_week",
    "channel_pos",
    "channel_online",
    "channel_atm",
    "country_risk",
    "merchant_risk",
    "is_foreign",
]

NIGHT_HOURS: set[int] = {22, 23, 0, 1, 2, 3, 4, 5}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw transaction records into model-ready numeric features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transactions DataFrame containing columns:
        amount, customer_id, hour, day_of_week, channel, country_risk,
        merchant_risk, is_foreign.

    Returns
    -------
    pd.DataFrame
        Engineered feature matrix with columns matching FEATURE_COLUMNS.
    """
    # Safe log-transform of amount
    amounts = df["amount"].astype(float).clip(lower=0.0)
    log_amount = np.log1p(amounts)

    # Ratio of transaction amount to customer historical median spend
    if "customer_id" in df.columns:
        cust_medians = df.groupby("customer_id")["amount"].transform("median").astype(float)
        ratio = np.where(cust_medians > 0, amounts / cust_medians, 1.0)
    else:
        ratio = np.ones(len(df), dtype=float)
    amount_to_customer_median = np.nan_to_num(ratio, nan=1.0, posinf=100.0, neginf=1.0)

    # Cyclical hour encoding
    hours = df["hour"].astype(float)
    hour_sin = np.sin(2.0 * np.pi * hours / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hours / 24.0)
    is_night = df["hour"].astype(int).isin(NIGHT_HOURS).astype(int)

    day_of_week = df["day_of_week"].astype(int)

    # Channel one-hot encoding
    channel_series = df["channel"].astype(str).str.lower()
    channel_pos = (channel_series == "pos").astype(int)
    channel_online = (channel_series == "online").astype(int)
    channel_atm = (channel_series == "atm").astype(int)

    country_risk = df["country_risk"].astype(float)
    merchant_risk = df["merchant_risk"].astype(float)
    is_foreign = df["is_foreign"].astype(int)

    features = pd.DataFrame(
        {
            "log_amount": log_amount,
            "amount_to_customer_median": amount_to_customer_median,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "is_night": is_night,
            "day_of_week": day_of_week,
            "channel_pos": channel_pos,
            "channel_online": channel_online,
            "channel_atm": channel_atm,
            "country_risk": country_risk,
            "merchant_risk": merchant_risk,
            "is_foreign": is_foreign,
        },
        columns=FEATURE_COLUMNS,
    )

    return features


def features_for_single(
    tx: dict[str, Any],
    customer_median: float | None = None,
) -> pd.DataFrame:
    """Build a single-row feature DataFrame for API inference.

    If customer_median is missing, the ratio defaults to 1.0.

    Parameters
    ----------
    tx : dict
        Transaction dictionary with keys: amount, hour, day_of_week,
        channel, country_risk, merchant_risk, is_foreign, and optionally
        customer_median_amount.
    customer_median : float or None, optional
        Explicit customer median amount if known.

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame with columns matching FEATURE_COLUMNS.
    """
    amount = float(tx["amount"])

    median = customer_median
    if median is None:
        median = tx.get("customer_median_amount")

    if median is None or float(median) <= 0.0:
        amount_ratio = 1.0
    else:
        amount_ratio = float(amount / float(median))

    hour = int(tx["hour"])
    day_of_week = int(tx["day_of_week"])
    channel = str(tx["channel"]).strip().lower()

    row = {
        "log_amount": float(np.log1p(max(0.0, amount))),
        "amount_to_customer_median": float(np.clip(amount_ratio, 0.01, 1000.0)),
        "hour_sin": float(np.sin(2.0 * np.pi * hour / 24.0)),
        "hour_cos": float(np.cos(2.0 * np.pi * hour / 24.0)),
        "is_night": 1 if hour in NIGHT_HOURS else 0,
        "day_of_week": day_of_week,
        "channel_pos": 1 if channel == "pos" else 0,
        "channel_online": 1 if channel == "online" else 0,
        "channel_atm": 1 if channel == "atm" else 0,
        "country_risk": float(tx["country_risk"]),
        "merchant_risk": float(tx["merchant_risk"]),
        "is_foreign": 1 if bool(tx["is_foreign"]) else 0,
    }

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
