"""Synthetic transaction dataset generation for fraud and anomaly detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate(
    n: int = 50000,
    fraud_rate: float = 0.02,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a realistic synthetic transaction dataset.

    Parameters
    ----------
    n : int
        Total number of transactions to generate.
    fraud_rate : float
        Fraction of transactions marked as fraudulent anomalies.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: transaction_id, customer_id, timestamp,
        amount, hour, day_of_week, channel, country_risk, merchant_risk,
        is_foreign, label.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if not (0.0 <= fraud_rate <= 1.0):
        raise ValueError("fraud_rate must be between 0.0 and 1.0")

    rng = np.random.default_rng(seed)

    # Establish customer pool (~5000 customers for n=50000)
    n_customers = min(5000, max(50, n // 10))
    customer_ids = np.arange(1000, 1000 + n_customers, dtype=int)
    # Each customer has a personal typical spend baseline (median spend)
    customer_typical_spend = rng.lognormal(mean=3.8, sigma=0.6, size=n_customers).clip(10.0, 2500.0)

    # Sample customer assignments for each transaction
    assigned_cust_indices = rng.integers(0, n_customers, size=n)
    assigned_customers = customer_ids[assigned_cust_indices]
    assigned_medians = customer_typical_spend[assigned_cust_indices]

    # Spread over 90 days starting from 2026-01-01
    start_dt = pd.Timestamp("2026-01-01 00:00:00")
    day_offsets = rng.integers(0, 90, size=n)
    minute_offsets = rng.integers(0, 60, size=n)
    second_offsets = rng.integers(0, 60, size=n)

    # Normal daytime-weighted transaction hours
    hour_probs = np.array(
        [
            0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 00:00 - 05:00
            0.03, 0.04, 0.06, 0.07, 0.08, 0.08,  # 06:00 - 11:00
            0.08, 0.07, 0.07, 0.07, 0.06, 0.06,  # 12:00 - 17:00
            0.05, 0.05, 0.04, 0.03, 0.02, 0.01,  # 18:00 - 23:00
        ],
        dtype=float,
    )
    hour_probs /= hour_probs.sum()
    hours = rng.choice(24, size=n, p=hour_probs)

    # Channels: pos (65%), online (25%), atm (10%)
    channels = rng.choice(["pos", "online", "atm"], size=n, p=[0.65, 0.25, 0.10])

    # Domestic vs foreign: mostly domestic (95%)
    is_foreign = rng.choice([0, 1], size=n, p=[0.95, 0.05])

    # Low baseline country and merchant risk
    country_risk = rng.beta(2.0, 8.0, size=n)
    merchant_risk = rng.beta(2.0, 8.0, size=n)

    # Baseline transaction amounts centered on customer median
    amount_multiplier = rng.lognormal(mean=0.0, sigma=0.35, size=n)
    amounts = np.round(assigned_medians * amount_multiplier, 2).clip(1.0, 25000.0)

    # Inject fraud anomalies
    n_fraud = int(round(n * fraud_rate))
    fraud_indices = rng.choice(n, size=n_fraud, replace=False)
    labels = np.zeros(n, dtype=int)
    labels[fraud_indices] = 1

    # Legitimate noise: some genuine customers also buy big, shop late, travel or use
    # riskier merchants, so no single signal separates fraud from normal behaviour.
    big_spend = rng.random(n) < 0.03
    amounts[big_spend] = np.round(amounts[big_spend] * rng.uniform(2.0, 6.0, big_spend.sum()), 2)
    night_owl = rng.random(n) < 0.06
    hours[night_owl] = rng.choice([22, 23, 0, 1, 2, 3], size=night_owl.sum())
    travel = rng.random(n) < 0.03
    is_foreign[travel] = 1
    country_risk[travel] = rng.beta(4.0, 4.0, size=travel.sum())
    risky_merchant = rng.random(n) < 0.05
    merchant_risk[risky_merchant] = rng.beta(4.0, 3.0, size=risky_merchant.sum())

    if n_fraud > 0:
        # Three fraud archetypes, each shifting only some signals:
        # 0 = account takeover (large amount, night, online)
        # 1 = card-not-present abroad (foreign, risky country, moderate amount)
        # 2 = low-and-slow testing (small amounts at risky merchants, otherwise normal)
        archetype = rng.choice(3, size=n_fraud, p=[0.4, 0.35, 0.25])
        medians = assigned_medians[fraud_indices]

        mult = np.where(
            archetype == 0,
            rng.lognormal(mean=1.0, sigma=0.6, size=n_fraud),
            np.where(
                archetype == 1,
                rng.lognormal(mean=0.5, sigma=0.5, size=n_fraud),
                rng.uniform(0.1, 0.6, size=n_fraud),
            ),
        )
        amounts[fraud_indices] = np.round((medians * mult).clip(1.0, 60000.0), 2)

        night = (archetype == 0) & (rng.random(n_fraud) < 0.6)
        hours[fraud_indices[night]] = rng.choice([22, 23, 0, 1, 2, 3, 4], size=night.sum())

        online = (archetype != 2) & (rng.random(n_fraud) < 0.6)
        channels[fraud_indices[online]] = "online"

        abroad = (archetype == 1) & (rng.random(n_fraud) < 0.7)
        is_foreign[fraud_indices[abroad]] = 1
        cnp, testing = fraud_indices[archetype == 1], fraud_indices[archetype == 2]
        country_risk[cnp] = rng.beta(4.0, 3.0, size=len(cnp))
        merchant_risk[testing] = rng.beta(4.0, 3.0, size=len(testing))

    # Construct coherent timestamps
    raw_timestamps = (
        start_dt
        + pd.to_timedelta(day_offsets, unit="D")
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(minute_offsets, unit="m")
        + pd.to_timedelta(second_offsets, unit="s")
    )
    days_of_week = np.asarray(raw_timestamps.dayofweek)
    hours_final = np.asarray(raw_timestamps.hour)

    # Sort deterministically by timestamp for chronological realism
    sort_order = np.argsort(raw_timestamps.to_numpy())
    sorted_timestamps = pd.Series(raw_timestamps[sort_order], name="timestamp")

    df = pd.DataFrame(
        {
            "transaction_id": [f"TX{idx + 1:08d}" for idx in range(n)],
            "customer_id": assigned_customers[sort_order],
            "timestamp": sorted_timestamps,
            "amount": amounts[sort_order],
            "hour": hours_final[sort_order],
            "day_of_week": days_of_week[sort_order],
            "channel": channels[sort_order],
            "country_risk": np.round(country_risk[sort_order], 4),
            "merchant_risk": np.round(merchant_risk[sort_order], 4),
            "is_foreign": is_foreign[sort_order],
            "label": labels[sort_order],
        }
    )

    return df


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Save the generated transactions DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of transactions.
    path : str or Path
        Target CSV file path.

    Returns
    -------
    Path
        Resolved path where the CSV was saved.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
