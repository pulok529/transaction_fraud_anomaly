"""FastAPI application for transaction fraud and anomaly detection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from fraud_anomaly.features import features_for_single
from fraud_anomaly.model import load_bundle, score, score_batch


class Transaction(BaseModel):
    """Input schema for a transaction to be evaluated."""

    amount: float = Field(..., gt=0.0, description="Transaction amount in currency units")
    hour: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of the week (0=Monday, 6=Sunday)")
    channel: Literal["pos", "online", "atm"] = Field(
        ..., description="Transaction channel: 'pos', 'online', or 'atm'"
    )
    country_risk: float = Field(
        ..., ge=0.0, le=1.0, description="Risk index of transaction country (0 to 1)"
    )
    merchant_risk: float = Field(
        ..., ge=0.0, le=1.0, description="Risk index of merchant category (0 to 1)"
    )
    is_foreign: bool = Field(
        ..., description="Whether the transaction was executed abroad/foreign"
    )
    customer_median_amount: float | None = Field(
        default=None,
        description="Historical median spend amount for this customer, if available",
    )


class ScoreResponse(BaseModel):
    """Scoring prediction and explainability response."""

    fraud_probability: float = Field(..., description="HistGradientBoosting fraud probability")
    anomaly_score: float = Field(
        ..., description="Isolation Forest anomaly score normalized to 0-1"
    )
    is_fraud: bool = Field(
        ..., description="True if fraud_probability exceeds optimal threshold"
    )
    reasons: list[str] = Field(..., description="Top human-readable risk indicators")
    model_version: str = Field(..., description="Semantic version of model bundle")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_version: str


def create_app(model_path: str | Path | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to the saved model bundle. Defaults to env MODEL_PATH or 'artifacts/model.joblib'.
        If the file does not exist, a model is trained automatically (n=20000).

    Returns
    -------
    FastAPI
        Configured FastAPI instance.
    """
    if model_path is None:
        model_path = os.getenv("MODEL_PATH", "artifacts/model.joblib")

    path = Path(model_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        from fraud_anomaly.train import train

        train(n=20000, out_dir=path.parent)

    bundle = load_bundle(path)

    app = FastAPI(
        title="Transaction Fraud & Anomaly Detection API",
        description=(
            "Dual-model fraud inference system combining unsupervised Isolation Forest "
            "and supervised HistGradientBoosting with human-readable risk reasoning."
        ),
        version=bundle.get("version", "1.0.0"),
    )

    app.state.bundle = bundle

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> dict[str, str]:
        """Health check endpoint returning service status and active model version."""
        active_bundle: dict[str, Any] = request.app.state.bundle
        return {
            "status": "ok",
            "model_version": active_bundle.get("version", "1.0.0"),
        }

    @app.post("/score", response_model=ScoreResponse)
    def score_single(tx: Transaction, request: Request) -> dict[str, Any]:
        """Score a single transaction for fraud probability and structural anomaly."""
        active_bundle: dict[str, Any] = request.app.state.bundle
        features_df = features_for_single(tx.model_dump())
        return score(active_bundle, features_df)

    @app.post("/score/batch", response_model=list[ScoreResponse])
    def score_batch_endpoint(
        txs: list[Transaction], request: Request
    ) -> list[dict[str, Any]]:
        """Score a batch of transactions (maximum 1000 items)."""
        if len(txs) > 1000:
            raise HTTPException(
                status_code=422,
                detail=f"Batch size {len(txs)} exceeds maximum limit of 1000 items",
            )
        if len(txs) == 0:
            return []

        active_bundle: dict[str, Any] = request.app.state.bundle
        features_list = [features_for_single(t.model_dump()) for t in txs]
        features_df = pd.concat(features_list, ignore_index=True)
        return score_batch(active_bundle, features_df)

    return app


# Lazy application loader for uvicorn and direct imports
_app: FastAPI | None = None


def get_app() -> FastAPI:
    """Return or initialize global application instance."""
    global _app
    if _app is None:
        _app = create_app()
    return _app


def __getattr__(name: str) -> Any:
    """Provide lazy attribute access for 'app' to avoid training during imports."""
    if name == "app":
        return get_app()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

