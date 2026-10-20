"""API integration tests using FastAPI TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from fraud_anomaly.train import train


@pytest.fixture(scope="session")
def client(tmp_path_factory: pytest.TempPathFactory):
    """Session fixture that trains a quick model and returns an active TestClient."""
    tmp_dir = tmp_path_factory.mktemp("test_model_artifacts")
    train(n=5000, out_dir=tmp_dir, fraud_rate=0.03, seed=42)
    model_file = tmp_dir / "model.joblib"

    app = create_app(model_path=model_file)
    with TestClient(app) as test_client:
        yield test_client


def test_health_200(client: TestClient):
    """Verify health check returns 200 and expected payload structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_version" in data
    assert data["model_version"] == "1.0.0"


def test_score_fraud_vs_normal(client: TestClient):
    """Verify fraudulent transaction scores higher probability and anomaly score than normal tx."""
    normal_payload = {
        "amount": 28.50,
        "hour": 13,
        "day_of_week": 2,
        "channel": "pos",
        "country_risk": 0.05,
        "merchant_risk": 0.10,
        "is_foreign": False,
        "customer_median_amount": 30.00,
    }
    fraud_payload = {
        "amount": 4500.00,
        "hour": 3,
        "day_of_week": 6,
        "channel": "online",
        "country_risk": 0.95,
        "merchant_risk": 0.90,
        "is_foreign": True,
        "customer_median_amount": 40.00,
    }

    norm_res = client.post("/score", json=normal_payload)
    assert norm_res.status_code == 200
    norm_data = norm_res.json()

    fraud_res = client.post("/score", json=fraud_payload)
    assert fraud_res.status_code == 200
    fraud_data = fraud_res.json()

    # Fraud transaction should have substantially higher risk metrics
    assert fraud_data["fraud_probability"] > norm_data["fraud_probability"]
    assert fraud_data["anomaly_score"] > norm_data["anomaly_score"]
    assert fraud_data["is_fraud"] is True
    assert isinstance(fraud_data["reasons"], list)
    assert len(fraud_data["reasons"]) > 0


def test_validation_error_hour_25(client: TestClient):
    """Verify pydantic validation rejects invalid hour=25 with HTTP 422."""
    invalid_payload = {
        "amount": 100.0,
        "hour": 25,  # Out of range 0-23
        "day_of_week": 1,
        "channel": "pos",
        "country_risk": 0.2,
        "merchant_risk": 0.2,
        "is_foreign": False,
    }
    response = client.post("/score", json=invalid_payload)
    assert response.status_code == 422


def test_validation_error_negative_amount(client: TestClient):
    """Verify pydantic validation rejects amount <= 0 with HTTP 422."""
    invalid_payload = {
        "amount": -5.0,
        "hour": 12,
        "day_of_week": 1,
        "channel": "pos",
        "country_risk": 0.2,
        "merchant_risk": 0.2,
        "is_foreign": False,
    }
    response = client.post("/score", json=invalid_payload)
    assert response.status_code == 422


def test_batch_endpoint_length(client: TestClient):
    """Verify batch endpoint scores all transactions and preserves output length."""
    batch_payload = [
        {
            "amount": 15.0,
            "hour": 10,
            "day_of_week": 1,
            "channel": "pos",
            "country_risk": 0.1,
            "merchant_risk": 0.1,
            "is_foreign": False,
        },
        {
            "amount": 2500.0,
            "hour": 2,
            "day_of_week": 5,
            "channel": "online",
            "country_risk": 0.8,
            "merchant_risk": 0.9,
            "is_foreign": True,
            "customer_median_amount": 20.0,
        },
        {
            "amount": 50.0,
            "hour": 18,
            "day_of_week": 3,
            "channel": "atm",
            "country_risk": 0.2,
            "merchant_risk": 0.3,
            "is_foreign": False,
        },
    ]

    response = client.post("/score/batch", json=batch_payload)
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    assert len(results) == len(batch_payload)
    for item in results:
        assert "fraud_probability" in item
        assert "anomaly_score" in item
        assert "is_fraud" in item
        assert "reasons" in item
        assert "model_version" in item


def test_batch_endpoint_limit_exceeded(client: TestClient):
    """Verify batch endpoint rejects requests exceeding maximum 1000 items with HTTP 422."""
    single_item = {
        "amount": 25.0,
        "hour": 12,
        "day_of_week": 2,
        "channel": "pos",
        "country_risk": 0.1,
        "merchant_risk": 0.1,
        "is_foreign": False,
    }
    oversized_batch = [single_item] * 1001

    response = client.post("/score/batch", json=oversized_batch)
    assert response.status_code == 422
    assert "exceeds maximum limit" in response.json()["detail"]
