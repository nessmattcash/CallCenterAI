import os
import sys

sys.path.append("src")

from fastapi.testclient import TestClient

from transformer_api import app

client = TestClient(app)


def test_transformer_predict():
    response = client.post("/predict", json={"text": "My computer won't turn on"})
    assert response.status_code == 200
    data = response.json()
    assert "category" in data
    assert "confidence" in data
    assert "scores" in data


def test_transformer_empty():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code in [200, 400]
    if response.status_code == 400:
        data = response.json()
        assert "error" in data


def test_transformer_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
