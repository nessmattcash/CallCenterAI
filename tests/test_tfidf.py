import os
import sys

sys.path.append("src")

from fastapi.testclient import TestClient

from tfidf_api import app

client = TestClient(app)


def test_tfidf_predict():
    response = client.post("/predict", json={"text": "My computer won't turn on"})
    assert response.status_code == 200
    data = response.json()
    assert "category" in data
    assert "confidence" in data
    assert isinstance(data["confidence"], float)


def test_tfidf_empty():
    response = client.post("/predict", json={"text": ""})
    # Accept either 200 or 400
    assert response.status_code in [200, 400]
    if response.status_code == 400:
        data = response.json()
        assert "error" in data


def test_tfidf_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


def test_model_loading():
    import joblib

    model = joblib.load("models/tfidf_svm_best.pkl")
    assert model is not None
    predictions = model.predict(["test text"])
    assert len(predictions) > 0
