# tests/test_agent.py
import os
import sys
import unittest.mock as mock

import pytest

sys.path.append("src")

with mock.patch("agent_api.requests.post") as mock_post:
    # Setup mock response
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"category": "Hardware", "confidence": 0.85}
    mock_response.raise_for_status = mock.Mock()
    mock_post.return_value = mock_response

    # Import after mock
    from fastapi.testclient import TestClient

    from agent_api import app, choose_model, detect_language_simple, scrub_pii

    client = TestClient(app)


# Mark test as expected to fail (or skip)
@pytest.mark.xfail(reason="Agent needs external services")
def test_agent_predict():
    response = client.post("/classify", json={"text": "My computer won't turn on"})
    assert response.status_code == 200
    data = response.json()
    assert "category" in data
    assert "confidence" in data
    assert "model_used" in data
    assert "model_choice_reason" in data


def test_pii_scrubbing():
    text = "My email is test@example.com and phone is 1234567890"
    cleaned, pii = scrub_pii(text)
    assert "[EMAIL]" in cleaned
    assert "[PHONE]" in cleaned
    assert "test@example.com" not in cleaned
    assert "1234567890" not in cleaned
    assert len(pii["emails"]) == 1
    assert len(pii["phones"]) == 1


def test_language_detection():
    assert detect_language_simple("My computer is broken") == "en"
    assert detect_language_simple("Mon ordinateur est cassé") == "fr"
    text_with_arabic = "الكمبيوتر لا يعمل"
    assert detect_language_simple(text_with_arabic) == "ar"


def test_model_choice():
    text, reason = choose_model("printer broken")
    assert text == "tfidf"
    assert "Short English" in reason

    text, reason = choose_model(
        "A very long and complex technical issue with multiple error codes and detailed description"
    )
    assert text == "transformer"
    assert "Length:" in reason


def test_agent_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
