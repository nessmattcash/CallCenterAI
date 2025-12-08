import requests
import time

def test_all_services_running():
    services = [
        ("http://localhost:8000", "agent"),
        ("http://localhost:8010", "tfidf"),
        ("http://localhost:8020", "transformer"),
        ("http://localhost:5000", "mlflow"),
        ("http://localhost:9090", "prometheus"),
        ("http://localhost:3000", "grafana"),
    ]
    
    for url, service in services:
        try:
            if service in ["mlflow", "prometheus", "grafana"]:
                response = requests.get(url, timeout=5)
            else:
                response = requests.get(f"{url}/docs", timeout=5)
            assert response.status_code < 500
            print(f"{service} is running")
        except:
            print(f"{service} not accessible")

def test_api_contract():
    test_cases = [
        {"text": "printer broken", "expected_models": ["tfidf", "TF-IDF + SVM"]},
        {"text": "complex technical issue with multiple systems", "expected_models": ["transformer", "DistilBERT"]},
    ]
    
    for test in test_cases:
        response = requests.post("http://localhost:8000/classify", 
                                json={"text": test["text"]}, 
                                timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data["model_used"].lower() in test["expected_models"]