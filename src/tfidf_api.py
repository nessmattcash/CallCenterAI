# src/tfidf_api.py
from fastapi import FastAPI, Response
from pydantic import BaseModel
import joblib
import time
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import Gauge
#beh chouft hethom touskie prometheus endpoint
REQUEST_COUNT = Counter(
    'requests_total',
    'Total number of requests',
    ['service', 'endpoint', 'status_code']
)
REQUEST_LATENCY = Histogram(
    'request_latency_seconds',
    'Request latency in seconds',
    ['service', 'endpoint']
)
MODEL_CONFIDENCE = Gauge('model_confidence', 'Confidence of prediction', ['model'])
PREDICTION_CATEGORY = Counter('prediction_category_total', 'Predictions per category', ['category'])

app = FastAPI(title="TF-IDF + SVM Service")

# Chargement du modèle
model = joblib.load("../models/tfidf_svm_best.pkl")

class TextInput(BaseModel):
    text: str

# Endpoint Prometheus → http://localhost:8010/metrics
@app.get("/metrics", response_class=Response)
async def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.post("/predict")
def predict(input: TextInput):
    start_time = time.time()
    
    try:
        if not input.text.strip():
            REQUEST_COUNT.labels(service="tfidf", endpoint="/predict", status_code="400").inc()
            return {"error": "Empty text"}

        pred = model.predict([input.text])[0]
        proba = model.predict_proba([input.text])[0].max()

        REQUEST_COUNT.labels(service="tfidf", endpoint="/predict", status_code="200").inc()
        REQUEST_LATENCY.labels(service="tfidf", endpoint="/predict").observe(time.time() - start_time)

        return {
            "category": pred,
            "confidence": float(proba)
        }

    except Exception as e:
        # Erreur → on logue 500
        REQUEST_COUNT.labels(service="tfidf", endpoint="/predict", status_code="500").inc()
        raise e

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)