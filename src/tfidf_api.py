# src/tfidf_api.py
import time
from pathlib import Path

import joblib
from fastapi import FastAPI, Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

# beh chouft hethom touskie prometheus endpoint
REQUEST_COUNT = Counter(
    "tfidf_requests_total",  # CHANGE THIS
    "Total number of requests",
    ["service", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "tfidf_request_latency_seconds",
    "Request latency in seconds",
    ["service", "endpoint"],
)
MODEL_CONFIDENCE = Gauge("model_confidence", "Confidence of prediction", ["model"])
PREDICTION_CATEGORY = Counter(
    "prediction_category_total", "Predictions per category", ["category"]
)
TFIDF_PREDICTION_TIME = Histogram("tfidf_prediction_seconds", "TF-IDF prediction time")
TFIDF_CONFIDENCE = Gauge("tfidf_confidence", "Confidence score of TF-IDF")
TFIDF_CATEGORY = Counter("tfidf_category_total", "Predicted category", ["category"])
TFIDF_MODEL_INFO = Gauge(
    "tfidf_model_info", "TF-IDF model info", ["model_name", "model_version"]
)

TFIDF_MODEL_INFO.labels(model_name="tfidf_svm", model_version="1.0").set(1)

app = FastAPI(title="TF-IDF + SVM Service")

# Chargement du modèle
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "tfidf_svm_best.pkl"

# Check if file exists
if not MODEL_PATH.exists():
    # Try alternative path
    MODEL_PATH = Path("models/tfidf_svm_best.pkl")
    if not MODEL_PATH.exists():
        MODEL_PATH = Path("../models/tfidf_svm_best.pkl")

model = joblib.load(str(MODEL_PATH))


class TextInput(BaseModel):
    text: str


# Endpoint Prometheus → http://localhost:8010/metrics
@app.get("/metrics", response_class=Response)
async def metrics():
    return Response(generate_latest(), media_type="text/plain")


@app.post("/predict")
def predict(input: TextInput):
    start_time = time.time()
    prediction_start = time.time()

    try:
        if not input.text.strip():
            REQUEST_COUNT.labels(
                service="tfidf", endpoint="/predict", status_code="400"
            ).inc()
            return {"error": "Empty text"}

        pred = model.predict([input.text])[0]
        proba = model.predict_proba([input.text])[0].max()

        # NEW: Track prediction-specific metrics
        TFIDF_PREDICTION_TIME.observe(time.time() - prediction_start)
        TFIDF_CONFIDENCE.set(proba)
        TFIDF_CATEGORY.labels(category=pred).inc()

        REQUEST_COUNT.labels(
            service="tfidf", endpoint="/predict", status_code="200"
        ).inc()
        REQUEST_LATENCY.labels(service="tfidf", endpoint="/predict").observe(
            time.time() - start_time
        )

        return {"category": pred, "confidence": float(proba)}

    except Exception as e:
        REQUEST_COUNT.labels(
            service="tfidf", endpoint="/predict", status_code="500"
        ).inc()
        raise e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8010)  # nosec B104
