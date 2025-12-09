# src/transformer_api.py
import json
import time

import torch
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REQUEST_COUNT = Counter(
    "transformer_requests_total",
    "Total requests",
    ["service", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "transformer_request_latency_seconds",
    "Request latency in seconds",
    ["service", "endpoint"],
)
MODEL_CONFIDENCE = Gauge(
    "transformer_model_confidence", "Confidence of prediction", ["model"]
)
PREDICTION_CATEGORY = Counter(
    "transformer_prediction_category_total", "Predictions per category", ["category"]
)
TRANSFORMER_PREDICTION_TIME = Histogram(
    "transformer_prediction_seconds", "Transformer prediction time"
)
TRANSFORMER_CONFIDENCE = Gauge(
    "transformer_confidence", "Confidence score of Transformer"
)
TRANSFORMER_CATEGORY = Counter(
    "transformer_category_total", "Predicted category", ["category"]
)
TRANSFORMER_MODEL_INFO = Gauge(
    "transformer_model_info", "Transformer model info", ["model_name", "model_version"]
)

# Add after loading your model
TRANSFORMER_MODEL_INFO.labels(
    model_name="distilbert-multilingual", model_version="1.0"
).set(1)

app = FastAPI(title="Enhanced Multilingual Transformer ")

MODEL_PATH = "./models/enhanced_multilingual_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)  # nosec B615
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)  # nosec B615

with open(f"{MODEL_PATH}/label_mappings.json") as f:
    mappings = json.load(f)
id2label = {int(k): v for k, v in mappings["id2label"].items()}  # Fix: str → int

model.eval()


class TextInput(BaseModel):
    text: str


@app.get("/metrics", response_class=Response)
async def metrics():
    return Response(generate_latest(), media_type="text/plain")


@app.post("/predict")
async def predict(input: TextInput):
    start_time = time.time()
    prediction_start = time.time()

    try:
        if not input.text.strip():
            REQUEST_COUNT.labels(
                service="transformer", endpoint="/predict", status_code="400"
            ).inc()
            return {"error": "Empty text"}

        inputs = tokenizer(
            input.text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            pred_id = probs.argmax().item()
            confidence = probs.max().item()
            pred_label = id2label[pred_id]

        # NEW: Track prediction-specific metrics
        TRANSFORMER_PREDICTION_TIME.observe(time.time() - prediction_start)
        TRANSFORMER_CONFIDENCE.set(confidence)
        TRANSFORMER_CATEGORY.labels(category=pred_label).inc()

        REQUEST_COUNT.labels(
            service="transformer", endpoint="/predict", status_code="200"
        ).inc()
        REQUEST_LATENCY.labels(service="transformer", endpoint="/predict").observe(
            time.time() - start_time
        )

        return {
            "category": pred_label,
            "confidence": float(confidence),
            "scores": {id2label[i]: float(p) for i, p in enumerate(probs)},
        }

    except Exception as e:
        REQUEST_COUNT.labels(
            service="transformer", endpoint="/predict", status_code="500"
        ).inc()
        raise e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8020)  # nosec B104
# beh si elyes fi cmd bash testi curl -X POST "http://localhost:8020/predict" -d "{\"text\": \"My computer won't start.\"}" -H "Content-Type: application/json" betbi3a fi /src python transformer_api.py
