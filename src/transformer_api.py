# src/transformer_api.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from prometheus_client import Counter, Histogram, generate_latest, Gauge
import time 
from fastapi import Response
from fastapi.responses import PlainTextResponse


REQUEST_COUNT = Counter('requests_total', 'Total requests', ['service', 'endpoint', 'status_code'])
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency', ['service', 'endpoint'])
MODEL_CONFIDENCE = Gauge('model_confidence', 'Confidence of prediction', ['model'])
PREDICTION_CATEGORY = Counter('prediction_category_total', 'Predictions per category', ['category'])

app = FastAPI(title="Enhanced Multilingual Transformer ")

MODEL_PATH = "./models/enhanced_multilingual_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

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
    
    try:
        if not input.text.strip():
            REQUEST_COUNT.labels(service="transformer", endpoint="/predict", status_code="400").inc()
            return {"error": "Empty text"}

        inputs = tokenizer(
            input.text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            pred_id = probs.argmax().item()
            confidence = probs.max().item()

        REQUEST_COUNT.labels(service="transformer", endpoint="/predict", status_code="200").inc()
        REQUEST_LATENCY.labels(service="transformer", endpoint="/predict").observe(time.time() - start_time)

        return {
            "category": id2label[pred_id],
            "confidence": float(confidence),
            "scores": {id2label[i]: float(p) for i, p in enumerate(probs)}
        }

    except Exception as e:
        REQUEST_COUNT.labels(service="transformer", endpoint="/predict", status_code="500").inc()
        raise e
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)
# beh si elyes fi cmd bash testi curl -X POST "http://localhost:8020/predict" -d "{\"text\": \"My computer won't start.\"}" -H "Content-Type: application/json" betbi3a fi /src python transformer_api.py    