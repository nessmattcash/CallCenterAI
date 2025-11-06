# src/transformer_api.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

app = FastAPI()

MODEL_PATH = "../models/enhanced_multilingual_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

with open(f"{MODEL_PATH}/label_mappings.json") as f:
    mappings = json.load(f)
id2label = {int(k): v for k, v in mappings["id2label"].items()}  # Fix: str → int

model.eval()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(input: TextInput):
    if not input.text.strip():
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

    return {
        "category": id2label[pred_id],
        "confidence": float(confidence),
        "scores": {id2label[i]: float(p) for i, p in enumerate(probs)}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)
# beh si elyes fi cmd bash testi curl -X POST "http://localhost:8020/predict" -d "{\"text\": \"My computer won't start.\"}" -H "Content-Type: application/json" betbi3a fi /src python transformer_api.py    