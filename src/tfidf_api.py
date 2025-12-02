# src/tfidf_api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="TF-IDF + SVM Service")

model = joblib.load("../models/tfidf_svm_best.pkl")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    if not input.text.strip():
        return {"error": "Empty text"}
    
    pred = model.predict([input.text])[0]
    proba = model.predict_proba([input.text])[0].max()
    
    return {
        "category": pred,
        "confidence": float(proba)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
    #testi bhethi fi cmd curl -X POST "http://localhost:8010/predict" -d "{\"text\": \"My computer won't start.\"}" -H "Content-Type: application/json"