from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load('../models/tfidf_svm_best.pkl')

@app.post("/predict")
async def predict(data: dict): 
    text = data.get("text")
    if not text:
        return {"detail": [{"type": "missing", "loc": ["body", "text"], "msg": "Field required", "input": None}]}
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0].max()
    return {"category": pred, "confidence": proba}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
    #testi bhethi fi cmd curl -X POST "http://localhost:8010/predict" -d "{\"text\": \"My computer won't start.\"}" -H "Content-Type: application/json"