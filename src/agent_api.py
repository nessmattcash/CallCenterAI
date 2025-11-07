# src/agent_api.py
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import re
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0 
import uvicorn


app = FastAPI(title="CallCenterAI Agent")

#model URLs
TFIDF_URL = "http://localhost:8010/predict"
TRANSFORMER_URL = "http://localhost:8020/predict"

# === PII Scrubber ===
def scrub_pii(text: str) -> str:
    text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)      # ken d5alt b email
    text = re.sub(r'\b\d{10,}\b', '[PHONE]', text)       # ken d5alt b phone
    text = re.sub(r'\b\d{6,}\b', '[ID]', text)           # ken d5alt b ID
    return text

#Smart Router
def choose_model(text: str):
    words = text.split()
    length = len(words)
    try:
        lang = detect(text)
    except:
        lang = "unknown"

    # TF-IDF: short, English, simple keywords
    simple_keywords = ["password", "imprimante", "printer", "forgot", "reset", "locked"]
    if (length < 20 or 
        lang not in ["fr", "ar"] or 
        any(kw in text.lower() for kw in simple_keywords)):
        return "tfidf", f"Short/simple ({length} words, lang: {lang})"

    # Transformer: long, French/Arabic, complex
    return "transformer", f"Complex/multilingual ({length} words, lang: {lang})"

class Ticket(BaseModel):
    text: str

@app.post("/classify")
async def classify(ticket: Ticket):
    clean_text = scrub_pii(ticket.text)
    model_choice, reason = choose_model(clean_text)

    if model_choice == "tfidf":
        resp = requests.post(TFIDF_URL, json={"text": clean_text})
    else:
        resp = requests.post(TRANSFORMER_URL, json={"text": clean_text})

    result = resp.json()
    result.update({
        "model_used": "TF-IDF + SVM" if model_choice == "tfidf" else "DistilBERT Multilingual",
        "reason": reason,
        "pii_removed": clean_text != ticket.text
    })
    return result
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)