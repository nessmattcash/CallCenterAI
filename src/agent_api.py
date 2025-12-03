from fastapi import FastAPI
from pydantic import BaseModel
import requests
import re

app = FastAPI(title="CallCenterAI Agent")

TFIDF_URL = "http://tfidf:8010/predict"
TRANSFORMER_URL = "http://transformer:8020/predict"

def scrub_pii(text: str) -> dict:
    email_pattern = r'\S+@\S+\.\S+'
    phone_pattern = r'\d{7,15}'
    cin_pattern = r'\b\d{8}\b'
    name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
    
    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)
    cins = re.findall(cin_pattern, text)
    names = re.findall(name_pattern, text)
    
    text = re.sub(email_pattern, '[EMAIL]', text)
    text = re.sub(phone_pattern, '[PHONE]', text)
    text = re.sub(cin_pattern, '[CIN]', text)
    text = re.sub(name_pattern, '[NAME]', text)
    
    pii_found = {
        "emails": emails,
        "phones": phones,
        "cins": cins,
        "names": names
    }
    
    return text, pii_found

def detect_language_simple(text: str) -> str:
    text_lower = text.lower()
    
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    french_words = len(re.findall(r'\b(je|tu|il|elle|nous|vous|ils|elles|le|la|les|un|une|des|et|mais|ou|où|donc|car|ne|pas|de|du|des|à|au|aux|avec|pour|sur|dans|par|est|son|ses|mon|ton|votre|notre|leur)\b', text_lower))
    english_words = len(re.findall(r'\b(the|and|for|with|that|this|have|from|what|when|where|why|how|you|your|need|help|please|thank|thanks|my|i|me|we|us|our|can|could|would|should|will|shall|may|might|must)\b', text_lower))
    
    if arabic_chars > 5:
        return "ar"
    
    if french_words > english_words and french_words >= 2:
        return "fr"
    
    if english_words >= 2:
        return "en"
    
    if any(word in text_lower for word in ["email", "phone", "cin", "name", "printer", "password", "login"]):
        return "en"
    
    return "unknown"

def choose_model(text: str):
    words = len(text.split())
    lang = detect_language_simple(text)
    
    if lang == "en" and words <= 10:
        simple_keywords = ["printer", "password", "login", "reset", "broken", "not working", "fix", "help", "issue", "problem", "error", "can't", "cannot", "won't", "doesn't"]
        
        text_lower = text.lower()
        simple_phrases = [
            "printer broken",
            "password reset",
            "can't login",
            "cannot login",
            "help login",
            "screen broken",
            "keyboard not working",
            "mouse broken",
            "internet down",
            "wifi not working",
            "printer not working",
            "forgot password",
            "reset password",
            "need help"
        ]
        
        if any(kw in text_lower for kw in simple_keywords) or any(phrase in text_lower for phrase in simple_phrases):
            return "tfidf", f"Short English ({words} words): {text[:50]}..."
    
    return "transformer", f"Length: {words}, Lang: {lang}"

class Ticket(BaseModel):
    text: str

@app.post("/classify")
def classify_ticket(ticket: Ticket):
    clean_text, pii_details = scrub_pii(ticket.text)
    model_choice, reason = choose_model(clean_text)
    url = TFIDF_URL if model_choice == "tfidf" else TRANSFORMER_URL
    
    try:
        resp = requests.post(url, json={"text": clean_text}, timeout=10)
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        return {"error": f"Model down: {str(e)}"}
    
    model_name = "TF-IDF + SVM" if model_choice == "tfidf" else "DistilBERT"
    
    result.update({
        "model_used": model_name,
        "model_choice_reason": reason,
        "pii_scrubbed": ticket.text != clean_text,
        "pii_details": pii_details,
        "original_text_length": len(ticket.text),
        "cleaned_text": clean_text,
        "detected_language": detect_language_simple(ticket.text)
    })
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)