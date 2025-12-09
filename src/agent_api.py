import os
import re
import time

import psutil
import requests
from fastapi import FastAPI, Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

REQUEST_COUNT = Counter(
    "requests_total",
    "Total number of requests",
    ["service", "endpoint", "status_code", "model_used"],
)
REQUEST_LATENCY = Histogram(
    "request_latency_seconds", "Request latency in seconds", ["service", "endpoint"]
)

PII_SCRUBBED = Counter(
    "pii_scrubbed_total", "Total PII scrubbed instances", ["pii_type"]
)
LANGUAGE_DETECTED = Counter(
    "language_detected_total", "Language detection counts", ["language"]
)
MODEL_CHOICE = Counter("model_choice_total", "Model selection counts", ["model_name"])
AGENT_DECISION_REASON = Counter(
    "agent_decision_reason_total", "Reason for model choice", ["reason"]
)
AGENT_PROCESSING_TIME = Histogram(
    "agent_processing_seconds", "Total agent processing time"
)
TEXT_LENGTH = Gauge("text_length", "Text length in characters")
AGENT_MEMORY_USAGE = Gauge("agent_memory_usage_bytes", "Agent memory usage in bytes")
AGENT_CPU_USAGE = Gauge("agent_cpu_usage_percent", "Agent CPU usage percentage")

app = FastAPI(title="CallCenterAI Agent")

TFIDF_URL = "http://tfidf:8010/predict"
TRANSFORMER_URL = "http://transformer:8020/predict"


def scrub_pii(text: str) -> dict:
    email_pattern = r"\S+@\S+\.\S+"
    phone_pattern = r"\d{7,15}"
    cin_pattern = r"\b\d{8}\b"
    name_pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"

    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)
    cins = re.findall(cin_pattern, text)
    names = re.findall(name_pattern, text)

    text = re.sub(email_pattern, "[EMAIL]", text)
    text = re.sub(phone_pattern, "[PHONE]", text)
    text = re.sub(cin_pattern, "[CIN]", text)
    text = re.sub(name_pattern, "[NAME]", text)

    pii_found = {"emails": emails, "phones": phones, "cins": cins, "names": names}

    return text, pii_found


def detect_language_simple(text: str) -> str:
    if not text or not text.strip():
        return "unknown"

    text_lower = text.lower()

    # Arabic detection
    if re.search(r"[\u0600-\u06FF]", text):
        return "ar"

    # Split text into words
    words = text_lower.split()

    # French keywords
    french_keywords = [
        "je",
        "tu",
        "il",
        "elle",
        "nous",
        "vous",
        "ils",
        "elles",
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "et",
        "mais",
        "ou",
        "où",
        "donc",
        "car",
        "de",
        "du",
        "des",
        "à",
        "au",
        "aux",
        "avec",
        "pour",
        "sur",
        "dans",
        "par",
        "est",
        "son",
        "ses",
        "mon",
        "ton",
        "votre",
        "notre",
        "leur",
        "ma",
        "ta",
        "sa",
    ]

    # English keywords
    english_keywords = [
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "have",
        "from",
        "what",
        "when",
        "where",
        "why",
        "how",
        "you",
        "your",
        "need",
        "help",
        "please",
        "my",
        "i",
        "me",
        "we",
        "us",
        "our",
        "can",
        "could",
        "would",
        "should",
        "will",
        "shall",
        "may",
        "might",
    ]

    # Count occurrences
    french_count = 0
    english_count = 0

    for word in words:
        if word in french_keywords:
            french_count += 1
        if word in english_keywords:
            english_count += 1

    # Also check for French-specific characters
    if re.search(r"[éèêëàâäçîïôöùûüÿ]", text_lower):
        french_count += 2

    # Decide
    if french_count > english_count and french_count >= 1:
        return "fr"
    elif english_count > 0:
        return "en"

    # Default to English for technical/short texts
    tech_terms = [
        "computer",
        "printer",
        "password",
        "login",
        "email",
        "phone",
        "network",
        "server",
        "software",
        "hardware",
        "system",
        "broken",
        "problem",
        "issue",
        "error",
        "fix",
        "help",
        "need",
        "want",
    ]

    if any(term in text_lower for term in tech_terms):
        return "en"

    return "unknown"


def choose_model(text: str):
    words = len(text.split())
    lang = detect_language_simple(text)

    if lang == "en" and words <= 10:
        simple_keywords = [
            "printer",
            "password",
            "login",
            "reset",
            "broken",
            "not working",
            "fix",
            "help",
            "issue",
            "problem",
            "error",
            "can't",
            "cannot",
            "won't",
            "doesn't",
        ]

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
            "need help",
        ]

        if any(kw in text_lower for kw in simple_keywords) or any(
            phrase in text_lower for phrase in simple_phrases
        ):
            return "tfidf", f"Short English ({words} words): {text[:50]}..."

    return "transformer", f"Length: {words}, Lang: {lang}"


class Ticket(BaseModel):
    text: str


@app.get("/metrics", response_class=Response)
async def metrics():
    return Response(generate_latest(), media_type="text/plain")


@app.post("/classify")
def classify_ticket(ticket: Ticket):
    start_time = time.time()
    process = psutil.Process(os.getpid())

    try:
        # Track text length
        TEXT_LENGTH.set(len(ticket.text))

        # Track system resources
        AGENT_MEMORY_USAGE.set(process.memory_info().rss)  # Memory in bytes
        AGENT_CPU_USAGE.set(process.cpu_percent(interval=0.1))  # CPU %

        clean_text, pii_details = scrub_pii(ticket.text)

        # Track PII
        for pii_type, items in pii_details.items():
            if items:
                PII_SCRUBBED.labels(pii_type=pii_type).inc(len(items))

        # Track language
        detected_lang = detect_language_simple(ticket.text)
        LANGUAGE_DETECTED.labels(language=detected_lang).inc()

        model_choice, reason = choose_model(clean_text)
        MODEL_CHOICE.labels(model_name=model_choice).inc()
        AGENT_DECISION_REASON.labels(reason=reason[:50]).inc()  # Truncate reason

        url = TFIDF_URL if model_choice == "tfidf" else TRANSFORMER_URL
        model_name = "TF-IDF + SVM" if model_choice == "tfidf" else "DistilBERT"

        resp = requests.post(url, json={"text": clean_text}, timeout=10)
        resp.raise_for_status()
        result = resp.json()

        # Track total processing time
        AGENT_PROCESSING_TIME.observe(time.time() - start_time)

        REQUEST_COUNT.labels(
            service="agent",
            endpoint="/classify",
            status_code="200",
            model_used=model_name.lower().replace(" ", "_"),
        ).inc()
        REQUEST_LATENCY.labels(service="agent", endpoint="/classify").observe(
            time.time() - start_time
        )

        result.update(
            {
                "model_used": model_name,
                "model_choice_reason": reason,
                "pii_scrubbed": ticket.text != clean_text,
                "pii_details": pii_details,
                "detected_language": detected_lang,
            }
        )
        return result

    except Exception as e:
        REQUEST_COUNT.labels(
            service="agent", endpoint="/classify", status_code="500", model_used="none"
        ).inc()
        return {"error": f"Model down: {str(e)}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec B104

# testing commend for i in {1..20}; do
# echo -e "\n=== Test $i ==="
# curl -X POST http://localhost:8000/classify \
#  -H "Content-Type: application/json" \
# --data @tests/test$i.json
# sleep 0.3
# done
