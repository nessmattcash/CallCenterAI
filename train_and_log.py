# train_and_log.py
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import numpy as np
import re
from sklearn.metrics import accuracy_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from stop_words import get_stop_words
import nltk

nltk.download('stopwords', quiet=True)

def clean_text_for_inference(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    stop_en = set(stopwords.words('english'))
    stop_fr = set(get_stop_words('french'))
    stop_ar = set(get_stop_words('arabic'))
    all_stops = stop_en.union(stop_fr, stop_ar)
    
    words = text.split()
    processed_words = []
    stemmer = SnowballStemmer('english')
    
    for word in words:
        if len(word) < 2:
            continue
        if word in all_stops:
            continue
        stemmed_word = stemmer.stem(word)
        processed_words.append(stemmed_word)
    
    return ' '.join(processed_words)

print("CONNECTING TO MLFLOW...")
mlflow.set_tracking_uri("http://localhost:5000")

print("LOADING MODEL...")
model = joblib.load("models/tfidf_svm_best.pkl")

print("LOADING TEST DATA...")
df = pd.read_csv("data/all_tickets_processed_improved_v3.csv")
df = df.dropna(subset=['Document', 'Topic_group'])
df['Document'] = df['Document'].astype(str)
df['Topic_group'] = df['Topic_group'].astype(str)

test_df = df.sample(1000, random_state=42)
X_test_raw = test_df['Document'].values

print("CLEANING TEST DATA (SAME AS TRAINING)...")
X_test_cleaned = [clean_text_for_inference(text) for text in X_test_raw]
y_test = test_df['Topic_group'].values

print("MAKING PREDICTIONS...")
y_pred = model.predict(X_test_cleaned)

real_accuracy = accuracy_score(y_test, y_pred)
real_f1 = f1_score(y_test, y_pred, average='macro')

print(f"REAL ACCURACY: {real_accuracy:.4f}")
print(f"REAL F1: {real_f1:.4f}")

print("LOGGING TO MLFLOW...")
with mlflow.start_run(run_name="TFIDF_SVM_Final_Production"):
    mlflow.log_param("model", "TF-IDF + LinearSVC")
    mlflow.log_param("dataset", "IT Service Tickets")
    mlflow.log_param("languages", "EN/FR/AR")
    mlflow.log_param("test_size", 1000)
    
    mlflow.log_metric("accuracy", real_accuracy)
    mlflow.log_metric("f1_macro", real_f1)
    mlflow.log_metric("f1_weighted", f1_score(y_test, y_pred, average='weighted'))
    
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="callcenterai",
        registered_model_name="CallCenterAI-Production"
    )

print(f" MODEL LOGGED WITH ACCURACY: {real_accuracy:.4f}")
print(f"GO TO: http://localhost:5000")
