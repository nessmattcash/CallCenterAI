import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import mlflow
import mlflow.sklearn
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from stop_words import get_stop_words
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
np.random.seed(42)

# Load data
df = pd.read_csv('data/all_tickets_processed_improved_v3.csv')

# Enhanced clean text function (from your code)
stop_en = set(stopwords.words('english'))
stop_fr = set(get_stop_words('french'))
stop_ar = set(get_stop_words('arabic'))
all_stops = stop_en.union(stop_fr, stop_ar)
key_terms = {'facturation', 'billing', 'invoice', 'hardware', 'computer', 'laptop', 'password', 'login', 'access', 'storage', 'space', 'disk', 'hr', 'human', 'resources', 'project', 'purchase', 'buy', 'software', 'administrative', 'rights', 'permission', 'miscellaneous', 'other', 'network', 'email'}
stemmer_en = SnowballStemmer('english')

def enhanced_clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    processed_words = []
    for word in words:
        if len(word) < 2:
            continue
        if word in key_terms:
            processed_words.append(word)
            continue
        if word in all_stops:
            continue
        stemmed_word = stemmer_en.stem(word)
        processed_words.append(stemmed_word)
    return ' '.join(processed_words)

df['cleaned_document'] = df['Document'].apply(enhanced_clean_text)

# Split data
X = df['cleaned_document']
y = df['Topic_group']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Base pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=30000, ngram_range=(1, 3), min_df=2, max_df=0.9, sublinear_tf=True)),
    ('svm', LinearSVC(C=1.0, class_weight=class_weight_dict, random_state=42, max_iter=2000))
])
pipeline.fit(X_train, y_train)
y_pred_base = pipeline.predict(X_test)
accuracy_base = accuracy_score(y_test, y_pred_base)
f1_base = f1_score(y_test, y_pred_base, average='weighted')
print("\nBase Model Performance:")
print(f"Base Accuracy: {accuracy_base:.4f}")
print(f"Base F1 Score: {f1_base:.4f}")
print(classification_report(y_test, y_pred_base))

# Calibrated pipeline
calibrated_svc = CalibratedClassifierCV(LinearSVC(C=1.0, class_weight=class_weight_dict, random_state=42, max_iter=2000), cv=5, method='sigmoid')
pipeline_calib = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=30000, ngram_range=(1, 3), min_df=2, max_df=0.9, sublinear_tf=True)),
    ('svm', calibrated_svc)
])
pipeline_calib.fit(X_train, y_train)
y_pred_calib = pipeline_calib.predict(X_test)
accuracy_calib = accuracy_score(y_test, y_pred_calib)
f1_calib = f1_score(y_test, y_pred_calib, average='weighted')
print("\nCalibrated Model Performance:")
print(f"Calibrated Accuracy: {accuracy_calib:.4f}")
print(f"Calibrated F1 Score: {f1_calib:.4f}")
print(classification_report(y_test, y_pred_calib))

# Select best model
best_model = pipeline_calib if f1_calib > f1_base else pipeline
best_accuracy = accuracy_calib if f1_calib > f1_base else accuracy_base
best_f1 = f1_calib if f1_calib > f1_base else f1_base
print(f"\nBest Model Selected - Accuracy: {best_accuracy:.4f}, F1: {best_f1:.4f}")

# Save model
model_filename = 'models/tfidf_svm_best.pkl'
joblib.dump(best_model, model_filename)
print(f"Model saved as: {model_filename}")

# MLflow logging
mlflow.set_experiment("CallCenterAI_TFIDF_SVM")
with mlflow.start_run(run_name="TFIDF_SVM_Run"):
    mlflow.log_param("max_features", 30000)
    mlflow.log_param("ngram_range", (1,3))
    mlflow.log_param("C", 1.0)
    mlflow.log_metric("accuracy", best_accuracy)
    mlflow.log_metric("f1", best_f1)
    mlflow.sklearn.log_model(best_model, "tfidf_svm_model")

# Test examples (your 10 examples)
examples = [
    "My computer won't start.",
    "I need help with my password.",
    "Problème de facturation ce mois.",
    "Request for new storage space.",
    "HR support for leave approval.",
    "Purchase new software license.",
    "Internal project delay reported.",
    "مشكلة في الشبكة",
    "Administrative rights needed.",
    "Miscellaneous query about office."
]
cleaned_examples = [enhanced_clean_text(example) for example in examples]
predictions = best_model.predict(cleaned_examples)
probabilities = best_model.predict_proba(cleaned_examples)
for i, example in enumerate(examples):
    confidence = max(probabilities[i]) * 100
    print(f"Example {i+1}: '{example}' -> Predicted: {predictions[i]}, Confidence: {confidence:.1f}%")