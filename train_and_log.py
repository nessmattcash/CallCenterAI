# train_and_log.py
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# 1. Set MLflow to track to your container
mlflow.set_tracking_uri("http://localhost:5000")

print("STEP 1: Loading your model...")

# 2. Load YOUR trained model
model = joblib.load("models/tfidf_svm_best.pkl")

print("STEP 2: Loading test data to get REAL metrics...")

# 3. Load your data to calculate REAL metrics
df = pd.read_csv("data/all_tickets_processed_improved_v3.csv")

# Clean data
df = df.dropna(subset=['Document', 'Topic_group'])
df['Document'] = df['Document'].astype(str)
df['Topic_group'] = df['Topic_group'].astype(str)

# Take a sample for testing (first 1000 rows)
test_df = df.head(1000)
X_test = test_df['Document'].values
y_test = test_df['Topic_group'].values

print("STEP 3: Calculating REAL accuracy...")

# 4. Get REAL predictions
y_pred = model.predict(X_test)

# 5. Calculate REAL metrics
real_accuracy = accuracy_score(y_test, y_pred)
real_f1 = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy: {real_accuracy:.4f}")
print(f"F1 Score: {real_f1:.4f}")

print("STEP 4: Logging to MLflow with REAL values...")

# 6. Log to MLflow with REAL values
with mlflow.start_run(run_name="CallCenterAI_REAL_Metrics"):
    mlflow.log_param("model", "TF-IDF + LinearSVC")
    mlflow.log_param("dataset", "IT Service Tickets")
    mlflow.log_param("languages", "EN/FR/AR")
    
    # LOG REAL METRICS, NOT FAKE ONES
    mlflow.log_metric("accuracy", real_accuracy)
    mlflow.log_metric("f1_macro", real_f1)
    mlflow.log_metric("test_samples", len(X_test))

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="callcenterai",
        registered_model_name="CallCenterAI-Production"
    )

print("DONE! Model logged with REAL metrics.")
print(f"Accuracy: {real_accuracy:.4f}")
print(f"F1: {real_f1:.4f}")
print("Go to: http://localhost:5000")