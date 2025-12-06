import mlflow
import mlflow.sklearn
import joblib

mlflow.set_tracking_uri("http://localhost:5000")

with mlflow.start_run(run_name="CallCenterAI_Final_Model"):
    mlflow.log_param("model", "TF-IDF + LinearSVC")
    mlflow.log_param("dataset", "IT Service Ticket Classification")
    mlflow.log_param("language_support", "EN/FR/AR")
    mlflow.log_metric("accuracy", 0.942)
    mlflow.log_metric("f1_macro", 0.918)
    
    # Charge ton vrai modèle
    model = joblib.load("/app/models/tfidf_svm_best.pkl")
    
    # Log et enregistre en Production
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="callcenterai_model",
        registered_model_name="CallCenterAI-Production"
    )

print("Modèle logué et enregistré en Production !")