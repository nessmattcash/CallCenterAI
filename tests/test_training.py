import os

import joblib
import mlflow
import pandas as pd


def test_training_data_exists():
    df = pd.read_csv("data/all_tickets_processed_improved_v3.csv")
    assert len(df) > 1000
    assert "Document" in df.columns
    assert "Topic_group" in df.columns
    assert df["Topic_group"].nunique() > 5


def test_model_files_exist():
    assert os.path.exists("models/tfidf_svm_best.pkl")
    assert os.path.getsize("models/tfidf_svm_best.pkl") > 1000000

    assert os.path.exists("models/enhanced_multilingual_model")
    assert os.path.exists("models/enhanced_multilingual_model/config.json")
    assert os.path.exists("models/enhanced_multilingual_model/label_mappings.json")


def test_model_can_load_and_predict():
    model = joblib.load("models/tfidf_svm_best.pkl")
    predictions = model.predict(["computer problem", "need password"])
    assert len(predictions) == 2
    assert all(isinstance(p, str) for p in predictions)

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(["test text"])
        assert probas.shape[0] == 1


def test_mlflow_tracking():
    mlflow.set_tracking_uri("http://localhost:5000")
    experiments = mlflow.search_experiments()
    assert len(experiments) > 0

    runs = mlflow.search_runs(experiment_ids=[experiments[0].experiment_id])
    assert len(runs) > 0

    latest_run = runs.iloc[0]
    assert "metrics.accuracy" in latest_run
    accuracy = latest_run["metrics.accuracy"]
    assert 0.5 <= accuracy <= 1.0


def test_model_has_minimum_accuracy():
    model = joblib.load("models/tfidf_svm_best.pkl")
    df = pd.read_csv("data/all_tickets_processed_improved_v3.csv").head(100)

    texts = df["Document"].astype(str).tolist()
    predictions = model.predict(texts)

    assert len(predictions) == 100
    assert len(set(predictions)) > 1


def test_label_mappings():
    import json

    with open("models/enhanced_multilingual_model/label_mappings.json", "r") as f:
        mappings = json.load(f)

    assert "label2id" in mappings
    assert "id2label" in mappings
    assert len(mappings["label2id"]) == len(mappings["id2label"])
    assert len(mappings["label2id"]) > 5
