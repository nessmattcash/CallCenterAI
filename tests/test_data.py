import joblib
import pandas as pd


def test_data_exists():
    df = pd.read_csv("data/all_tickets_processed_improved_v3.csv")
    assert len(df) > 0
    assert "Document" in df.columns
    assert "Topic_group" in df.columns


def test_model_exists():
    model = joblib.load("models/tfidf_svm_best.pkl")
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_transformer_model_exists():
    import os

    assert os.path.exists("models/enhanced_multilingual_model")
    assert os.path.exists("models/enhanced_multilingual_model/config.json")
