from src.utils import preprocess_data
import joblib, os
from sklearn.metrics import classification_report
import pandas as pd

def predict(model, x_predict):
    if model is None or x_predict is None:
        raise FileNotFoundError("❌ Model or data not found, run training first or pass data!!")

    y_predict = model.predict(x_predict)

    return y_predict

def evaluate(y_test, y_predict):
    print(classification_report(y_test, y_predict))


