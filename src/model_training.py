import joblib, os
from sklearn.ensemble import RandomForestClassifier
from src.configuration.config import config

def train(x_train, y_train):
    model = RandomForestClassifier(**config.model.hyperparameters)
    model.fit(x_train, y_train)

    if config.data.save_model:
        os.makedirs(config.data.model_save_path, exist_ok=True)
        joblib.dump(model, config.data.model_save_path + "model.pkl")

    return model