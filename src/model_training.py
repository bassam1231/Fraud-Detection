import joblib, os
from sklearn.ensemble import RandomForestClassifier

def train(x_train, y_train):
    model = RandomForestClassifier(random_state=42)

    model.fit(x_train, y_train)

    os.makedirs("../model", exist_ok=True)
    joblib.dump(model, 'models/Random Forest.pkl')

    return model