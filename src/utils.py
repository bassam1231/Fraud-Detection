import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path

FEATURE_COLS = ['step','type','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']
TARGET_COL = "isFraud"

def load_data(file_path="data/raw/transactions.csv"):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def preprocess_data(df, save_scaler=False, scaler_path="../models/scaler.pkl", encoder_path="../models/labelencoder.pkl"):
    if df is None:
        print("DataFrame is None, cannot preprocess.")
        return None, None, None, None

    df.drop(columns=['nameOrig', 'nameDest'], inplace=True)
    le = LabelEncoder()
    df["type"] = le.fit_transform(df["type"].astype(str))



    # select columns, ensure order
    df = df.copy()
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if save_scaler:
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        joblib.dump(le, encoder_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


