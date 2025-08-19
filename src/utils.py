import os
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

from src.configuration.config import config

def load_data():
    try:
        df = pd.read_csv(config.data.dataset_path)
        return df
    except Exception as e:
        print(f"Error loading Data from {config.data.dataset_path}: {e}")
        return None

def preprocess_data(df):
    if df is None:
        raise ValueError("Data is None")

    df.drop(columns=config.data.features_to_drop, inplace=True)

    for column_to_encode in config.data.features_to_encode:
        le = LabelEncoder()
        df[column_to_encode] = le.fit_transform(df[column_to_encode])

        if config.data.save_encoder:
            os.makedirs(config.data.encoder_save_path, exist_ok=True)
            dump(le, config.data.encoder_save_path + f'label_encoder_{column_to_encode}.pkl')

    x = df[config.data.features]
    y = df[config.data.target]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=config.training.test_size,
                                                        random_state=config.seed, stratify=y)

    if config.data.use_smote:
        smote = SMOTE(random_state=config.seed)
        x_train, y_train = smote.fit_resample(x_train, y_train)


    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    if config.data.save_scaler:
        os.makedirs(config.data.scaler_save_path, exist_ok=True)
        dump(scaler, config.data.scaler_save_path + f'scaler.pkl')

    if config.data.save_data_after_processing:
        df.to_csv(config.data.processed_dataset_path, index=False)

    return x_train_scaled, x_test_scaled, y_train, y_test
