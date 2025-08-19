from dataclasses import dataclass
from typing import Dict, List

# ----------------------------
# Data Configuration
# ----------------------------
@dataclass
class DataConfig:
    # Paths
    data_dir: str = "/Data"
    dataset_path: str = "./Data/raw/transactions.csv"
    processed_dataset_path: str = "./Data/processed/processed.csv"
    save_data_after_processing: bool = True

    # Preprocessing
    features: List[str] = None # Features used in the X variable
    target: str = "isFraud"  # Name of the target variable
    features_to_encode: List[str] = None  # Columns needing label encoding
    features_to_scale: List[str] = None # Columns to scale
    features_to_drop: List[str] = None # Columns to drop
    use_smote: bool = True  # Whether to oversample with SMOTE
    save_scaler: bool = True  # Whether to save StandardScaler
    save_encoder: bool = True # Save LabelEncoder(s)
    save_model: bool = True # Save Model after training
    model_save_path: str = "models/"
    scaler_save_path: str = "models/scalers/"
    encoder_save_path: str = "models/encoders/"
    visualizations_save_path: str = "Visualizations/"

    def __post_init__(self):
        self.features = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        self.features_to_scale = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        self.features_to_encode = ['type']
        self.features_to_drop = ['nameOrig', 'nameDest']


# ----------------------------
# Model Configuration (Random Forest)
# ----------------------------
@dataclass
class ModelConfig:
    model_name: str = "Random Forest"
    hyperparameters: Dict = None

    # Random Forest Default Hyperparameters
    def __post_init__(self):
        self.hyperparameters = {
            "n_estimators": 300,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": 7,  # Auto for classification
            "bootstrap": True,
            "random_state": 42,
            "class_weight": "balanced",  # Handles class imbalance
        }

# ----------------------------
# Training & Evaluation
# ----------------------------
@dataclass
class TrainingConfig:
    # Evaluation
    test_size: float = 0.2
    cv_folds: int = 5
    metrics: List[str] = None

    def __post_init__(self):
        self.metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]


# ----------------------------
# Logging (MLflow)
# ----------------------------
@dataclass
class MLflowConfig:
    tracking_uri: str = "./mlruns"  # Local MLflow tracking
    experiment_name: str = "random_forest_experiment"
    log_artifacts: bool = True  # Save models/plots
    log_parameters: bool = True  # Log hyperparameters


# ----------------------------
# Experiment Configuration (Master Config)
# ----------------------------
@dataclass
class GlobalConfig:
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    mlflow: MLflowConfig = None
    seed: int = None  # Global random seed

    def __post_init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.mlflow = MLflowConfig()
        self.seed = 42


# Singleton Config Instance
config = GlobalConfig()