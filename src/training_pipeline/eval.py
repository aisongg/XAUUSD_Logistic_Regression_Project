import keras
import tensorflow as tf
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from pathlib import Path
from joblib import dump, load

DEFAULT_X_VAL = Path("data/processed/x_val_scaled_df.csv")
DEFAULT_Y_VAL = Path("data/processed/y_val.csv") #.values.ravel()
DEFAULT_MODEL_PATH = Path("models/lr_v1.pkl")

def evaluate_model(
        x_val_scaled_path: Path | str = DEFAULT_X_VAL, 
        y_val_path: Path | str = DEFAULT_Y_VAL, 
        model_path: Path | str = DEFAULT_MODEL_PATH):
    """
    Evaluate a trained machine learning model on the validation set.

    Parameters:
    - x_val_scaled_path: Path to the scaled validation features.
    - y_val_path: Path to the validation labels.
    - model_path: Path to the trained model.

    Returns:
    - Dictionary containing evaluation metrics.
    """
    # Load the validation data
    x_val_scaled = pd.read_csv(x_val_scaled_path)

    x_val_scaled_dict = {feature: x_val_scaled[feature].values for feature in x_val_scaled.columns}

    y_val = pd.read_csv(y_val_path)

    # Load the trained model
    model = load(model_path)

    # Make predictions on the validation set
    y_pred_prob = model.predict(x_val_scaled_dict)
    y_pred = (y_pred_prob >= 0.506).astype(int)  # Convert probabilities to binary predictions

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_prob)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

    print("📊 Evaluation:")
    print(f"   Accuracy={accuracy:.2f}  Precision={precision:.2f}  Recall={recall:.2f}  F1={f1:.2f}  AUC={auc:.4f}")
    
    return metrics

if __name__ == "__main__":
    # Example usage
    metrics = evaluate_model()
