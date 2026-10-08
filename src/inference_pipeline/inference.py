"""
Inference pipeline for XAUUSD Logistic Regression model.

- Loads xauusd historical data through MataTrader5
- Applies feature engineering and preprossing 
- Returns predictions on if there will be a 3% increase in the next 10 5m candles 
"""

import pandas as pd
import MetaTrader5 as mt5
from pathlib import Path
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from joblib import load

from src.feature_pipeline.load import load_data_from_mt5
from src.feature_pipeline.preprocessing import (
    clean_data, select_features_and_label, split_data, scale_predict_data, preprocess_data)
from src.feature_pipeline.feature_engineering import (read_XAUUSD_i_data, generate_features)

# ============================================================
# Default paths
# ============================================================

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PREDICTION_DIR = Path("data/model_prediction/model_out.csv")

DEFAULT_X_TRAIN = Path("data/processed/x_train_scaled_df.csv")
DEFAULT_Y_TRAIN = Path("data/processed/y_train.csv") 
DEFAULT_X_VAL = Path("data/processed/x_val_scaled_df.csv")
DEFAULT_Y_VAL = Path("data/processed/y_val.csv") 
DEFAULT_MODEL_PATH = Path("models/tuned_xauusd_lr_v1.pkl")
DEFAULT_OUTPUT_PATH = Path("data/model_prediction/prediction.csv")

# ============================================================
# Core infereence function
# ============================================================

def predict(model_path=DEFAULT_MODEL_PATH,
            raw_data_dir: Path | str = DATA_DIR,
            processed_data_dir: Path | str = PROCESSED_DIR  ):

    # Load historical data from MT5
    raw_df = load_data_from_mt5(
        file_name="_09_22_26",
        output_dir=raw_data_dir)
    
    raw_df = raw_df.set_index('time')

    # Generate features (features engineering)
    gen_features_df = generate_features(
        df=raw_df, 
        output_dir=processed_data_dir)

    # Process features
    cleaned_df = clean_data(dataframe=gen_features_df)

    # Scale cleaned data
    #scaled_cleaned_df = scale_predict_data(cleaned_df)
    
    features_list = cleaned_df.columns.tolist()
    features_list.remove("target")
    input_features = features_list

    model_inputs = {feature: cleaned_df[feature].values for feature in input_features} 

    # Load model & predict
    model = load(model_path)
    model_prob_output = model.predict(model_inputs)

    # Convert probabilities to binary predictions
    model_bin_output = (model_prob_output >= 0.532).astype(int)

    # Build output
    model_output = cleaned_df.copy()
    model_output["ml_signal"] = model_bin_output
    print(model_output.head(5))

    return model_output


if __name__ == "__main__":

    prediction_df = predict()

    prediction_df.to_csv(PREDICTION_DIR, index=False)
    print(f"✅ Predictions saved to {PREDICTION_DIR}")


