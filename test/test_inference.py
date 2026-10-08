from pathlib import Path
from src.inference_pipeline.inference import predict

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PREDICTION_DIR = Path("data/model_prediction")

DEFAULT_X_TRAIN = Path("data/processed/x_train_scaled_df.csv")
DEFAULT_Y_TRAIN = Path("data/processed/y_train.csv") 
DEFAULT_X_VAL = Path("data/processed/x_val_scaled_df.csv")
DEFAULT_Y_VAL = Path("data/processed/y_val.csv") 
DEFAULT_MODEL_PATH = Path("models/tuned_xauusd_lr_v1.pkl")
DEFAULT_OUTPUT_PATH = Path("data/model_prediction/prediction.csv")

# =========================
# inference.py – unit test
# =========================
# Confirms that the trained model performs prediction

def test_predict(tmp_path):
    model_output = predict(model_path=DEFAULT_MODEL_PATH,
            raw_data_dir = DATA_DIR,
            processed_data_dir = PROCESSED_DIR)

    # Check output is not empty
    assert not model_output.empty

    # Must include prediction column
    assert "ml_signal" in model_output.columns

    print("✅ Inference pipeline test passed. Predictions:")
    print(model_output[["ml_signal"]].head())
