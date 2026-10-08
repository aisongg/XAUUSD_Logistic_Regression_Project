import keras
import pytest
from pathlib import Path


from src.training_pipeline.train import train_model
from src.training_pipeline.eval import evaluate_model
from src.training_pipeline.tune import tune_model

# =========================
# train.py – unit test
# =========================
# Confirms that model is trained and metrics are generated

DEFAULT_X_TRAIN = Path("data/processed/x_train_scaled_df.csv")
DEFAULT_Y_TRAIN = Path("data/processed/y_train.csv") #.values.ravel()
DEFAULT_X_VAL = Path("data/processed/x_val_scaled_df.csv")
DEFAULT_Y_VAL = Path("data/processed/y_val.csv") #.values.ravel()
DEFAULT_MODEL_PATH = Path("models/testing_lr_v1.pkl")

def test_train_model(tmp_path):
    model, metrics = train_model(
            x_train_scaled_path= DEFAULT_X_TRAIN, 
            y_train_path= DEFAULT_Y_TRAIN, 
            x_val_scaled_path = DEFAULT_X_VAL, 
            y_val_path = DEFAULT_Y_VAL, 
            model_export_path = tmp_path / 'testing_lr_v1.pkl',
            model_type ='logistic_regression'
            )
    
    # Confirm a model was returned
    assert model is not None
    # Confirm the returned object is a Keras model
    assert isinstance(model, keras.Model)

    # Confirm expected metrics exist
    expected_metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    for metric in expected_metrics:

        assert metric in metrics
    """
    # Confirm metric values are valid
    for metric in expected_metrics:

        assert 0 <= metrics[metric] <= 2
    """
    #assert not model.empty
    assert (tmp_path / "testing_lr_v1.pkl").exists()
    print("✅ Model trained successfully")


# =========================
# eval.py – unit test
# =========================
# Confirms that model is evaluated

def test_evaluate_model(tmp_path):
    model, metrics = train_model(
                x_train_scaled_path= DEFAULT_X_TRAIN, 
                y_train_path= DEFAULT_Y_TRAIN, 
                x_val_scaled_path = DEFAULT_X_VAL, 
                y_val_path = DEFAULT_Y_VAL, 
                model_export_path = tmp_path / 'testing_lr_v1.pkl',
                model_type ='logistic_regression'
                )
    
    eval_metrics = evaluate_model(
        DEFAULT_X_VAL,
        DEFAULT_Y_VAL,
        tmp_path / 'testing_lr_v1.pkl'
    )

    # Confirm expected metrics exist
    expected_metrics = ["accuracy", "precision", "recall", "f1", "auc"]

    for metric in expected_metrics:

        assert metric in eval_metrics
"""
    # Confirm metric values are valid
    for metric in expected_metrics:

        assert 0 <= eval_metrics[metric] <= 1
"""
# =========================
# tune.py – unit test
# =========================
# Confirms that model is tuned

def test_tune_model(tmp_path):
    model, metrics = train_model(
                    x_train_scaled_path= DEFAULT_X_TRAIN, 
                    y_train_path= DEFAULT_Y_TRAIN, 
                    x_val_scaled_path = DEFAULT_X_VAL, 
                    y_val_path = DEFAULT_Y_VAL, 
                    model_export_path = tmp_path / 'testing_lr_v1.pkl',
                    model_type ='logistic_regression'
                    )
        
    eval_metrics = evaluate_model(
        DEFAULT_X_VAL,
        DEFAULT_Y_VAL,
        tmp_path / 'testing_lr_v1.pkl'
    )
    
    best_params, best_metrics = tune_model(model_export_path=tmp_path / 'tuned_lr_v1.pkl',
                                           model_name="tuned_lr_v1",
                                           n_trail=2
                                           )

    # Confirm expected metrics exist
    expected_metrics = ["best_test_auc", "best_test_accuracy", "best_test_precision", "best_test_recall", "best_test_f1_score"]
    expected_params = ["learning_rate", "batch_size", "epochs", "threshold"]

    for param in expected_params:

        assert param in best_params

    # Confirm expected matrics exist
    for metric in expected_metrics:

        assert metric in best_metrics

    
