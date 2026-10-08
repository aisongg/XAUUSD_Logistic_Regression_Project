import pandas as pd
#import numpy as np
import optuna
import mlflow
import keras
#import tensorflow as tf
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
#import mlflow.tensorflow
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from pathlib import Path
from joblib import dump, load

DEFAULT_X_TRAIN = Path("data/processed/x_train_scaled_df.csv")
DEFAULT_Y_TRAIN = Path("data/processed/y_train.csv") #.values.ravel
DEFAULT_X_VAL = Path("data/processed/x_val_scaled_df.csv")
DEFAULT_Y_VAL = Path("data/processed/y_val.csv") #.values.ravel()
DEFAULT_X_TEST = Path("data/processed/x_test_scaled_df.csv")
DEFAULT_Y_TEST = Path("data/processed/y_test.csv") #.values.ravel()
DEFAULT_MODEL_PATH = Path("models/tuned_xauusd_lr_v1.pkl")

"""
# Enter this in CLI to load the MLflow UI
mlflow ui --backend-store-uri "sqlite:///C:/Users/AKWAENO/Documents/Gold_USD_Bot/Machine_learning_based_XAU_bot-main/mlflow.db" --host 127.0.0.1 --port 5000
# Load the link below on your browser
http://localhost:5000

"""

def tune_model(
        x_train_scaled_path: Path | str = DEFAULT_X_TRAIN, 
        y_train_path: Path | str = DEFAULT_Y_TRAIN, 
        x_val_scaled_path: Path | str = DEFAULT_X_VAL, 
        y_val_path: Path | str = DEFAULT_Y_VAL, 
        model_export_path: Path | str = DEFAULT_MODEL_PATH,
        model_name: str = 'tuned_xauusd_lr_v1',
        n_trail: int = 50):
    """
    Tune hyperparameters of a machine learning model based on the specified model type.

    Parameters:
    - x_train_scaled_path: Path to the scaled training features.
    - y_train_path: Path to the training labels.
    - x_val_scaled_path: Path to the scaled validation features.
    - y_val_path: Path to the validation labels.
    - model_export_path: Path to save the tuned model.
    - model_name: Name of the model to be saved.

    Returns:
    - Tuned model.
    """
    global input_features, train_inputs, train_labels, validation_inputs, validation_labels


    x_train_scaled_df = pd.read_csv(x_train_scaled_path)
    train_labels = pd.read_csv(y_train_path).values
    x_val_scaled_df = pd.read_csv(x_val_scaled_path)
    validation_labels = pd.read_csv(y_val_path).values
    x_test_scaled_df = pd.read_csv(DEFAULT_X_TEST)
    test_labels = pd.read_csv(DEFAULT_Y_TEST).values

    input_features = x_train_scaled_df.columns

    # Prepare the input data for training by creating a dictionary of feature arrays
    train_inputs = {feature: x_train_scaled_df[feature].values for feature in input_features}

    # Prepare validation inputs
    validation_inputs = {feature: x_val_scaled_df[feature].values for feature in input_features}

    # Prepare evaluation inputs
    evaluation_inputs = {feature: x_test_scaled_df[feature].values for feature in input_features}

    

    def objective(trial):

        learning_rate = trial.suggest_float(
            "learning_rate",
            1e-5,
            1e-2,
            log=True
        )

        batch_size = trial.suggest_categorical(
            "batch_size",
            [64, 128, 256, 512]
        )

        epochs = trial.suggest_int(
            "epochs",
            10,
            50
        )

        threshold = trial.suggest_float(
            "threshold",
            0.50,
            0.90
        )

        # Make sure there isn't an old run
        if mlflow.active_run() is not None:
            mlflow.end_run()

        # Start MLflow run
        with mlflow.start_run(
            run_name=f"optuna_trial_{trial.number}",
            nested=True
        ):
            
            # ----------------------------------------------------
            # Log hyperparameters
            # ----------------------------------------------------

            mlflow.log_params({

                "learning_rate": learning_rate,

                "batch_size": batch_size,

                "epochs": epochs,

                "threshold": threshold           
            })

        # Create a dictionary of input layers for each feature
        model_inputs = {feature: keras.Input(shape=(1,), name=feature) for feature in input_features}

        # Concatenate all input layers into a single tensor (feature vector)
        x = keras.layers.Concatenate()(list(model_inputs.values()))     

        # Create the output layer with a single neuron and sigmoid activation for binary classification
        model_output = keras.layers.Dense(units = 1, activation='sigmoid')(x)

        # Create the model
        model = keras.Model(inputs=model_inputs, outputs=model_output)

        # Compile the model with binary cross-entropy loss and an optimizer (specify how the model should learn)
        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate), metrics=[
            keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            keras.metrics.AUC(
                name="auc"
            ),
            keras.metrics.Precision(
                name="precision",
                thresholds= threshold
            ),
            keras.metrics.Recall(
                name="recall",
                thresholds= threshold
            ), 
            keras.metrics.RootMeanSquaredError(
                name='rmse'
            ),
            keras.metrics.MeanAbsoluteError(
                name='mae'
            )
        ])

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        # Train the model using the training data and validate it using the validation data
        history = model.fit(
            x=train_inputs,
            y=train_labels,
            validation_data=(validation_inputs, validation_labels),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0)

        
        # Get validation probabilities
        val_probability = model.predict(
            x = validation_inputs,
            verbose=0
        ).ravel()

        # Convert probabilities into signals
        val_prediction = (
            val_probability >= threshold
        ).astype(int)

        # Calculate precision
        precision = precision_score(
            validation_labels,
            val_prediction,
            zero_division=0
        )

        return precision


    # Close any run left open from previous execution
    if mlflow.active_run() is not None:
        mlflow.end_run()

    mlflow.set_tracking_uri(
    r"sqlite:///C:/Users/AKWAENO/Documents/Gold_USD_Bot/Machine_learning_based_XAU_bot-main/mlflow.db"
    )

    mlflow.set_experiment(
    "XAUUSD_LR_Training_Pipeline_Experiment_Optuna"
    )

    # Create an Optuna study to maximize the precision score
    study = optuna.create_study(
    study_name="XAUUSD_LR_Training_Pipeline_Experiment_Optuna_v1",
    storage="sqlite:///optuna.db",
    direction="maximize",
    load_if_exists=True
    )

    target_trials = n_trail

    remaining_trials = target_trials - len(study.trials)

    if remaining_trials > 0:

        print(f"Running {remaining_trials} more trials...")
        study.optimize(objective, n_trials=remaining_trials)

    else:

        print(f"Study already has {len(study.trials)} trials.")

    best_params = study.best_params

    # Create a dictionary of input layers for each feature
    model_inputs = {feature: keras.Input(shape=(1,), name=feature) for feature in input_features}

    # Concatenate all input layers into a single tensor (feature vector)
    x = keras.layers.Concatenate()(list(model_inputs.values()))     

    # Create the output layer with a single neuron and sigmoid activation for binary classification
    model_output = keras.layers.Dense(units = 1, activation='sigmoid')(x)

    # Create the model
    best_model = keras.Model(inputs=model_inputs, outputs=model_output)

    # Compile the model with binary cross-entropy loss and an optimizer (specify how the model should learn)
    best_model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(learning_rate=best_params['learning_rate']), metrics=[
            keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            keras.metrics.AUC(
                name="auc"
            ),
            keras.metrics.Precision(
                name="precision",
                thresholds= best_params['threshold']
            ),
            keras.metrics.Recall(
                name="recall",
                thresholds= best_params['threshold']
            ), 
            keras.metrics.RootMeanSquaredError(
                name='rmse'
            ),
            keras.metrics.MeanAbsoluteError(
                name='mae'
            )
        ])

        # Train the model using the training data and validate it using the validation data
    history = best_model.fit(
        x=train_inputs,
        y=train_labels,
        validation_data=(validation_inputs, validation_labels),
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        verbose=1)

    # Evaluate the best model on test set
    val_probability = best_model.predict(
        validation_inputs,
        verbose=0
    ).ravel()
    
    test_probability = best_model.predict(
        evaluation_inputs,
        verbose=0
    ).ravel()

    # Apply the optimized classification threshold
    threshold = best_params["threshold"]

    val_prediction = (
        val_probability >= threshold
    ).astype(int)

    test_prediction = (
        test_probability >= threshold
    ).astype(int)

    # calculate your metrics
    val_accuracy = accuracy_score(validation_labels, val_prediction)
    val_precision = precision_score(
        validation_labels,
        val_prediction,
        zero_division=0
    )
    val_recall = recall_score(
        validation_labels,
        val_prediction,
        zero_division=0
    )
    val_f1 = f1_score(
        validation_labels,
        val_prediction,
        zero_division=0
    )
    val_auc = roc_auc_score(
        validation_labels,
        val_probability
    )

    test_accuracy = accuracy_score(test_labels, test_prediction)
    test_precision = precision_score(
        test_labels,
        test_prediction,
        zero_division=0
    )
    test_recall = recall_score(
        test_labels,
        test_prediction,
        zero_division=0
    )
    test_f1 = f1_score(
        test_labels,
        test_prediction,
        zero_division=0
    )
    test_auc = roc_auc_score(
        test_labels,
        test_probability
    )
    best_metrics = {

            "best_test_auc":
                test_auc,

            "best_test_accuracy":
                test_accuracy,

            "best_test_precision":
                test_precision,

            "best_test_recall":
                test_recall,

            "best_test_f1_score":
                test_f1
        }
    print(f"✅ Model tuned. Best parameters: {best_params}")
    print(f"   Best metrics: {best_metrics}")

    out = Path(model_export_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(best_model, out)

    # Make sure there isn't an old run
    if mlflow.active_run() is not None:
        mlflow.end_run()

    with mlflow.start_run(
        run_name=model_name,
        nested=True
        ):

        # Log Optuna best parameters
        mlflow.log_params(best_params)

        # Log validation metrics
        mlflow.log_metrics({
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "val_auc": val_auc
        })

        # Log test metrics
        mlflow.log_metrics({
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "test_auc": test_auc
        })

        # Log final model
        mlflow.tensorflow.log_model(
            best_model,
            name=model_name
        )

        mlflow.log_param(
        "classification_threshold",
        best_params["threshold"]
    )

        mlflow.set_tag("model_type", "Logistic Regression")
        mlflow.set_tag("asset", "XAUUSD")
        mlflow.set_tag("timeframe", "M5")
        mlflow.set_tag("prediction_horizon", "10 candles")
        mlflow.set_tag("model_stage", "final")

    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

    return best_params, best_metrics


if __name__ == "__main__":

    tune_model()
