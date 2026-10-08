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
from joblib import dump

DEFAULT_X_TRAIN = Path("data/processed/x_train_scaled_df.csv")
DEFAULT_Y_TRAIN = Path("data/processed/y_train.csv") #.values.ravel()
DEFAULT_X_VAL = Path("data/processed/x_val_scaled_df.csv")
DEFAULT_Y_VAL = Path("data/processed/y_val.csv") #.values.ravel()
DEFAULT_MODEL_PATH = Path("models/lr_v1.pkl")

def train_model(
        x_train_scaled_path: Path | str = DEFAULT_X_TRAIN, 
        y_train_path: Path | str = DEFAULT_Y_TRAIN, 
        x_val_scaled_path: Path | str = DEFAULT_X_VAL, 
        y_val_path: Path | str = DEFAULT_Y_VAL, 
        model_export_path: Path | str = DEFAULT_MODEL_PATH,
        model_type='logistic_regression'):
    """
    Train a machine learning model based on the specified model type.

    Parameters:
    - x_train_scaled_path: Path to the scaled training features.
    - y_train_path: Path to the training labels.
    - x_val_scaled_path: Path to the scaled validation features.
    - y_val_path: Path to the validation labels.
    - model_export_path: Path to save the trained model.
    - model_type: Type of model to train ('logistic_regression' or 'neural_network').

    Returns:
    - Trained model.
    """
    # Load the training and validation data
    x_train_scaled_df = pd.read_csv(x_train_scaled_path)
    y_train_df = pd.read_csv(y_train_path)
    x_val_scaled_df = pd.read_csv(x_val_scaled_path)
    y_val_df = pd.read_csv(y_val_path)

    input_features = x_train_scaled_df.columns

    params = {
        'learning_rate': 4.3964397704767564e-05,
        'batch_size': 128,
        'epochs': 49,
        'threshold': 0.5060975391356682}

    # Create a dictionary of input layers for each feature
    model_inputs = {feature: keras.Input(shape=(1,), name=feature) for feature in input_features}

    # Concatenate all input layers into a single tensor (feature vector)
    x = keras.layers.Concatenate()(list(model_inputs.values()))     

    # Create the output layer with a single neuron and sigmoid activation for binary classification
    model_output = keras.layers.Dense(units = 1, activation='sigmoid')(x)

    if model_type == 'logistic_regression':
        
        # Create the model
        model = keras.Model(inputs=model_inputs, outputs=model_output)

        # Compile the model with binary cross-entropy loss and an optimizer (specify how the model should learn)
        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(learning_rate=params['learning_rate']), metrics=[
                keras.metrics.BinaryAccuracy(
                    name="accuracy"
                ),
                keras.metrics.AUC(
                    name="auc"
                ),
                keras.metrics.Precision(
                    name="precision",
                    thresholds= params['threshold']
                ),
                keras.metrics.Recall(
                    name="recall",
                    thresholds= params['threshold']
                ), 
                keras.metrics.RootMeanSquaredError(
                    name='rmse'
                ),
                keras.metrics.MeanAbsoluteError(
                    name='mae'
                )
            ])

        # Prepare the input data for training by creating a dictionary of feature arrays
        train_inputs = {feature: x_train_scaled_df[feature].values for feature in input_features}

        # Prepare validation inputs
        validation_inputs = {feature: x_val_scaled_df[feature].values for feature in input_features}

        # Prepare evaluation inputs
        #evaluation_inputs = {feature: x_test[feature].values for feature in input_features}

        # Train the model using the training data and validate it using the validation data
        model.fit(
            x=train_inputs,
            y=y_train_df.values,
            validation_data=(validation_inputs, y_val_df.values),
            epochs=params['epochs'],
            batch_size=params['batch_size'])

        val_probability = model.predict(
            validation_inputs,
            verbose=0
            ).ravel()

        threshold = params["threshold"]

        val_prediction = (
            val_probability >= threshold
        ).astype(int)

        # calculate your metrics

        val_accuracy = accuracy_score(y_val_df.values, val_prediction)
        val_precision = precision_score(
            y_val_df.values,
            val_prediction,
            zero_division=0
        )
        val_recall = recall_score(
            y_val_df.values,
            val_prediction,
            zero_division=0
        )
        val_f1 = f1_score(
            y_val_df.values,
            val_prediction,
            zero_division=0
        )
        val_auc = roc_auc_score(
            y_val_df.values,
            val_probability
        )

        metrics = {
            "accuracy": val_accuracy,
            "precision": val_precision,
            "recall": val_recall,
            "f1": val_f1,
            "auc": val_auc
        }

         # Save the trained model
        out = Path(model_export_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        dump(model, out)

        print(f"✅ Model trained. Saved to {out}")
        print(f" Accuracy={metrics['accuracy']:.2f}  Precision={metrics['precision']:.2f}  Recall={metrics['recall']:.2f}  F1={metrics['f1']:.2f}  AUC={metrics['auc']:.4f}")


        return model, metrics
    

    elif model_type == 'neural_network':
        from keras.models import Sequential
        from keras.layers import Dense

        # Define a simple neural network architecture
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(x_train_scaled_df.shape[1],)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(x_train_scaled_df, y_train_df.values, epochs=params['epochs'], batch_size=params['batch_size'], validation_data=(x_val_scaled_df, y_val_df.values))

        return model

    else:
        raise ValueError("Unsupported model type. Choose 'logistic_regression' or 'neural_network'.")


if __name__ == "__main__":
    train_model()
