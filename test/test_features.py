import pandas as pd
import pytest
from pathlib import Path
from datetime import datetime, UTC
import MetaTrader5 as mt5


from src.feature_pipeline.load import load_data_from_mt5
from src.feature_pipeline.preprocessing import (
    clean_data, select_features_and_label, split_data, scale_training_data, scale_predict_data, preprocess_data
)
from src.feature_pipeline.feature_engineering import (
    read_XAUUSD_i_data, generate_features)

# =========================
# load.py – unit test
# =========================
# Confirms that data can be loaded from MetaTrader5 and saved to CSV.
def test_load_data_from_mt5(tmp_path):
    df = load_data_from_mt5(
        start=0, 
        end=250, 
        symbol="XAUUSD_i", 
        timeframe=mt5.TIMEFRAME_M5, 
        output_dir=tmp_path
    )
    assert not df.empty
    assert (tmp_path / "XAUUSD_i_training_data.csv").exists()
    print("✅ Data loading from MetaTrader5 test passed")

# =========================
# feature_engineering.py – unit test
# =========================
# Confirms that features can be generated from the raw data.
def test_generate_features(tmp_path):

    df = load_data_from_mt5(
            start=0, 
            end=250, 
            symbol="XAUUSD_i", 
            file_name = "training_data",
            timeframe=mt5.TIMEFRAME_M5,
            output_dir=tmp_path
        )

    # Load raw data
    df = read_XAUUSD_i_data(tmp_path / "XAUUSD_i_training_data.csv")
    
    # Generate features
    features_df = generate_features(df)
    
    # Check that features are generated
    assert not features_df.empty
    assert "body" in features_df.columns
    assert "body_ratio" in features_df.columns
    assert "lower_wick_ratio" in features_df.columns
    assert "ma_100" in features_df.columns
    assert "volume_ratio" in features_df.columns
    assert "target" in features_df.columns
    print("✅ Feature generation test passed")


# =========================
# preprocessing.py – unit test
# =========================
# Confirms that data can be cleaned, features and labels selected, data split, and scaled.
def test_clean_data(tmp_path):
    df = load_data_from_mt5(
            start=0, 
            end=250, 
            symbol="XAUUSD_i", 
            file_name = "training_data",
            timeframe=mt5.TIMEFRAME_M5, 
            output_dir=tmp_path
        )

     # Load raw data
    df = read_XAUUSD_i_data(tmp_path / "XAUUSD_i_training_data.csv")

    features_df = generate_features(df, output_dir=tmp_path)
    
    cleaned_df = clean_data(tmp_path / "feature_engineered_data.csv")
    assert cleaned_df.isnull().sum().sum() == 0
    print("✅ Data cleaning test passed")

def test_select_features_and_label(tmp_path):
    df = load_data_from_mt5(
            start=0, 
            end=250, 
            symbol="XAUUSD_i", 
            file_name = "training_data",
            timeframe=mt5.TIMEFRAME_M5, 
            output_dir=tmp_path
        )

    # Load raw data
    df = read_XAUUSD_i_data(tmp_path / "XAUUSD_i_training_data.csv")

    features_df = generate_features(df, output_dir=tmp_path)
    
    cleaned_df = clean_data(tmp_path / "feature_engineered_data.csv")
    
    selected_features, labels = select_features_and_label(cleaned_df.columns.drop(['target']).tolist(), cleaned_df, target_column="target")
    
    assert selected_features.columns.tolist() == cleaned_df.columns.drop(['target']).tolist()
    assert labels.equals(cleaned_df['target'])
    print("✅ Feature and label selection test passed")

def test_split_data(tmp_path):
    df = load_data_from_mt5(
            start=0, 
            end=250, 
            symbol="XAUUSD_i", 
            file_name = "training_data",
            timeframe=mt5.TIMEFRAME_M5, 
            output_dir=tmp_path
        )

    # Load raw data
    df = read_XAUUSD_i_data(tmp_path / "XAUUSD_i_training_data.csv")

    features_df = generate_features(df, output_dir=tmp_path)
    
    cleaned_df = clean_data(tmp_path / "feature_engineered_data.csv")

    selected_features, labels = select_features_and_label(cleaned_df.columns.drop(['target']).tolist(), cleaned_df, target_column="target")
    
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(selected_features, labels, output_dir=tmp_path)
    
    assert not x_train.empty and not y_train.empty
    assert not x_val.empty and not y_val.empty
    assert not x_test.empty and not y_test.empty
    print("✅ Data splitting test passed")

def test_scale_predict_data(tmp_path):
    df = load_data_from_mt5(
                start=0, 
                end=250, 
                symbol="XAUUSD_i", 
                file_name = "training_data",
                timeframe=mt5.TIMEFRAME_M5, 
                output_dir=tmp_path
            )
    
        # Load raw data
    df = read_XAUUSD_i_data(tmp_path / "XAUUSD_i_training_data.csv")
    
    features_df = generate_features(df, output_dir=tmp_path)
        
    cleaned_df = clean_data(tmp_path / "feature_engineered_data.csv")

    scaled_predict_data = scale_predict_data(cleaned_df)

    assert not scaled_predict_data.empty
    print("✅ Predict Data scaling test passed")


def test_scale_training_data(tmp_path):
    df = load_data_from_mt5(
            start=0, 
            end=250, 
            symbol="XAUUSD_i", 
            file_name = "training_data",
            timeframe=mt5.TIMEFRAME_M5, 
            output_dir=tmp_path
        )

    # Load raw data
    df = read_XAUUSD_i_data(tmp_path / "XAUUSD_i_training_data.csv")

    features_df = generate_features(df, output_dir=tmp_path)
    
    cleaned_df = clean_data(tmp_path / "feature_engineered_data.csv")

    selected_features, labels = select_features_and_label(cleaned_df.columns.drop(['target']).tolist(), cleaned_df, target_column="target")
    
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(selected_features, labels, output_dir=tmp_path)
    
    x_train_scaled_df, x_val_scaled_df, x_test_scaled_df = scale_training_data(x_train, x_val, x_test)
    
    assert not x_train_scaled_df.empty and not x_val_scaled_df.empty and not x_test_scaled_df.empty
    print("✅ Data scaling test passed")

def test_preprocess_data(tmp_path):
    df = load_data_from_mt5(
            start=0, 
            end=250, 
            symbol="XAUUSD_i", 
            file_name = "training_data",
            timeframe=mt5.TIMEFRAME_M5, 
            output_dir=tmp_path
        )

    # Load raw data
    df = read_XAUUSD_i_data(tmp_path / "XAUUSD_i_training_data.csv")

    features_df = generate_features(df, output_dir=tmp_path)
    
    cleaned_df = clean_data(tmp_path / "feature_engineered_data.csv")

    selected_features, labels = select_features_and_label(cleaned_df.columns.drop(['target']).tolist(), cleaned_df, target_column="target")
    
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(selected_features, labels, output_dir=tmp_path)
    
    x_train_scaled_df, y_train, x_val_scaled_df, y_val, x_test_scaled_df, y_test = preprocess_data(tmp_path / "feature_engineered_data.csv", selected_features=selected_features.columns.tolist(), target_column="target", output_dir=tmp_path)
    
    assert not x_train_scaled_df.empty and not x_val_scaled_df.empty and not x_test_scaled_df.empty
    assert not y_train.empty and not y_val.empty and not y_test.empty
    assert y_train.isin([0, 1]).all() and y_val.isin([0, 1]).all() and y_test.isin([0, 1]).all()
    print("✅ Full preprocessing pipeline test passed")
