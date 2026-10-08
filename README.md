# Machine Learning-Based XAUUSD Trading Bot

An end-to-end Python project for researching short-horizon XAUUSD (gold/USD) price direction and automating technical-analysis trading workflows through MetaTrader 5.

The repository contains two complementary, but currently separate, components:

- **Machine-learning research pipeline:** extracts market data, engineers features, prepares chronological datasets, trains a binary classifier, tunes it with Optuna, and tracks experiments with MLflow.
- **Rule-based execution bot:** monitors live MetaTrader 5 data, evaluates multi-timeframe technical signals, manages orders and position state, and sends Telegram notifications.

> **Important:** The live execution script is rule-based at present; it does **not** load the trained ML model to make trading decisions. This repository is a research/prototype project, not investment advice or a guarantee of trading performance.

## Project goals

- Build a reproducible data and modelling workflow for XAUUSD five-minute (`M5`) candles.
- Predict whether the price will rise by more than **0.3% over the following 10 M5 candles**.
- Evaluate and tune a binary classification model using time-ordered data splits.
- Provide a separate automated execution prototype with risk controls and notifications.

## Features

### Data and feature pipeline

- Fetches OHLCV/tick-volume candle data from MetaTrader 5.
- Engineers market, candlestick, volatility, volume, trend, momentum, and time-of-day features, including:
  - 1-, 3-, 5-, and 10-period returns
  - Candle body, range, upper/lower wicks, and their ratios
  - ATR and ATR as a percentage of price
  - 20-, 50-, and 100-period moving averages and MA-distance features
  - RSI, rolling volume, and volume ratio
  - Cyclical intraday time features
- Removes missing/duplicate observations, standardises features with `StandardScaler`, and retains chronological ordering.
- Uses a 70% / 15% / 15% train, validation, and test split to respect the time-series nature of market data.

### Modelling and experimentation

- Implements a Keras binary logistic-regression classifier with a sigmoid output.
- Measures accuracy, precision, recall, F1 score, and ROC-AUC.
- Optimises learning rate, batch size, epochs, and probability threshold with Optuna.
- Logs tuned parameters, metrics, model metadata, and model artifacts with MLflow.
- Includes saved model artifacts and notebooks documenting extraction, cleaning, normalisation, training, and tuning.

### MetaTrader 5 execution prototype

- Polls XAUUSD data on M5 and H1 timeframes.
- Uses MA trend, RSI, ADX, stochastic, volume, and candle-strength conditions for long/short signals.
- Submits MT5 market orders with configured position size, stop loss, take profit, deviation, magic number, and retry handling.
- Persists active-position state to JSON, uses rotating logs, and sends Telegram alerts for signals, entries, and exits.

## Repository structure

```text
.
├── data/
│   ├── raw/                         # Extracted MT5 market data
│   └── processed/                   # Feature sets and train/validation/test splits
├── models/                          # Serialized trained models
├── notebooks/
│   ├── 00_ohlcv_data_extraction.ipynb
│   ├── 01_ohlcv_data_cleaning.ipynb
│   ├── 02_data_normalization.ipynb
│   ├── 03_logistic_regression.ipynb
│   └── 04_lr_hyperparameter_tuning_MLFlow.ipynb
├── src/
│   ├── feature_pipeline/
│   │   ├── load.py                  # MT5 data extraction
│   │   ├── feature_engineering.py   # Feature and label creation
│   │   └── preprocessing.py         # Cleaning, splitting, scaling
│   └── training_pipeline/
│       ├── train.py                 # Model training
│       ├── eval.py                  # Model evaluation
│       └── tune.py                  # Optuna + MLflow tuning workflow
├── tests/                           # Pipeline and training tests
└── xau_bot_v_6.1.0.py               # Rule-based MT5 execution bot
```

## Tech stack

Python, Pandas, NumPy, TensorFlow/Keras, scikit-learn, Optuna, MLflow, MetaTrader5, `ta`, Joblib, Pytest, Requests, and Telegram Bot API.

## Getting started

### Prerequisites

- Python 3.10+
- MetaTrader 5 desktop terminal installed and logged in to a broker account that exposes the required XAUUSD symbol (the project defaults to `XAUUSD_i`)
- A virtual environment

### Installation

```bash
git clone <your-repository-url>
cd Machine_learning_based_XAU_bot-main

python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install pandas numpy tensorflow scikit-learn optuna mlflow MetaTrader5 ta joblib pytest requests
```

> MetaTrader 5 must be running and connected before scripts that call the MT5 API can obtain data or place orders.

## Typical ML workflow

Run commands from the repository root.

```bash
# 1. Extract XAUUSD M5 data from MetaTrader 5
python -m src.feature_pipeline.load

# 2. Generate technical, price-action, and target features
python -m src.feature_pipeline.feature_engineering

# 3. Clean, split chronologically, and standardise features
python -m src.feature_pipeline.preprocessing

# 4. Train the baseline classifier
python -m src.training_pipeline.train

# 5. Evaluate the saved model
python -m src.training_pipeline.eval

# 6. Run Optuna tuning and log the final run with MLflow
python -m src.training_pipeline.tune
```

To inspect the tracked experiments locally, start the MLflow UI from the repository root:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Tests

```bash
pytest
```

The test suite covers data loading, feature generation, cleaning, feature/label selection, chronological splitting, scaling, preprocessing, and model-training paths. Some tests require an active MetaTrader 5 connection because they request market data.

## Configuration and security

Before running the execution bot, configure the symbol, lot size, polling interval, and messaging credentials for your environment. Do not commit broker credentials, Telegram tokens, account details, model databases, or live state files.

Recommended local-only files:

```gitignore
.venv/
.venv-tf/
__pycache__/
.pytest_cache/
*.db
*.pkl
*.keras
mlruns/
mt5_bot.log
live_state.json
hft_live_state.json
.env
```

Store secrets in environment variables or a local `.env` file that is excluded from version control. If a token has ever been committed or shared, revoke and regenerate it before publishing.

## Risk disclaimer

This project is for educational and research purposes. Financial markets are volatile, and historical or validation results do not predict future performance. Validate logic on historical data and a demo account before any live use; you are solely responsible for all trading decisions and risk management.

## Future improvements

- Integrate the trained model into the live signal-generation path.
- Add walk-forward validation, transaction costs, slippage, and backtesting.
- Add probability calibration, class-imbalance handling, and trading-specific performance metrics.
- Version datasets, scalers, feature definitions, and model releases.
- Externalise configuration and secrets; add CI and reproducible dependency locking.
