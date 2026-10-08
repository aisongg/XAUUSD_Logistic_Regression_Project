import pandas as pd
import numpy as np
from datetime import datetime, UTC
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import MetaTrader5 as mt5
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

def read_XAUUSD_i_data(
        raw_path: str = "data/raw/XAUUSD_i_training_data.csv",
        ) -> pd.DataFrame:
    """
    Read XAUUSD_i data from a CSV file and return as a DataFrame.
    
    Parameters:
    - raw_path: The path to the CSV file containing the raw data.
    - output_dir: The directory where the processed data will be saved.
    
    Returns:
    - A pandas DataFrame with 'time' as the index.
    """
    df = pd.read_csv(raw_path)
    #print("Columns:", df.columns.tolist())
    #print(df.head())

    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index('time')

    print("✅ Data loaded from", raw_path)

    return df

def generate_features(
        df: pd.DataFrame,
        file_name: str = "feature_engineered_data",
        output_dir: Path | str = None
        ) -> pd.DataFrame:

    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)
    df["return_10"] = df["close"].pct_change(10)

    #candle body
    df["body"] = df["close"] - df["open"]

    #absolute body
    df["body_abs"] = abs( df["close"] - df["open"])

    # Range
    df["range"] = df["high"] - df["low"]

    # Body to range ratio
    df["body_ratio"] = ( df["body_abs"] / df["range"])

    # Upper wick
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1))

    # Lower wick
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"])

    # Upper wick ratio
    df["upper_wick_ratio"] = (df["upper_wick"] / df["range"])

    # lower wick ratio
    df["lower_wick_ratio"] = (df["lower_wick"] / df["range"])

    # Average True Range

    atr = AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14
    )

    df["atr"] = atr.average_true_range()

    #Normalising ATR
    df["atr_pct"] = df["atr"] / df["close"]

    # Moving Averages

    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_50"] = df["close"].rolling(50).mean()
    df["ma_100"] = df["close"].rolling(100).mean()

    df["price_ma20_distance"] = (df["close"] / df["ma_20"] - 1)

    df["ma20_ma50_distance"] = (df["ma_20"] / df["ma_50"] - 1)

    # Relative strength index
    rsi = RSIIndicator(
        close=df["close"],
        window=14
    )

    df["rsi"] = rsi.rsi()

    # Average volume

    df["volume_ma20"] = ( df["tick_volume"].rolling(20).mean() )

    #Volume ratio
    df["volume_ratio"] = ( df["tick_volume"] / df["volume_ma20"] )

    # Time of day features

    df["hour"] = df.index.hour
    df["minute"] = df.index.minute

    # Cyclical time
    minutes = df.index.hour * 60 + df.index.minute

    df["time_sin"] = np.sin( 2 * np.pi * minutes / 1440 )

    df["time_cos"] = np.cos( 2 * np.pi * minutes / 1440 )

    #predicting whether the market will rise over the next 10 M5 candles.

    future_return = ( df["close"].shift(-10) / df["close"] - 1 )

    df["target"] = (future_return > 0.003 ).astype(int)

    # Save
    if output_dir is not None:
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(outdir / f"{file_name}.csv", index=False)

        print("✅ Feature engineering complete.")

    return df


if __name__ == "__main__":
    # Load data from CSV
    df = read_XAUUSD_i_data()

    # Generate features
    df = generate_features(df)

    # Save the DataFrame to CSV
    #df.to_csv(PROCESSED_DIR / "feature_engineered_data.csv", index=True)
    print(f"✅ Feature engineered data saved to {PROCESSED_DIR}/feature_engineered_data.csv.")

