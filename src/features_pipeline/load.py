import pandas as pd
import MetaTrader5 as mt5
from pathlib import Path

DATA_DIR = Path("data/raw")

def load_data_from_mt5(
        start: int = 0, 
        end: int = 99999, 
        symbol: str = "XAUUSD_i", 
        file_name: str = "training_data",
        timeframe=mt5.TIMEFRAME_M5, 
        output_dir: Path | str = DATA_DIR,) -> pd.DataFrame:
    """
    Load data from MetaTrader5 for the specified symbol and return as a DataFrame.
    
    Parameters:
    - start: The starting position for fetching rates.
    - end: The ending position for fetching rates.
    - symbol: The trading symbol for which to fetch data.
    - timeframe: The timeframe for which to fetch data.
    - output_dir: The directory where the loaded data will be saved if provided
    
    Returns:
    - A pandas DataFrame containing the rates data with 'time' as the index.
    """
    # Initialize MetaTrader5 connection
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return pd.DataFrame()  # Return empty DataFrame on failure

    # Fetch rates from MetaTrader5
    rates = mt5.copy_rates_from_pos(
        symbol,
        timeframe,
        start,
        end
    )

    # Convert to DataFrame
    df = pd.DataFrame(rates)

    # Convert 'time' column to datetime and set as index
    df["time"] = pd.to_datetime(df["time"], unit="s")
    #df = df.set_index('time')

    # Save 
    if output_dir is not None:
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(outdir / f"{symbol}_{file_name}.csv", index=False)

        #print("✅ Data loaded and saved to", outdir / f"{symbol}_training_data.csv")

    # Shutdown MetaTrader5 connection
    mt5.shutdown()

    return df


if __name__ == "__main__":
    # Load data from MetaTrader5
    df = load_data_from_mt5(start=0, end=99999)

    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save the DataFrame to CSV
    #df.to_csv(DATA_DIR / "XAUUSD_i_data.csv", index=True)
    print(f"✅ Data loaded and saved to {DATA_DIR / 'XAUUSD_i_training_data.csv'}")
