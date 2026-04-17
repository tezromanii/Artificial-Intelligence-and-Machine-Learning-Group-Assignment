import pandas as pd
import numpy as np
import os


RAW_DATA_PATH = "Groupwork/data/raw_stock_data.csv"
OUTPUT_FOLDER = "Groupwork/datasets"

# Yahoo Finance tickers
ticker_map = {
    "AMZN": "amzn",
    "NVDA": "nvda",
    "GOOGL": "googl",
    "MSFT": "msft",
    "^GSPC": "sp500"   # S&P 500 index
}

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# Load raw data
print(f"Looking for raw data at: {os.path.abspath(RAW_DATA_PATH)}")

if not os.path.exists(RAW_DATA_PATH):
    print(f"\nERROR: Could not find {RAW_DATA_PATH}")
    exit(1)

print("Loading raw dataset...")
df = pd.read_csv(RAW_DATA_PATH, header=[0,1], index_col=0)

# Convert index to datetime
df.index = pd.to_datetime(df.index)

print(f"Loaded data with shape: {df.shape}")
print(f"Available tickers: {df.columns.levels[0].tolist()}")


# Process each ticker individually
for yf_symbol, filename in ticker_map.items():
    print(f"\nProcessing {yf_symbol} ...")

    try:
        # Extract this ticker's OHLCV columns
        temp = df[yf_symbol].copy()
        
        # Check if we got the data
        if temp.empty:
            print(f"Warning: No data found for {yf_symbol}")
            continue

        # Print what columns we actually have
        print(f"  Columns found: {temp.columns.tolist()}")
        
        # Flatten column names (remove any multi-index)
        temp.columns = temp.columns.tolist()

        # Feature Engineering
        
        # 1. Daily Return (%)
        temp["Daily_Return_%"] = temp["Close"].pct_change() * 100

        # 2. SMA (Simple Moving Averages)
        temp["SMA5"] = temp["Close"].rolling(5).mean()
        temp["SMA10"] = temp["Close"].rolling(10).mean()
        temp["SMA20"] = temp["Close"].rolling(20).mean()

        # 3. Price Distance from SMA10
        temp["Price_Distance_SMA10"] = temp["Close"] - temp["SMA10"]

        # 4. Daily Volatility Proxy (absolute % change)
        temp["Volatility"] = temp["Daily_Return_%"].abs()

        # 5. Volume Change (%)
        temp["Volume_Change_%"] = temp["Volume"].pct_change() * 100

        # Create target label
        temp["Target"] = (temp["Close"].shift(-1) > temp["Close"]).astype(int)

        # Cleaning dataset
        temp = temp.dropna()  # drop rows with NA created by rolling windows

        # Add ticker column for identity
        temp["Ticker"] = filename.upper()


        # Save final cleaned file
        save_path = f"{OUTPUT_FOLDER}/{filename}.csv"
        temp.to_csv(save_path, index=True)

        print(f"✓ Saved → {save_path} ({len(temp)} rows)")

    except KeyError as e:
        print(f"✗ Error processing {yf_symbol}: {e}")
        print(f"  Available tickers in data: {df.columns.levels[0].tolist()}")
    except Exception as e:
        print(f"✗ Unexpected error processing {yf_symbol}: {e}")

print("\n" + "="*60)
print("Done! All cleaned datasets created successfully.")
print(f"Output location: {os.path.abspath(OUTPUT_FOLDER)}")