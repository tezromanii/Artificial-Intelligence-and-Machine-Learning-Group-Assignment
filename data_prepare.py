import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

DATASETS_FOLDER = "Groupwork/datasets"
OUTPUT_FOLDER = "Groupwork/processed_data"

stocks = ["amzn", "nvda", "googl", "msft", "sp500"]
TRAIN_RATIO = 0.80

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "Daily_Return_%", "SMA5", "SMA10", "SMA20",
    "Price_Distance_SMA10", "Volatility", "Volume_Change_%"
]

TARGET_COL = "Target"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("="*60)
print("LOADING CLEANED DATASETS...")
print("="*60)

all_data = []

for stock in stocks:
    file_path = f"{DATASETS_FOLDER}/{stock}.csv"
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df['Stock_Symbol'] = stock.upper()
    all_data.append(df)
    print(f"Loaded {stock.upper()}: {len(df)} rows")

combined_df = pd.concat(all_data, ignore_index=False)
combined_df = combined_df.sort_index()

print(f"\nCombined dataset: {len(combined_df)} total rows")
print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")

print("\n" + "="*60)
print("PROCESSING EACH STOCK INDIVIDUALLY...")
print("="*60)

scaler_dict = {}

for stock in stocks:
    print(f"\n{'='*50}")
    print(f"Processing {stock.upper()}...")
    print(f"{'='*50}")
    
    stock_df = combined_df[combined_df['Stock_Symbol'] == stock.upper()].copy()
    stock_df = stock_df.sort_index()
    
    X = stock_df[FEATURE_COLS].copy()
    y = stock_df[TARGET_COL].copy()
    
    print(f"  Original shape: {X.shape}")
    
    # Replace infinite values with NaN and drop
    X = X.replace([np.inf, -np.inf], np.nan)
    
    nan_count = X.isnull().sum().sum()
    if nan_count > 0:
        print(f"  Warning: Found {nan_count} NaN/Inf values, cleaning...")
        
        nan_cols = X.isnull().sum()
        nan_cols = nan_cols[nan_cols > 0]
        if len(nan_cols) > 0:
            print(f"    Columns with NaN/Inf:")
            for col, count in nan_cols.items():
                print(f"      - {col}: {count} values")
        
        nan_mask = X.isnull().any(axis=1)
        X = X[~nan_mask]
        y = y[~nan_mask]
        print(f"  After cleaning: {X.shape}")
    
    # Chronological train-test split
    split_idx = int(len(X) * TRAIN_RATIO)
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    print(f"\n  Train-Test Split (Chronological):")
    print(f"    Training set: {len(X_train)} samples ({TRAIN_RATIO*100:.0f}%)")
    print(f"    Test set: {len(X_test)} samples ({(1-TRAIN_RATIO)*100:.0f}%)")
    print(f"    Train period: {X_train.index.min().date()} to {X_train.index.max().date()}")
    print(f"    Test period: {X_test.index.min().date()} to {X_test.index.max().date()}")
    
    train_up = (y_train == 1).sum()
    train_down = (y_train == 0).sum()
    test_up = (y_test == 1).sum()
    test_down = (y_test == 0).sum()
    
    print(f"\n  Target Distribution:")
    print(f"    Train: Up={train_up} ({train_up/len(y_train)*100:.1f}%), Down={train_down} ({train_down/len(y_train)*100:.1f}%)")
    print(f"    Test:  Up={test_up} ({test_up/len(y_test)*100:.1f}%), Down={test_down} ({test_down/len(y_test)*100:.1f}%)")
    
    print(f"\n  Applying StandardScaler...")
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=FEATURE_COLS, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=FEATURE_COLS, index=X_test.index)
    
    print(f"    Scaled training features: mean = 0, std = 1")
    print(f"      Example - Close price: mean={X_train_scaled['Close'].mean():.4f}, std={X_train_scaled['Close'].std():.4f}")
    
    scaler_dict[stock] = scaler
    
    stock_output_folder = f"{OUTPUT_FOLDER}/{stock}"
    os.makedirs(stock_output_folder, exist_ok=True)
    
    X_train_scaled.to_csv(f"{stock_output_folder}/X_train.csv")
    X_test_scaled.to_csv(f"{stock_output_folder}/X_test.csv")
    y_train.to_csv(f"{stock_output_folder}/y_train.csv")
    y_test.to_csv(f"{stock_output_folder}/y_test.csv")
    joblib.dump(scaler, f"{stock_output_folder}/scaler.pkl")
    
    print(f"\n  Saved processed data to: {stock_output_folder}/")
    print(f"    - X_train.csv ({X_train_scaled.shape})")
    print(f"    - X_test.csv ({X_test_scaled.shape})")
    print(f"    - y_train.csv ({y_train.shape})")
    print(f"    - y_test.csv ({y_test.shape})")
    print(f"    - scaler.pkl")

print("\n" + "="*60)
print("CREATING COMBINED DATASET (ALL STOCKS)...")
print("="*60)

X_all = combined_df[FEATURE_COLS].copy()
y_all = combined_df[TARGET_COL].copy()

X_all = X_all.replace([np.inf, -np.inf], np.nan)
nan_mask = X_all.isnull().any(axis=1)
X_all = X_all[~nan_mask]
y_all = y_all[~nan_mask]

print(f"Combined cleaned data: {len(X_all)} samples")

split_idx = int(len(X_all) * TRAIN_RATIO)

X_train_all = X_all.iloc[:split_idx]
X_test_all = X_all.iloc[split_idx:]
y_train_all = y_all.iloc[:split_idx]
y_test_all = y_all.iloc[split_idx:]

scaler_all = StandardScaler()
scaler_all.fit(X_train_all)

X_train_all_scaled = pd.DataFrame(
    scaler_all.transform(X_train_all),
    columns=FEATURE_COLS,
    index=X_train_all.index
)
X_test_all_scaled = pd.DataFrame(
    scaler_all.transform(X_test_all),
    columns=FEATURE_COLS,
    index=X_test_all.index
)

combined_folder = f"{OUTPUT_FOLDER}/combined"
os.makedirs(combined_folder, exist_ok=True)

X_train_all_scaled.to_csv(f"{combined_folder}/X_train.csv")
X_test_all_scaled.to_csv(f"{combined_folder}/X_test.csv")
y_train_all.to_csv(f"{combined_folder}/y_train.csv")
y_test_all.to_csv(f"{combined_folder}/y_test.csv")
joblib.dump(scaler_all, f"{combined_folder}/scaler.pkl")

print(f"Saved combined dataset to: {combined_folder}/")
print(f"  Training: {len(X_train_all_scaled)} samples")
print(f"  Test: {len(X_test_all_scaled)} samples")

print("\n" + "="*60)
print("DATA PREPARATION SUMMARY")
print("="*60)

print(f"\nTotal Dataset:")
print(f"  - Total samples: {len(X_all)}")
print(f"  - Date range: {X_all.index.min().date()} to {X_all.index.max().date()}")
print(f"  - Number of features: {len(FEATURE_COLS)}")
print(f"  - Number of stocks: {len(stocks)}")

print(f"\nPer-Stock Statistics:")
for stock in stocks:
    stock_folder = f"{OUTPUT_FOLDER}/{stock}"
    train_file = f"{stock_folder}/X_train.csv"
    if os.path.exists(train_file):
        train_df = pd.read_csv(train_file, index_col=0)
        test_df = pd.read_csv(f"{stock_folder}/X_test.csv", index_col=0)
        print(f"  {stock.upper()}: Train={len(train_df)}, Test={len(test_df)}")

print(f"\nTrain-Test Split:")
print(f"  - Training: {TRAIN_RATIO*100:.0f}% of data (chronological)")
print(f"  - Testing: {(1-TRAIN_RATIO)*100:.0f}% of data (most recent)")

print(f"\nFeature Scaling:")
print(f"  - Method: StandardScaler (mean=0, std=1)")
print(f"  - Fitted on training data only")
print(f"  - Applied to both train and test sets")

print(f"\nOutput Location:")
print(f"  {os.path.abspath(OUTPUT_FOLDER)}/")

print("\n" + "="*60)
print("DATA PREPARATION COMPLETE")
print("="*60)