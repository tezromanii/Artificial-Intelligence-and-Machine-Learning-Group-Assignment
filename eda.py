import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for prettier plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


DATASETS_FOLDER = "Groupwork/datasets"
FIGURES_FOLDER = "Groupwork/figures"

# Stocks we're analyzing
stocks = ["amzn", "nvda", "googl", "msft", "sp500"]

# Create figures folder if it doesn't exist
os.makedirs(FIGURES_FOLDER, exist_ok=True)


# Load all datasets

print("="*60)
print("LOADING STOCK DATA...")
print("="*60)

# Dictionary to store each stock's data
stock_data = {}

for stock in stocks:
    file_path = f"{DATASETS_FOLDER}/{stock}.csv"
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    stock_data[stock] = df
    print(f"✓ Loaded {stock.upper()}: {len(df)} rows, {len(df.columns)} columns")

print("\n")


# 1. DESCRIPTIVE STATISTICS

print("="*60)
print("1. DESCRIPTIVE STATISTICS")
print("="*60)

for stock in stocks:
    df = stock_data[stock]
    print(f"\n📊 {stock.upper()} Summary Statistics:")
    print("-" * 50)
    # Show key columns only
    key_cols = ["Close", "Daily_Return_%", "Volatility", "Volume"]
    print(df[key_cols].describe())

print("\n")


# 2. CLOSING PRICE TRENDS OVER TIME

print("="*60)
print("2. ANALYZING PRICE TRENDS...")
print("="*60)

plt.figure(figsize=(14, 8))

for stock in stocks:
    df = stock_data[stock]
    plt.plot(df.index, df['Close'], label=stock.upper(), linewidth=2)

plt.title('Stock Closing Prices Over Time (2022-2024)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Closing Price ($)', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

save_path = f"{FIGURES_FOLDER}/01_price_trends.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {save_path}")
plt.close()


# 3. DAILY RETURNS DISTRIBUTION

print("="*60)
print("3. ANALYZING DAILY RETURNS DISTRIBUTION...")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, stock in enumerate(stocks):
    df = stock_data[stock]
    
    # Plot histogram
    axes[idx].hist(df['Daily_Return_%'].dropna(), bins=50, 
                   color='skyblue', edgecolor='black', alpha=0.7)
    axes[idx].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero return')
    axes[idx].set_title(f'{stock.upper()} Daily Returns', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Daily Return (%)', fontsize=10)
    axes[idx].set_ylabel('Frequency', fontsize=10)
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

# Remove extra subplot
axes[5].axis('off')

plt.suptitle('Distribution of Daily Returns', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()

save_path = f"{FIGURES_FOLDER}/02_returns_distribution.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {save_path}")
plt.close()


# 4. VOLATILITY COMPARISON (Box Plot)

print("="*60)
print("4. COMPARING VOLATILITY ACROSS STOCKS...")
print("="*60)

# Collect volatility data for all stocks
volatility_data = []
for stock in stocks:
    df = stock_data[stock]
    temp_df = pd.DataFrame({
        'Stock': stock.upper(),
        'Volatility': df['Volatility'].dropna()
    })
    volatility_data.append(temp_df)

# Combine all data
volatility_combined = pd.concat(volatility_data, ignore_index=True)

plt.figure(figsize=(12, 6))
sns.boxplot(data=volatility_combined, x='Stock', y='Volatility', palette='Set2')
plt.title('Volatility Comparison Across Stocks', fontsize=16, fontweight='bold')
plt.xlabel('Stock', fontsize=12)
plt.ylabel('Daily Volatility (%)', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

save_path = f"{FIGURES_FOLDER}/03_volatility_comparison.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {save_path}")
plt.close()


# 5. CORRELATION ANALYSIS

print("="*60)
print("5. ANALYZING CORRELATIONS BETWEEN STOCKS...")
print("="*60)

# Create a dataframe with closing prices of all stocks
close_prices = pd.DataFrame()
for stock in stocks:
    close_prices[stock.upper()] = stock_data[stock]['Close']

# Calculate correlation matrix
correlation_matrix = close_prices.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix: Stock Closing Prices', fontsize=16, fontweight='bold')
plt.tight_layout()

save_path = f"{FIGURES_FOLDER}/04_correlation_heatmap.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {save_path}")
plt.close()

print("\nCorrelation Matrix:")
print(correlation_matrix)


# 6. MOVING AVERAGES VISUALIZATION

print("\n" + "="*60)
print("6. VISUALIZING MOVING AVERAGES...")
print("="*60)

# Pick one stock for detailed MA analysis (e.g., NVDA)
sample_stock = "nvda"
df_sample = stock_data[sample_stock]

# Plot last 200 days for clarity
df_plot = df_sample.tail(200)

plt.figure(figsize=(14, 7))
plt.plot(df_plot.index, df_plot['Close'], label='Close Price', linewidth=2, color='black')
plt.plot(df_plot.index, df_plot['SMA5'], label='SMA5', linewidth=1.5, linestyle='--', color='blue')
plt.plot(df_plot.index, df_plot['SMA10'], label='SMA10', linewidth=1.5, linestyle='--', color='orange')
plt.plot(df_plot.index, df_plot['SMA20'], label='SMA20', linewidth=1.5, linestyle='--', color='red')

plt.title(f'{sample_stock.upper()} - Closing Price with Moving Averages (Last 200 Days)', 
          fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price ($)', fontsize=12)
plt.legend(loc='best', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

save_path = f"{FIGURES_FOLDER}/05_moving_averages_{sample_stock}.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {save_path}")
plt.close()


# 7. TARGET DISTRIBUTION (UP vs DOWN days)

print("="*60)
print("7. ANALYZING TARGET DISTRIBUTION...")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, stock in enumerate(stocks):
    df = stock_data[stock]
    
    # Count up days (1) vs down days (0)
    target_counts = df['Target'].value_counts()
    
    # Plot bar chart
    axes[idx].bar(['Down (0)', 'Up (1)'], 
                  [target_counts.get(0, 0), target_counts.get(1, 0)],
                  color=['red', 'green'], alpha=0.7, edgecolor='black')
    axes[idx].set_title(f'{stock.upper()} Target Distribution', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Number of Days', fontsize=10)
    axes[idx].grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    total = len(df)
    for i, count in enumerate([target_counts.get(0, 0), target_counts.get(1, 0)]):
        percentage = (count / total) * 100
        axes[idx].text(i, count + 5, f'{percentage:.1f}%', 
                      ha='center', fontsize=10, fontweight='bold')

# Remove extra subplot
axes[5].axis('off')

plt.suptitle('Target Distribution: Up Days vs Down Days', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()

save_path = f"{FIGURES_FOLDER}/06_target_distribution.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {save_path}")
plt.close()

# Print summary
print("\nTarget Balance Summary:")
for stock in stocks:
    df = stock_data[stock]
    target_counts = df['Target'].value_counts()
    up_pct = (target_counts.get(1, 0) / len(df)) * 100
    down_pct = (target_counts.get(0, 0) / len(df)) * 100
    print(f"{stock.upper()}: Up={up_pct:.1f}%, Down={down_pct:.1f}%")


# 8. VOLUME ANALYSIS

print("\n" + "="*60)
print("8. ANALYZING TRADING VOLUME PATTERNS...")
print("="*60)

plt.figure(figsize=(14, 8))

for stock in stocks:
    df = stock_data[stock]
    # Use a rolling average to smooth volume data
    plt.plot(df.index, df['Volume'].rolling(20).mean(), label=stock.upper(), linewidth=2)

plt.title('Trading Volume Over Time (20-Day Moving Average)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Volume', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

save_path = f"{FIGURES_FOLDER}/07_volume_trends.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {save_path}")
plt.close()


# FINAL SUMMARY

print("\n" + "="*60)
print("EDA COMPLETE! 🎉")
print("="*60)
print(f"\n✓ All visualizations saved to: {os.path.abspath(FIGURES_FOLDER)}")
print("\nGenerated figures:")
print("  1. Price trends over time")
print("  2. Daily returns distribution")
print("  3. Volatility comparison")
print("  4. Correlation heatmap")
print("  5. Moving averages analysis")
print("  6. Target distribution")
print("  7. Volume trends")
print("="*60)