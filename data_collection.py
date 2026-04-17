import yfinance as yf
import pandas as pd

tickers = ["AMZN", "NVDA", "GOOGL","MSFT","^GSPC"] # S&P500 = ^GSPC

data = yf.download(tickers, start="2022-01-01", end="2024-12-31", group_by='ticker')

# Save as CSV for submission
data.to_csv("raw_stock_data.csv")