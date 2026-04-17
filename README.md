# Artificial-Intelligence-and-Machine-Learning-Group-Assignment
Machine learning project predicting stock price direction using Logistic Regression and LSTM. Includes feature engineering, time-series modelling, and evaluation (ROC-AUC, F1). Highlights challenges like class imbalance and trend bias in financial prediction


This project develops and evaluates machine learning models to predict stock price direction using historical OHLCV (Open, High, Low, Close, Volume) data spanning 2022–2024. The objective was to assess the effectiveness of both traditional machine learning and deep learning approaches in a real-world financial forecasting context.

The project implements two core models: a Logistic Regression classifier as a baseline and a Long Short-Term Memory (LSTM) neural network to capture temporal dependencies in time-series data. A full end-to-end pipeline was built, including data cleaning, preprocessing, and feature engineering. Key engineered features include daily returns, moving averages (SMA), and volatility measures, designed to capture both short-term trends and market behaviour.

Exploratory Data Analysis (EDA) was conducted to understand price patterns, class distribution, and data quality issues. Particular attention was given to class imbalance and the inherent noise within financial data. Models were evaluated using appropriate metrics such as ROC-AUC, F1-score, precision, and recall, rather than relying solely on accuracy.

Time-series validation techniques were applied to avoid data leakage and ensure realistic model performance. The results highlight key challenges in financial prediction, including model overfitting, bias toward dominant trends, and the limited predictive power of historical price data alone.

Overall, the project demonstrates both the practical implementation of machine learning pipelines and a critical understanding of the limitations of applying AI in financial markets.
