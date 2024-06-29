import os
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(ticker, data_file, start_date, end_date, seq_length=5, test_size=0.2):
    """
    Load and preprocess stock data for training.

    Args:
        ticker (str): Stock ticker symbol.
        data_file (str): Path to the data file.
        start_date (str): Start date for data fetching.
        end_date (str): End date for data fetching.
        seq_length (int): Sequence length for RNN.
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        X_train_full, X_test, y_train_full, y_test, close_scaler
    """
    if not os.path.exists(data_file):
        data = yf.download(ticker, start=start_date, end=end_date)
        data.to_csv(data_file)
    else:
        data = pd.read_csv(data_file, index_col=0)
        data.index = pd.to_datetime(data.index)
        data = data.loc[start_date:end_date]

    # Feature Engineering
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['RSI'] = 100 - (100 / (1 + data['Close'].diff().rolling(window=14).apply(lambda x: np.mean(np.maximum(x, 0)) / np.mean(np.abs(x)), raw=False)))
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data.dropna(inplace=True)

    # Normalization
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'RSI', 'MACD']])
    close_scaler = StandardScaler()
    close_prices = data[['Close']]
    close_scaler.fit(close_prices)

    # Prepare data for training
    X = []
    y = []
    for i in range(len(data_normalized) - seq_length):
        X.append(data_normalized[i:i + seq_length])
        y.append(data_normalized[i + seq_length][3])  # Close price is at index 3
    X = np.array(X)
    y = np.array(y)

    # Split data into train and test sets
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train_full, X_test, y_train_full, y_test, close_scaler
