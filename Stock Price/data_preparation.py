"""
Data Preparation Module for Stock Price Prediction
Downloads RELIANCE.NS data and prepares it for SNN/LSTM training
"""

import yfinance as yf
import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
import os
import pickle

def download_stock_data(ticker="RELIANCE.NS", start="2025-03-04", end="2026-03-04"):
    """
    Download stock data from Yahoo Finance
    
    Args:
        ticker: Stock ticker symbol (default: RELIANCE.NS)
        start: Start date (default: 2025-03-04)
        end: End date (default: 2026-03-04)
    
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Downloading {ticker} data from {start} to {end}...")
    data = yf.download(ticker, start=start, end=end, progress=False)
    
    # Flatten multi-level columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    print(f"Downloaded {len(data)} trading days")
    return data


def add_technical_indicators(data):
    """
    Add technical indicators to the stock data
    
    Args:
        data: DataFrame with OHLCV columns
    
    Returns:
        DataFrame with added technical indicators
    """
    df = data.copy()
    
    # Ensure we have proper 1D float64 arrays for TA-Lib (TA-Lib requires double precision)
    close = np.asarray(df['Close'], dtype=np.float64).flatten()
    high = np.asarray(df['High'], dtype=np.float64).flatten()
    low = np.asarray(df['Low'], dtype=np.float64).flatten()
    open_price = np.asarray(df['Open'], dtype=np.float64).flatten()
    volume = np.asarray(df['Volume'], dtype=np.float64).flatten()
    
    # Basic features
    df['Returns'] = df['Close'].pct_change()
    df['High_Low'] = (df['High'] - df['Low']) / df['Close']
    df['Close_Open'] = (df['Close'] - df['Open']) / df['Open']
    
    # Technical indicators
    df['RSI'] = talib.RSI(close, timeperiod=14)
    rsi_values = np.asarray(df['RSI'], dtype=np.float64).flatten()
    df['RSI_SMA'] = talib.SMA(rsi_values, timeperiod=10)
    
    # MACD
    macd, macd_signal, macd_hist = talib.MACD(close)
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Hist'] = macd_hist
    
    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(close, timeperiod=20)
    df['BB_Upper'] = upper
    df['BB_Middle'] = middle
    df['BB_Lower'] = lower
    df['BB_Width'] = (upper - lower) / middle
    
    # Moving averages
    df['SMA_10'] = talib.SMA(close, timeperiod=10)
    df['SMA_20'] = talib.SMA(close, timeperiod=20)
    df['SMA_50'] = talib.SMA(close, timeperiod=50)
    df['EMA_12'] = talib.EMA(close, timeperiod=12)
    df['EMA_26'] = talib.EMA(close, timeperiod=26)
    
    # Volume indicators
    df['Volume_SMA'] = talib.SMA(volume, timeperiod=20)
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Momentum indicators
    df['MOM'] = talib.MOM(close, timeperiod=10)
    df['ROC'] = talib.ROC(close, timeperiod=10)
    
    # Volatility
    df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    print(f"Added technical indicators. Total features: {len(df.columns)}")
    return df


def create_target(data, target_type='direction'):
    """
    Create target variable for prediction
    
    Args:
        data: DataFrame with Close price
        target_type: 'direction' for up/down classification
    
    Returns:
        DataFrame with Target column
    """
    df = data.copy()
    
    if target_type == 'direction':
        # Binary classification: 1=next day up, 0=down
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df.dropna(inplace=True)
    
    up_days = df['Target'].sum()
    total_days = len(df)
    print(f"Target created: {up_days}/{total_days} up days ({up_days/total_days*100:.1f}%)")
    
    return df


def split_data(data, train_ratio=0.8):
    """
    Split data into train and test sets
    
    Args:
        data: DataFrame with features and target
        train_ratio: Ratio of training data (default: 0.8)
    
    Returns:
        Tuple of (train_df, test_df)
    """
    split_idx = int(len(data) * train_ratio)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    
    print(f"Train set: {len(train)} days")
    print(f"Test set: {len(test)} days")
    
    return train, test


class StockSequenceDataset(Dataset):
    """
    PyTorch Dataset for stock sequences
    """
    
    def __init__(self, df, seq_len=20, feature_cols=None, fit_scaler=True, scaler=None):
        """
        Args:
            df: DataFrame with features and Target column
            seq_len: Length of input sequences
            feature_cols: List of feature column names (if None, use all except Target)
            fit_scaler: Whether to fit the scaler on this data
            scaler: Pre-fitted scaler (if fit_scaler=False)
        """
        self.seq_len = seq_len
        
        # Select features
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != 'Target']
        self.feature_cols = feature_cols
        
        X = df[feature_cols].values
        y = df['Target'].values
        
        # Normalize features
        if fit_scaler:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            if scaler is None:
                raise ValueError("Must provide scaler if fit_scaler=False")
            self.scaler = scaler
            X = self.scaler.transform(X)
        
        # Create sequences
        self.X, self.y = [], []
        for i in range(len(X) - seq_len):
            self.X.append(X[i:i+seq_len])
            self.y.append(y[i+seq_len])
        
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        
        print(f"Created dataset: {len(self.X)} sequences of shape {self.X.shape[1:]}")
        print(f"Positive class ratio: {self.y.mean():.3f}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.LongTensor([self.y[idx]])[0]
    
    def get_scaler(self):
        return self.scaler


def prepare_full_dataset(ticker="RELIANCE.NS", start="2025-03-04", end="2026-03-04", 
                         seq_len=20, train_ratio=0.8, save_path=None):
    """
    Complete pipeline to prepare dataset for training
    
    Args:
        ticker: Stock ticker symbol
        start: Start date
        end: End date
        seq_len: Sequence length for time series
        train_ratio: Train/test split ratio
        save_path: Path to save processed data (optional)
    
    Returns:
        Dictionary with train_loader, test_loader, feature_cols, scaler
    """
    # Download data
    data = download_stock_data(ticker, start, end)
    
    # Add technical indicators
    data = add_technical_indicators(data)
    
    # Create target
    data = create_target(data, target_type='direction')
    
    # Split data
    train_df, test_df = split_data(data, train_ratio)
    
    # Create datasets
    print("\nCreating training dataset...")
    train_dataset = StockSequenceDataset(train_df, seq_len=seq_len, fit_scaler=True)
    
    print("\nCreating testing dataset...")
    test_dataset = StockSequenceDataset(test_df, seq_len=seq_len, 
                                       fit_scaler=False, 
                                       scaler=train_dataset.get_scaler(),
                                       feature_cols=train_dataset.feature_cols)
    
    # Save processed data if requested
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
        # Save dataframes
        train_df.to_csv(os.path.join(save_path, 'train_data.csv'))
        test_df.to_csv(os.path.join(save_path, 'test_data.csv'))
        
        # Save scaler
        with open(os.path.join(save_path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(train_dataset.get_scaler(), f)
        
        # Save metadata
        metadata = {
            'ticker': ticker,
            'start': start,
            'end': end,
            'seq_len': seq_len,
            'train_ratio': train_ratio,
            'feature_cols': train_dataset.feature_cols,
            'num_features': len(train_dataset.feature_cols)
        }
        with open(os.path.join(save_path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\nData saved to {save_path}")
    
    return {
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'feature_cols': train_dataset.feature_cols,
        'scaler': train_dataset.get_scaler(),
        'num_features': len(train_dataset.feature_cols),
        'train_df': train_df,
        'test_df': test_df
    }


if __name__ == "__main__":
    # Example usage
    result = prepare_full_dataset(
        ticker="RELIANCE.NS",
        start="2025-03-04",
        end="2026-03-04",
        seq_len=20,
        train_ratio=0.8,
        save_path="./processed_data"
    )
    
    print(f"\nDataset prepared successfully!")
    print(f"Number of features: {result['num_features']}")
    print(f"Feature columns: {result['feature_cols'][:5]}... (showing first 5)")