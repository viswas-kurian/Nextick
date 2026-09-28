import pandas as pd
import numpy as np
import os

def create_features_and_target(df, is_training=True):
    """
    Creates features and binary target for the Nextick ML pipeline.
    Ensures strict leakage prevention by using ONLY past data (t <= T) for all features.
    
    Multi-stock handling:
    - Grouped by ticker so rolling and lag computations never cross ticker boundaries.
    - Added ticker-independent normalized features so models can compare stocks of 
      different price scales (e.g. $50 stock vs $500 stock).
    
    Target:
    1 if Next_Close > Current_Close else 0
    
    Parameters:
        df (pd.DataFrame): Raw OHLCV market data containing ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker'].
        is_training (bool): If True, computes the shifted target and drops the last row per ticker.
                            If False (inference), retains the latest valid row for real-time prediction.
    """
    df = df.copy()
    
    # Parse dates and sort chronologically
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)
    
    processed_dfs = []
    
    for ticker, group in df.groupby('Ticker'):
        # Sort chronologically within ticker group to guarantee temporal integrity
        group = group.sort_values('Date').copy()
        
        # --- 1. BASE RETURNS & MOMENTUM ---
        group['Daily_Return'] = group['Close'].pct_change()
        group['Lag1_Return'] = group['Daily_Return'].shift(1)
        group['Lag2_Return'] = group['Daily_Return'].shift(2)
        group['Return_5d'] = group['Close'].pct_change(periods=5)
        
        # --- 2. MOVING AVERAGES (Past data only) ---
        group['SMA_5'] = group['Close'].rolling(window=5).mean()
        group['SMA_10'] = group['Close'].rolling(window=10).mean()
        group['SMA_20'] = group['Close'].rolling(window=20).mean()
        group['EMA_10'] = group['Close'].ewm(span=10, adjust=False).mean()
        group['EMA_20'] = group['Close'].ewm(span=20, adjust=False).mean()
        
        # --- 3. ROLLING VOLATILITY ---
        group['Rolling_Vol_10'] = group['Daily_Return'].rolling(window=10).std()
        group['Rolling_Vol_20'] = group['Daily_Return'].rolling(window=20).std()
        
        # --- 4. VOLUME INDICATORS ---
        group['Volume_Change'] = group['Volume'].pct_change()
        group['Volume_MA_5'] = group['Volume'].rolling(window=5).mean()
        group['Volume_to_MA5'] = (group['Volume'] / (group['Volume_MA_5'] + 1e-8)) - 1.0
        
        # --- 5. TICKER-INDEPENDENT NORMALIZED FEATURES ---
        # These features describe relative distance and momentum without depending on raw dollar values
        group['Close_to_SMA5'] = (group['Close'] - group['SMA_5']) / group['SMA_5']
        group['Close_to_SMA10'] = (group['Close'] - group['SMA_10']) / group['SMA_10']
        group['Close_to_SMA20'] = (group['Close'] - group['SMA_20']) / group['SMA_20']
        group['SMA5_to_SMA20'] = (group['SMA_5'] - group['SMA_20']) / group['SMA_20']
        group['EMA10_to_EMA20'] = (group['EMA_10'] - group['EMA_20']) / group['EMA_20']
        group['High_Low_Range'] = (group['High'] - group['Low']) / group['Close']
        group['Open_Close_Range'] = (group['Close'] - group['Open']) / group['Open']
        
        # --- 6. TARGET CREATION ---
        if is_training:
            # The target is whether the NEXT day's close is higher than TODAY's close.
            # Strictly derived by shifting Close backward by 1 observation.
            group['Next_Close'] = group['Close'].shift(-1)
            group['Target'] = (group['Next_Close'] > group['Close']).astype(int)
            
            # The last observation for each ticker has no next-day close; must be dropped
            group = group.dropna(subset=['Next_Close'])
            group = group.drop(columns=['Next_Close'])
            
        # Drop rows with NaN resulting from the rolling calculations (first 20 rows of each ticker)
        group = group.dropna().reset_index(drop=True)
        processed_dfs.append(group)
        
    if not processed_dfs:
        raise ValueError("Feature processing resulted in an empty dataset.")
        
    final_df = pd.concat(processed_dfs, ignore_index=True)
    final_df = final_df.sort_values(by=['Date', 'Ticker']).reset_index(drop=True)
    
    return final_df

if __name__ == '__main__':
    raw_path = 'data/raw/combined_raw.csv'
    if not os.path.exists(raw_path):
        from src.data_loader import load_data
        load_data()
    raw_df = pd.read_csv(raw_path)
    processed_df = create_features_and_target(raw_df, is_training=True)
    processed_df.to_csv('data/processed/features.csv', index=False)
    print(f"Feature engineering complete. Shape: {processed_df.shape}")
    print(f"Target distribution:\n{processed_df['Target'].value_counts(normalize=True)}")
