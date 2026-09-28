import joblib
import pandas as pd
import yfinance as yf
import json
import os
import logging
from src.features import create_features_and_target

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/best_model.pkl')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), '../models/feature_columns.json')

_MODEL_CACHE = {}

def load_trained_model():
    """
    Loads and caches the trained model pipeline and feature columns.
    """
    if 'model' not in _MODEL_CACHE:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
            raise FileNotFoundError("Trained model or feature columns specification not found. Run main.py first.")
        _MODEL_CACHE['model'] = joblib.load(MODEL_PATH)
        with open(FEATURES_PATH, 'r') as f:
            _MODEL_CACHE['feature_columns'] = json.load(f)
    return _MODEL_CACHE['model'], _MODEL_CACHE['feature_columns']

def fetch_and_predict(ticker: str):
    """
    Fetches recent historical data for a ticker, computes features identically to Phase 1,
    and performs model inference on the latest available observation.
    
    Returns:
        dict: Complete prediction and market data dictionary.
    """
    # Fetch historical data (6 months provides sufficient window for 20-day rolling indicators + plotting)
    try:
        df = yf.download(ticker, period='6mo', progress=False)
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Yahoo Finance: {str(e)}")
        
    if df is None or df.empty:
        raise ValueError(f"No market data returned for ticker '{ticker}'.")
    
    # --- Robust MultiIndex column normalization ---
    # yfinance returns MultiIndex columns like ('Close', 'AAPL') with level names ['Price', 'Ticker'].
    # The actual OHLCV names are at level 0, and the ticker symbol is at level 1.
    # We must handle: single-ticker MultiIndex, multi-ticker MultiIndex, and plain Index.
    if isinstance(df.columns, pd.MultiIndex):
        # For single-ticker downloads, droplevel the ticker level to get clean column names
        # Identify which level contains the ticker symbol
        level_values = [df.columns.get_level_values(i).unique().tolist() for i in range(df.columns.nlevels)]
        ticker_level = None
        for i, vals in enumerate(level_values):
            if ticker.upper() in [v.upper() for v in vals if isinstance(v, str)]:
                ticker_level = i
                break
        
        if ticker_level is not None:
            df.columns = df.columns.droplevel(ticker_level)
        else:
            # Fallback: take the level that contains known OHLCV names
            for i, vals in enumerate(level_values):
                lower_vals = [v.lower() for v in vals if isinstance(v, str)]
                if 'close' in lower_vals and 'open' in lower_vals:
                    df.columns = df.columns.get_level_values(i)
                    break
            else:
                # Last resort: flatten to level 0
                df.columns = df.columns.get_level_values(0)
        
        logger.debug(f"[{ticker}] Flattened MultiIndex columns to: {df.columns.tolist()}")
        
    df['Ticker'] = ticker
    df = df.reset_index()
    
    # Ensure standard column naming (case-insensitive normalization)
    rename_dict = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower == 'date':
            rename_dict[col] = 'Date'
        elif col_lower == 'open':
            rename_dict[col] = 'Open'
        elif col_lower == 'high':
            rename_dict[col] = 'High'
        elif col_lower == 'low':
            rename_dict[col] = 'Low'
        elif col_lower == 'close':
            rename_dict[col] = 'Close'
        elif col_lower == 'volume':
            rename_dict[col] = 'Volume'
    df = df.rename(columns=rename_dict)
    
    # Validate all required OHLCV columns are present
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Market data for '{ticker}' is incomplete. "
            f"Missing field(s): {', '.join(missing_cols)}. "
            f"Available columns: {[c for c in df.columns if c not in ['Ticker']]}"
        )
    
    # Calculate features identically to training (is_training=False retains the latest prediction row)
    features_df = create_features_and_target(df, is_training=False)
    
    if features_df.empty or len(features_df) < 2:
        raise ValueError(f"Insufficient historical data to compute technical features for '{ticker}'.")
        
    # Extract the latest valid feature row
    latest_row = features_df.iloc[-1:]
    
    # Load model and expected feature columns
    model, feature_columns = load_trained_model()
    
    # Verify all expected columns are present
    missing_cols = [col for col in feature_columns if col not in latest_row.columns]
    if missing_cols:
        raise KeyError(f"Missing required feature columns: {missing_cols}")
        
    # Ensure exact column ordering
    X = latest_row[feature_columns]
    
    # Inference
    pred = int(model.predict(X)[0])
    
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0]
        prob_up = float(prob[1])
        prob_down = float(prob[0])
    else:
        prob_up = 1.0 if pred == 1 else 0.0
        prob_down = 1.0 if pred == 0 else 0.0
        
    latest_close = float(features_df['Close'].iloc[-1])
    prev_close = float(features_df['Close'].iloc[-2])
    daily_change = latest_close - prev_close
    daily_return = float(features_df['Daily_Return'].iloc[-1])
    daily_change_pct = daily_return * 100.0
    volume = int(features_df['Volume'].iloc[-1])
    
    date_val = features_df['Date'].iloc[-1]
    if hasattr(date_val, 'strftime'):
        date_str = date_val.strftime('%Y-%m-%d')
    else:
        date_str = str(date_val)[:10]
        
    return {
        "ticker": ticker,
        "latest_date": date_str,
        "latest_close": latest_close,
        "previous_close": prev_close,
        "daily_change": daily_change,
        "daily_return": daily_return,
        "daily_change_pct": daily_change_pct,
        "volume": volume,
        "prediction": "UP" if pred == 1 else "DOWN",
        "probability_up": prob_up,
        "probability_down": prob_down,
        "historical_data": df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    }

if __name__ == '__main__':
    for t in ["AAPL", "MSFT", "GOOGL"]:
        res = fetch_and_predict(t)
        print(f"\n--- {t} ---")
        print(f"Date: {res['latest_date']} | Close: ${res['latest_close']:.2f} | Prev: ${res['previous_close']:.2f} | Change: {res['daily_change_pct']:+.2f}%")
        print(f"Prediction: {res['prediction']} (UP: {res['probability_up']:.1%}, DOWN: {res['probability_down']:.1%})")
