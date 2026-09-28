import yfinance as yf
import pandas as pd
import os
import datetime

DEFAULT_TICKERS = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "ORCL", "ADBE", "CSCO", "INTC",
    # Financial
    "JPM", "V", "MA", "BAC", "WFC", "GS",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY",
    # Consumer
    "WMT", "COST", "KO", "PEP", "MCD", "PG", "NKE",
    # Industrial & Energy
    "CAT", "GE", "XOM", "CVX", "BA", "HON"
]

def load_data(tickers=None, start_date="2015-01-01", end_date=None, raw_data_path="data/raw/"):
    """
    Downloads historical daily stock data from Yahoo Finance for a diversified universe of stocks.
    Handles failed tickers gracefully and saves individual and combined raw CSV files.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)

    all_data = []
    successful_tickers = []
    failed_tickers = []
    
    print(f"Starting data download for {len(tickers)} tickers from {start_date} to {end_date or 'today'}...")
    
    for ticker in tickers:
        print(f"Downloading {ticker}...", end=" ", flush=True)
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df is None or df.empty or len(df) < 50:
                print("FAILED (no/insufficient data)")
                failed_tickers.append(ticker)
                continue
            
            # Flatten MultiIndex columns if present (yfinance format changes)
            if isinstance(df.columns, pd.MultiIndex):
                # Identify the level containing the ticker symbol and drop it
                level_values = [df.columns.get_level_values(i).unique().tolist() for i in range(df.columns.nlevels)]
                ticker_level = None
                for i, vals in enumerate(level_values):
                    if ticker.upper() in [v.upper() for v in vals if isinstance(v, str)]:
                        ticker_level = i
                        break
                if ticker_level is not None:
                    df.columns = df.columns.droplevel(ticker_level)
                else:
                    # Fallback: take the level containing known OHLCV names
                    for i, vals in enumerate(level_values):
                        lower_vals = [v.lower() for v in vals if isinstance(v, str)]
                        if 'close' in lower_vals and 'open' in lower_vals:
                            df.columns = df.columns.get_level_values(i)
                            break
                    else:
                        df.columns = df.columns.get_level_values(0)
                
            df['Ticker'] = ticker
            df = df.reset_index()
            
            # Standardize column naming
            rename_dict = {}
            for col in df.columns:
                if col.lower() == 'date':
                    rename_dict[col] = 'Date'
                elif col.lower() == 'open':
                    rename_dict[col] = 'Open'
                elif col.lower() == 'high':
                    rename_dict[col] = 'High'
                elif col.lower() == 'low':
                    rename_dict[col] = 'Low'
                elif col.lower() == 'close':
                    rename_dict[col] = 'Close'
                elif col.lower() == 'volume':
                    rename_dict[col] = 'Volume'
            df = df.rename(columns=rename_dict)
            
            # Ensure essential columns exist
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']
            if not all(col in df.columns for col in required_cols):
                print("FAILED (missing required columns)")
                failed_tickers.append(ticker)
                continue
                
            df = df[required_cols].copy()
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            
            # Save individual raw data
            df.to_csv(f"{raw_data_path}{ticker}_raw.csv", index=False)
            all_data.append(df)
            successful_tickers.append(ticker)
            print(f"SUCCESS ({len(df)} rows)")
            
        except Exception as e:
            print(f"FAILED (error: {str(e)})")
            failed_tickers.append(ticker)
            
    if not all_data:
        raise RuntimeError("No tickers were successfully downloaded!")
        
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv(f"{raw_data_path}combined_raw.csv", index=False)
    
    # Summary reporting
    min_date = combined_df['Date'].min().strftime('%Y-%m-%d')
    max_date = combined_df['Date'].max().strftime('%Y-%m-%d')
    
    print("\n==========================================")
    print("DATA LOADING SUMMARY")
    print("==========================================")
    print(f"Source: Yahoo Finance (yfinance)")
    print(f"Requested tickers: {len(tickers)}")
    print(f"Successful tickers ({len(successful_tickers)}): {successful_tickers}")
    print(f"Failed tickers ({len(failed_tickers)}): {failed_tickers}")
    print(f"Date Range: {min_date} to {max_date}")
    print(f"Total Raw Records: {len(combined_df):,}")
    print("\nRecords per ticker:")
    print(combined_df['Ticker'].value_counts())
    print("==========================================\n")
    
    return combined_df, successful_tickers, failed_tickers

if __name__ == '__main__':
    load_data()
