# data_loader.py
import yfinance as yf
import pandas as pd
import time

def fetch_btc_data(retries=5, pause=5):
    """
    Fetch BTC-USD historical data from Yahoo Finance.
    Retries safely in case of network issues or rate limits.
    """
    for attempt in range(1, retries + 1):
        try:
            df = yf.download("BTC-USD", period="2y", interval="1h", progress=False)
            if df.empty:
                raise ValueError("Empty data received from Yahoo Finance.")
            # Flatten MultiIndex columns returned by newer yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            return df
        except Exception as e:
            print(f"[data_loader] Attempt {attempt}/{retries} failed: {e}")
            if "Too Many Requests" in str(e):
                print(f"[data_loader] Rate limited. Waiting {pause} seconds before retrying...")
            time.sleep(pause)
    # After retries exhausted
    print("[data_loader] Failed to fetch BTC data after multiple attempts.")
    return pd.DataFrame()  # return empty DF instead of None
