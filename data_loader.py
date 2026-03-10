# data_loader.py
import requests
import pandas as pd
import time
from datetime import datetime, timezone

KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"
PAIR = "XBTUSD"       # Kraken's BTC/USD pair name
INTERVAL = 60         # 60 minutes = 1h candles
YEARS = 5             # how many years of history to fetch
CANDLES_PER_REQUEST = 720  # Kraken max per call

def fetch_btc_data(retries=3, pause=5):
    """
    Fetch BTC/USD hourly OHLCV data from Kraken public API.
    Paginates backwards to collect up to YEARS years of 1h candles.
    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    and a timezone-aware UTC DatetimeIndex — compatible with the rest of the pipeline.
    """
    now = int(datetime.now(timezone.utc).timestamp())
    since = now - (YEARS * 365 * 24 * 3600)

    all_candles = []
    current_since = since
    print(f"[data_loader] Fetching {YEARS}y of BTC/USD 1h candles from Kraken...")

    while True:
        params = {
            "pair": PAIR,
            "interval": INTERVAL,
            "since": current_since
        }

        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(KRAKEN_OHLC_URL, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()

                if data.get("error"):
                    raise ValueError(f"Kraken API error: {data['error']}")

                # Kraken returns pair under internal name e.g. "XXBTZUSD" not "XBTUSD"
                result_key = [k for k in data["result"].keys() if k != "last"][0]
                candles = data["result"][result_key]
                last = int(data["result"]["last"])
                break  # success

            except Exception as e:
                print(f"[data_loader] Attempt {attempt}/{retries} failed: {e}")
                if attempt < retries:
                    time.sleep(pause)
                else:
                    print("[data_loader] Failed to fetch data after retries.")
                    return pd.DataFrame()

        if not candles:
            break

        all_candles.extend(candles)
        print(f"[data_loader] Fetched {len(all_candles)} candles so far...")

        # Kraken returns the last timestamp as next since
        if last <= current_since:
            break
        current_since = last

        # Stop if we've gone past now
        if current_since >= now:
            break

        time.sleep(0.5)  # be polite to the API

    if not all_candles:
        print("[data_loader] No candles received.")
        return pd.DataFrame()

    # Kraken OHLC format: [time, open, high, low, close, vwap, volume, count]
    df = pd.DataFrame(all_candles, columns=[
        "Time", "Open", "High", "Low", "Close", "VWAP", "Volume", "Count"
    ])

    # Convert and clean
    df["Time"] = pd.to_datetime(df["Time"], unit="s", utc=True)
    df = df.set_index("Time")
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    # Drop the last (incomplete) candle
    df = df.iloc[:-1]

    print(f"[data_loader] Done. {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    return df
