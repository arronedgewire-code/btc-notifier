# hmm_model.py
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

def detect_regimes(df, n_states=3):
    """
    Detect market regimes using HMM fitted on DAILY resampled features,
    then map regime labels back to the hourly index.

    Fitting on daily data reduces noise from hourly fluctuations, producing
    more stable and meaningful regime boundaries.

    Args:
        df (pd.DataFrame): Hourly input dataframe with indicators
        n_states (int): Number of hidden regimes

    Returns:
        df (pd.DataFrame): Original hourly df with 'regime' column
        bull_state (int): Index of Bull regime
        bear_state (int): Index of Bear/Crash regime
    """
    df = df.copy()

    # -----------------------------------------------
    # Step 1: Resample hourly features to daily
    # -----------------------------------------------
    daily = pd.DataFrame()
    daily["Returns"] = df["Close"].resample("D").last().pct_change()
    daily["Range"] = (
        df["High"].resample("D").max() - df["Low"].resample("D").min()
    ) / df["Close"].resample("D").last()
    daily["volume_vol"] = (
        df["Volume"].resample("D").sum().pct_change().rolling(7).std()
    )
    daily = daily.dropna()

    if daily.empty or daily.nunique().min() <= 1:
        print("[detect_regimes] Daily features empty or constant. Assigning default regimes.")
        df["regime"] = "Neutral"
        return df, None, None

    # -----------------------------------------------
    # Step 2: Fit HMM on daily features
    # -----------------------------------------------
    scaler = StandardScaler()
    X = scaler.fit_transform(daily)

    try:
        model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000, random_state=42)
        model.fit(X)
        hidden_states = model.predict(X)
    except Exception as e:
        print(f"[detect_regimes] HMM failed: {e}")
        df["regime"] = "Neutral"
        return df, None, None

    # -----------------------------------------------
    # Step 3: Map states to regime labels by mean return
    # -----------------------------------------------
    state_means = [
        daily["Returns"].iloc[np.where(hidden_states == i)[0]].mean()
        for i in range(n_states)
    ]
    sorted_states = np.argsort(state_means)  # low -> high
    bull_state = int(sorted_states[-1])
    bear_state = int(sorted_states[0])

    # Bull = highest return state, Crash = lowest return state, Neutral = middle
    regime_map = {bull_state: "Bull", bear_state: "Crash"}
    daily["regime"] = pd.Series(hidden_states, index=daily.index).map(
        lambda x: regime_map.get(x, "Neutral")
    )

    # -----------------------------------------------
    # Step 4: Map daily regimes back to hourly index
    # Forward-fill so every hourly candle inherits its
    # day's regime label, including partial days
    # -----------------------------------------------
    df["regime"] = daily["regime"].reindex(df.index, method="ffill")

    # Fill any remaining NaNs at the start with the first known regime
    df["regime"] = df["regime"].ffill().bfill()

    print(f"[detect_regimes] Regime distribution:\n{df['regime'].value_counts()}")

    return df, bull_state, bear_state
