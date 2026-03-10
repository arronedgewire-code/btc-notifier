# backtester.py
import pandas as pd
import numpy as np

# -----------------------------
# Safe float conversion
# -----------------------------
def safe_float(val):
    """
    Convert a scalar or single-element Series to float.
    """
    if isinstance(val, pd.Series):
        return float(val.iloc[0]) if not val.empty else 0.0
    return float(val)

# -----------------------------
# Bullish Confirmation score (Voting System)
# -----------------------------
def confirmation_score(row):
    """
    Compute number of bullish confirmations for long entry.
    Returns integer between 0-8.
    """
    score = 0
    try:
        rsi = safe_float(row.get("RSI", 0))
        momentum = safe_float(row.get("Momentum", 0))
        vol = safe_float(row.get("Volatility", 0))
        volume = safe_float(row.get("Volume", 0))
        volume_sma = safe_float(row.get("Volume_SMA", 0))
        adx = safe_float(row.get("ADX", 0))
        close = safe_float(row.get("Close", 0))
        ema50 = safe_float(row.get("EMA50", 0))
        ema100 = safe_float(row.get("EMA100", 0))
        ema200 = safe_float(row.get("EMA200", 0))
        macd = safe_float(row.get("MACD", 0))
        signal = safe_float(row.get("Signal", 0))

        conditions = [
            rsi < 90,
            momentum > 0.01,
            vol < 0.06,
            volume > volume_sma,
            adx > 25,
            close > ema50,
            close > ema100,
            close > ema200,
            macd > signal
        ]

        score = sum(conditions)
    except Exception as e:
        print(f"[backtester] confirmation_score error: {e}")
    return score

# -----------------------------
# Bearish Confirmation score (Voting System)
# -----------------------------
def bearish_confirmation_score(row):
    """
    Compute number of bearish confirmations for short entry.
    Returns integer between 0-8.
    """
    score = 0
    try:
        rsi = safe_float(row.get("RSI", 0))
        momentum = safe_float(row.get("Momentum", 0))
        vol = safe_float(row.get("Volatility", 0))
        volume = safe_float(row.get("Volume", 0))
        volume_sma = safe_float(row.get("Volume_SMA", 0))
        adx = safe_float(row.get("ADX", 0))
        close = safe_float(row.get("Close", 0))
        ema50 = safe_float(row.get("EMA50", 0))
        ema100 = safe_float(row.get("EMA100", 0))
        ema200 = safe_float(row.get("EMA200", 0))
        macd = safe_float(row.get("MACD", 0))
        signal = safe_float(row.get("Signal", 0))

        conditions = [
            (rsi > 70 or rsi < 60),  # overbought exhaustion OR bears in control
            momentum < -0.01,    # negative momentum
            vol > 0.03,          # elevated volatility (panic selling)
            volume > volume_sma, # volume surge on down move
            adx > 25,            # strong trend conviction
            close < ema50,       # price broken below ema50
            close < ema100,      # price broken below ema200
            close < ema200,      # price broken below ema200
            macd < signal        # bearish MACD crossover
        ]

        score = sum(conditions)
    except Exception as e:
        print(f"[backtester] bearish_confirmation_score error: {e}")
    return score

# -----------------------------
# Backtesting engine
# -----------------------------
def run_backtest(df, starting_capital=1000, leverage=25, min_confirmations=6, short_min_confirmations=6, cooldown_hours=12):
    """
    Run regime-based backtest with long and short positions.
    # Initial capital: $1000 | Leverage: 25x | Risk per trade: 2% of current capital (dynamic)
    - Long: entered on Bull regime with >= min_confirmations, exits on Crash
    - Short: entered on Crash regime with >= short_min_confirmations, exits only on Bull
    - Neutral regime: no new positions
    - Cooldown only applies to long re-entries, not short -> long transitions
    """
    df = df.copy()
    capital = starting_capital
    risk_per_trade = capital * 0.02  # initialised here, updated dynamically on each entry
    position = 0          # size of position (always positive)
    position_side = None  # "long" or "short"
    entry_price = 0
    cooldown_until = None
    equity_curve = []
    trades = []

    for i in range(len(df)):
        row = df.iloc[i]
        time = df.index[i]

        # Check regime safely
        regime = row.get("regime", "Neutral")
        if isinstance(regime, pd.Series):
            regime = regime.iloc[0] if not regime.empty else "Neutral"

        close_price = safe_float(row.get("Close", 0))

        # --- Exit logic (checked before entry) ---

        # Close long on Bear or Crash
        if position_side == "long" and regime in ["Bear", "Crash"]:
            exit_price = close_price
            pnl = (exit_price - entry_price) * position
            capital += pnl
            pnl_pct = (pnl / risk_per_trade) * 100 if risk_per_trade != 0 else 0.0  # % return on capital risked
            trades.append({"Time": time, "Type": "SELL (Long Exit)", "Price": round(exit_price, 2), "PnL ($)": round(pnl, 2), "PnL (%)": f"{pnl_pct:+.2f}%"})
            position = 0
            position_side = None
            cooldown_until = time + pd.Timedelta(hours=cooldown_hours)

        # Close short only on Bull signal
        elif position_side == "short" and regime == "Bull":
            exit_price = close_price
            pnl = (entry_price - exit_price) * position  # profit when price falls
            capital += pnl
            pnl_pct = (pnl / risk_per_trade) * 100 if risk_per_trade != 0 else 0.0  # % return on capital risked
            trades.append({"Time": time, "Type": "COVER (Short Exit)", "Price": round(exit_price, 2), "PnL ($)": round(pnl, 2), "PnL (%)": f"{pnl_pct:+.2f}%"})
            position = 0
            position_side = None
            # No cooldown on short -> long transition so Bull entry can fire immediately

        # --- Entry logic ---

        if position_side is None:
            # Long entry: Bull regime, bullish confirmations, no cooldown active
            in_cooldown = cooldown_until is not None and time < cooldown_until
            if regime == "Bull" and not in_cooldown:
                score = confirmation_score(row)
                if score >= min_confirmations:
                    risk_per_trade = capital * 0.02  # 2% of current capital
                    position = (risk_per_trade * leverage) / close_price
                    entry_price = close_price
                    position_side = "long"
                    notional = risk_per_trade * leverage
                    trades.append({"Time": time, "Type": "BUY (Long)", "Price": round(entry_price, 2), "Risk ($)": round(risk_per_trade, 2), "Notional ($)": f"x{leverage} = ${notional:.2f}"})

            # Short entry: Crash regime, bearish confirmations
            elif regime == "Crash":
                score = bearish_confirmation_score(row)
                if score >= short_min_confirmations:
                    risk_per_trade = capital * 0.02  # 2% of current capital
                    position = (risk_per_trade * leverage) / close_price
                    entry_price = close_price
                    position_side = "short"
                    notional = risk_per_trade * leverage
                    trades.append({"Time": time, "Type": "SELL SHORT", "Price": round(entry_price, 2), "Risk ($)": round(risk_per_trade, 2), "Notional ($)": f"x{leverage} = ${notional:.2f}"})

        # --- Equity curve update ---
        if position_side == "long":
            equity_curve.append(capital + (close_price - entry_price) * position)
        elif position_side == "short":
            equity_curve.append(capital + (entry_price - close_price) * position)
        else:
            equity_curve.append(capital)

    df["Equity"] = equity_curve
    return df, trades
