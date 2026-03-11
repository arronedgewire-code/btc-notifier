# monitor.py
# Runs 24/7 on Railway — checks for new BTC trade signals every hour
# Posts to Discord on: new trade, regime change, daily summary
#
# Required environment variables (set in Railway dashboard):
#   DISCORD_WEBHOOK_URL  — your Discord webhook URL
#   DISCORD_USER_ID      — your Discord user ID for @mentions

import os
import time
import json
import requests
import pandas as pd
from datetime import datetime, timezone

from data_loader import fetch_btc_data
from indicators import add_indicators
from hmm_model import detect_regimes
from backtester import run_backtest

# -----------------------------------------------
# Config
# -----------------------------------------------
WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
USER_ID     = os.environ.get("DISCORD_USER_ID")
STATE_FILE  = "last_trade.json"

# -----------------------------------------------
# Helpers
# -----------------------------------------------
def mention():
    return f"<@{USER_ID}>" if USER_ID else ""

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "last_trade_time": None,
        "last_trade_type": None,
        "last_regime": None,
        "last_summary_date": None,
        "entry_price": None,
        "entry_risk": None,
        "position_side": None,
    }

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, default=str)

def post_to_discord(payload):
    if not WEBHOOK_URL:
        print("[monitor] No DISCORD_WEBHOOK_URL — skipping.")
        return
    try:
        resp = requests.post(WEBHOOK_URL, json=payload, timeout=10)
        if resp.status_code in (200, 204):
            print("[monitor] Discord notified.")
        else:
            print(f"[monitor] Discord {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[monitor] Post failed: {e}")

def post_error(msg):
    if not WEBHOOK_URL:
        return
    try:
        requests.post(WEBHOOK_URL, json={"content": f"⚠️ {mention()} {msg}"}, timeout=10)
    except Exception:
        pass

def regime_emoji(regime):
    return {"Bull": "🟢", "Crash": "🔴", "Neutral": "⬜", "Bear": "🟠"}.get(regime, "❔")

# -----------------------------------------------
# Discord message builders
# -----------------------------------------------
def trade_embed(trade, current_capital=None):
    t = trade["Type"]

    if t == "BUY (Long)":
        color, title, emoji = 0x28a745, "LONG ENTRY", "🟢"
        fields = [
            {"name": "Entry Price", "value": f"${trade['Price']:,.2f}", "inline": True},
            {"name": "Risk",        "value": f"${trade['Risk ($)']:.2f}", "inline": True},
            {"name": "Notional",   "value": trade["Notional ($)"], "inline": True},
        ]
    elif t == "SELL SHORT":
        color, title, emoji = 0xdc3545, "SHORT ENTRY", "🔴"
        fields = [
            {"name": "Entry Price", "value": f"${trade['Price']:,.2f}", "inline": True},
            {"name": "Risk",        "value": f"${trade['Risk ($)']:.2f}", "inline": True},
            {"name": "Notional",   "value": trade["Notional ($)"], "inline": True},
        ]
    elif t == "SELL (Long Exit)":
        pnl   = trade.get("PnL ($)", 0)
        color  = 0x28a745 if pnl >= 0 else 0xdc3545
        title, emoji = "LONG EXIT", "📤"
        outcome = "✅ WIN" if pnl >= 0 else "❌ LOSS"
        fields = [
            {"name": "Exit Price", "value": f"${trade['Price']:,.2f}", "inline": True},
            {"name": "PnL ($)",    "value": f"${pnl:+.2f}", "inline": True},
            {"name": "PnL (%)",    "value": trade.get("PnL (%)", "N/A"), "inline": True},
            {"name": "Result",     "value": outcome, "inline": True},
        ]
        if current_capital:
            fields.append({"name": "Account Balance", "value": f"${current_capital:,.2f}", "inline": True})
    elif t == "COVER (Short Exit)":
        pnl   = trade.get("PnL ($)", 0)
        color  = 0x28a745 if pnl >= 0 else 0xdc3545
        title, emoji = "SHORT EXIT", "📤"
        outcome = "✅ WIN" if pnl >= 0 else "❌ LOSS"
        fields = [
            {"name": "Exit Price", "value": f"${trade['Price']:,.2f}", "inline": True},
            {"name": "PnL ($)",    "value": f"${pnl:+.2f}", "inline": True},
            {"name": "PnL (%)",    "value": trade.get("PnL (%)", "N/A"), "inline": True},
            {"name": "Result",     "value": outcome, "inline": True},
        ]
        if current_capital:
            fields.append({"name": "Account Balance", "value": f"${current_capital:,.2f}", "inline": True})
    else:
        color, title, emoji, fields = 0x888888, t, "📊", []

    embed = {
        "title":  f"{emoji} BTC/USD — {title}",
        "color":  color,
        "fields": fields,
        "footer": {"text": f"Regime Bot • {trade.get('Time', '')}"},
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    return {"content": f"{mention()} New trade signal:", "embeds": [embed]}


def regime_change_embed(old_regime, new_regime, price, position_side):
    emoji_old = regime_emoji(old_regime)
    emoji_new = regime_emoji(new_regime)
    color_map = {"Bull": 0x28a745, "Crash": 0xdc3545, "Neutral": 0x888888, "Bear": 0xfd7e14}
    color = color_map.get(new_regime, 0x888888)

    pos_note = "No open position"
    if position_side == "long":
        pos_note = "⚠️ Holding Long — monitoring for exit"
    elif position_side == "short":
        pos_note = "⚠️ Holding Short — monitoring for exit"

    embed = {
        "title": f"🔄 Regime Change: {old_regime} → {new_regime}",
        "color": color,
        "fields": [
            {"name": "Previous", "value": f"{emoji_old} {old_regime}", "inline": True},
            {"name": "New",      "value": f"{emoji_new} {new_regime}", "inline": True},
            {"name": "BTC Price","value": f"${price:,.2f}",            "inline": True},
            {"name": "Position", "value": pos_note,                    "inline": False},
        ],
        "footer": {"text": "Regime Bot"},
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    return {"content": f"{mention()} Regime change detected!", "embeds": [embed]}


def daily_summary_embed(trades, df, position_side, entry_price, current_price):
    trades_df = pd.DataFrame(trades)
    total_pnl = 0.0
    win_rate  = 0.0
    total_trades = 0

    if not trades_df.empty and "PnL ($)" in trades_df.columns:
        closed = trades_df[trades_df["PnL ($)"].notna()]
        total_pnl    = closed["PnL ($)"].sum()
        total_trades = len(closed)
        win_rate     = (closed["PnL ($)"] > 0).sum() / total_trades * 100 if total_trades > 0 else 0

    regime_now = df["regime"].iloc[-1] if "regime" in df.columns else "N/A"

    # Unrealised P&L on open position
    unrealised = ""
    if position_side == "long" and entry_price:
        pnl_est = (current_price - entry_price) / entry_price * 100
        unrealised = f"{'📈' if pnl_est >= 0 else '📉'} {pnl_est:+.2f}% unrealised"
    elif position_side == "short" and entry_price:
        pnl_est = (entry_price - current_price) / entry_price * 100
        unrealised = f"{'📈' if pnl_est >= 0 else '📉'} {pnl_est:+.2f}% unrealised"

    pos_str = position_side.upper() if position_side else "CASH"

    fields = [
        {"name": "Position",       "value": pos_str,                         "inline": True},
        {"name": "Regime",         "value": f"{regime_emoji(regime_now)} {regime_now}", "inline": True},
        {"name": "BTC Price",      "value": f"${current_price:,.2f}",        "inline": True},
        {"name": "Total PnL",      "value": f"${total_pnl:+.2f}",            "inline": True},
        {"name": "Trades (closed)","value": str(total_trades),               "inline": True},
        {"name": "Win Rate",       "value": f"{win_rate:.1f}%",              "inline": True},
    ]
    if unrealised:
        fields.append({"name": "Unrealised", "value": unrealised, "inline": False})

    embed = {
        "title":  "📊 Daily Strategy Summary",
        "color":  0x5865F2,
        "fields": fields,
        "footer": {"text": "Regime Bot — Daily Digest"},
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    return {"content": f"{mention()} Daily summary:", "embeds": [embed]}


# -----------------------------------------------
# Main check
# -----------------------------------------------
def run_check(state):
    print(f"[monitor] Check at {datetime.now(timezone.utc).isoformat()}")
    now = datetime.now(timezone.utc)

    try:
        df = fetch_btc_data()
        if df.empty:
            print("[monitor] Empty data — skipping.")
            return state
    except Exception as e:
        print(f"[monitor] Fetch failed: {e}")
        post_error(f"Data fetch failed: {e}")
        return state

    try:
        df = add_indicators(df)
        df, _, _ = detect_regimes(df)
        df, trades = run_backtest(df)
    except Exception as e:
        print(f"[monitor] Strategy error: {e}")
        post_error(f"Strategy error: {e}")
        return state

    current_price = float(df["Close"].iloc[-1])
    current_regime = str(df["regime"].iloc[-1]) if "regime" in df.columns else "N/A"

    # --- Track current open position from trade log ---
    position_side = None
    entry_price   = None
    for tr in trades:
        if tr["Type"] in ["BUY (Long)", "SELL SHORT"]:
            position_side = "long" if tr["Type"] == "BUY (Long)" else "short"
            entry_price   = tr.get("Price")
        elif tr["Type"] in ["SELL (Long Exit)", "COVER (Short Exit)"]:
            position_side = None
            entry_price   = None

    # --- 1. New trade notification ---
    latest_trade = trades[-1] if trades else None
    if latest_trade:
        last_key = f"{latest_trade['Time']}_{latest_trade['Type']}"
        prev_key = f"{state.get('last_trade_time')}_{state.get('last_trade_type')}"
        if last_key != prev_key:
            # Estimate current capital from equity curve
            current_capital = None
            if "Equity" in df.columns:
                current_capital = float(df["Equity"].iloc[-1])
            payload = trade_embed(latest_trade, current_capital)
            post_to_discord(payload)
            state["last_trade_time"] = str(latest_trade["Time"])
            state["last_trade_type"] = latest_trade["Type"]
            print(f"[monitor] New trade: {latest_trade['Type']}")

    # --- 2. Regime change notification ---
    last_regime = state.get("last_regime")
    if last_regime and current_regime != last_regime:
        payload = regime_change_embed(last_regime, current_regime, current_price, position_side)
        post_to_discord(payload)
        print(f"[monitor] Regime change: {last_regime} → {current_regime}")
    state["last_regime"] = current_regime

    # --- 3. Daily summary at 08:00 UTC ---
    today_str = now.strftime("%Y-%m-%d")
    if now.hour == 8 and state.get("last_summary_date") != today_str:
        payload = daily_summary_embed(trades, df, position_side, entry_price, current_price)
        post_to_discord(payload)
        state["last_summary_date"] = today_str
        print("[monitor] Daily summary sent.")

    # Update position state
    state["position_side"] = position_side
    state["entry_price"]   = entry_price

    print(f"[monitor] Regime: {current_regime} | Position: {position_side or 'CASH'} | BTC: ${current_price:,.2f}")
    return state


# -----------------------------------------------
# Entry point
# -----------------------------------------------
if __name__ == "__main__":
    print("[monitor] Starting BTC regime monitor...")
    if not WEBHOOK_URL:
        print("[monitor] WARNING: DISCORD_WEBHOOK_URL not set.")

    post_to_discord({"content": f"{mention()} ✅ Regime bot is live and connected!"})

    state = load_state()
    state = run_check(state)
    save_state(state)

    while True:
        now = datetime.now(timezone.utc)
        minutes_past = now.minute * 60 + now.second
        if now.minute < 3:
            wait = (3 * 60) - minutes_past
        else:
            wait = 3600 - minutes_past + (3 * 60)
        print(f"[monitor] Next check in {wait//60}m {wait%60}s")
        time.sleep(wait)
        state = run_check(state)
        save_state(state)
