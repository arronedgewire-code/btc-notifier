# monitor.py
# Runs 24/7 on Railway — checks for new BTC trade signals every hour
# and posts to Discord via webhook when a new trade is detected.
#
# Required environment variables (set in Railway dashboard):
#   DISCORD_WEBHOOK_URL  — your Discord webhook URL
#   DISCORD_USER_ID      — your Discord user ID for @mentions (right-click profile -> Copy ID)

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
USER_ID = os.environ.get("DISCORD_USER_ID")       # e.g. "123456789012345678"
CHECK_INTERVAL = 3600                              # seconds between checks (1 hour)
STATE_FILE = "last_trade.json"                     # persists last seen trade across restarts

# -----------------------------------------------
# Helpers
# -----------------------------------------------
def mention():
    return f"<@{USER_ID}>" if USER_ID else ""

def load_last_trade():
    """Load the timestamp+type of the last trade we already notified about."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"time": None, "type": None}

def save_last_trade(trade):
    """Persist the latest trade we notified about."""
    with open(STATE_FILE, "w") as f:
        json.dump({"time": str(trade["Time"]), "type": trade["Type"]}, f)

def is_new_trade(trade, last):
    """True if this trade hasn't been notified yet."""
    if last["time"] is None:
        return True
    return str(trade["Time"]) != last["time"] or trade["Type"] != last["type"]

def format_discord_message(trade):
    """Build a Discord embed message for a trade."""
    t = trade["Type"]

    if t == "BUY (Long)":
        emoji = "🟢"
        color = 0x28a745
        title = "LONG ENTRY"
        fields = [
            {"name": "Price", "value": f"${trade['Price']:,.2f}", "inline": True},
            {"name": "Risk", "value": f"${trade['Risk ($)']:.2f}", "inline": True},
            {"name": "Notional", "value": trade["Notional ($)"], "inline": True},
        ]
    elif t == "SELL SHORT":
        emoji = "🔴"
        color = 0xdc3545
        title = "SHORT ENTRY"
        fields = [
            {"name": "Price", "value": f"${trade['Price']:,.2f}", "inline": True},
            {"name": "Risk", "value": f"${trade['Risk ($)']:.2f}", "inline": True},
            {"name": "Notional", "value": trade["Notional ($)"], "inline": True},
        ]
    elif t == "SELL (Long Exit)":
        emoji = "📤"
        pnl = trade.get("PnL ($)", 0)
        color = 0x28a745 if pnl >= 0 else 0xdc3545
        title = "LONG EXIT"
        fields = [
            {"name": "Price", "value": f"${trade['Price']:,.2f}", "inline": True},
            {"name": "PnL ($)", "value": f"${pnl:+.2f}", "inline": True},
            {"name": "PnL (%)", "value": trade.get("PnL (%)", "N/A"), "inline": True},
        ]
    elif t == "COVER (Short Exit)":
        emoji = "📤"
        pnl = trade.get("PnL ($)", 0)
        color = 0x28a745 if pnl >= 0 else 0xdc3545
        title = "SHORT EXIT"
        fields = [
            {"name": "Price", "value": f"${trade['Price']:,.2f}", "inline": True},
            {"name": "PnL ($)", "value": f"${pnl:+.2f}", "inline": True},
            {"name": "PnL (%)", "value": trade.get("PnL (%)", "N/A"), "inline": True},
        ]
    else:
        emoji = "📊"
        color = 0x888888
        title = t
        fields = []

    timestamp = str(trade.get("Time", ""))
    embed = {
        "title": f"{emoji} BTC/USD — {title}",
        "color": color,
        "fields": fields,
        "footer": {"text": f"Regime Bot • {timestamp}"},
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    payload = {
        "content": f"{mention()} New trade signal:",
        "embeds": [embed]
    }
    return payload

def post_to_discord(payload):
    """Send the payload to Discord webhook."""
    if not WEBHOOK_URL:
        print("[monitor] No DISCORD_WEBHOOK_URL set — skipping notification.")
        return
    try:
        resp = requests.post(WEBHOOK_URL, json=payload, timeout=10)
        if resp.status_code in (200, 204):
            print(f"[monitor] Discord notified successfully.")
        else:
            print(f"[monitor] Discord returned {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[monitor] Failed to post to Discord: {e}")

def post_error_to_discord(msg):
    """Send a plain error message to Discord."""
    if not WEBHOOK_URL:
        return
    try:
        requests.post(WEBHOOK_URL, json={"content": f"⚠️ {mention()} Regime bot error: {msg}"}, timeout=10)
    except Exception:
        pass

# -----------------------------------------------
# Main loop
# -----------------------------------------------
def run_check():
    print(f"[monitor] Running check at {datetime.now(timezone.utc).isoformat()}")

    try:
        df = fetch_btc_data()
        if df.empty:
            print("[monitor] Empty data — skipping.")
            return
    except Exception as e:
        print(f"[monitor] Data fetch failed: {e}")
        post_error_to_discord(f"Data fetch failed: {e}")
        return

    try:
        df = add_indicators(df)
        df, bull_state, bear_state = detect_regimes(df)
        df, trades = run_backtest(df)
    except Exception as e:
        print(f"[monitor] Strategy error: {e}")
        post_error_to_discord(f"Strategy error: {e}")
        return

    if not trades:
        print("[monitor] No trades found.")
        return

    latest_trade = trades[-1]
    last = load_last_trade()

    if is_new_trade(latest_trade, last):
        print(f"[monitor] New trade detected: {latest_trade['Type']} at {latest_trade['Time']}")
        payload = format_discord_message(latest_trade)
        post_to_discord(payload)
        save_last_trade(latest_trade)
    else:
        print(f"[monitor] No new trades since last check.")

    # Post regime distribution once daily at 1am UTC
    now = datetime.now(timezone.utc)
    if now.hour == 0 and 15 <= now.minute < 19:  # within the 00:15 UTC check window
        try:
            counts_str = df['regime'].value_counts().to_string()
        except Exception:
            counts_str = "N/A"
        message = (
            f"{mention()} 📊 Daily Regime Distribution Update\n"
            f"```text\n{counts_str}\n```"
        )
        post_to_discord({"content": message})
        print("[monitor] Daily regime distribution posted.")


if __name__ == "__main__":
    print("[monitor] Starting BTC regime monitor...")
    if not WEBHOOK_URL:
        print("[monitor] WARNING: DISCORD_WEBHOOK_URL not set. Notifications disabled.")

    post_to_discord({"content": f"{mention()} ✅ Regime bot is live and connected!"})
    print("[monitor] Startup ping sent to Discord.")

    # Run once immediately on startup
    run_check()

    # Then loop — wakes up at HH:01 every hour
    while True:
        now = datetime.now(timezone.utc)
        minutes_past = now.minute * 60 + now.second
        if now.minute < 1:
            wait = 60 - minutes_past
        else:
            wait = 3600 - minutes_past + 180 #+3mins to allow main model to propegate first.
        print(f"[monitor] Next check in {wait//60}m {wait%180}s (at next HH:03)")
        time.sleep(wait)
        run_check()
