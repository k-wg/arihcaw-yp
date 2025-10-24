"""
Binance Range Bar Aggregator with WebSockets
============================
Fetches 1s klines via Binance WebSocket (for precise 1R aggregation), aggregates into configurable range bars,
saves to CSV with archiving, and displays a live dashboard with Rich.

Requirements:
pip install python-binance pandas pytz rich

Config:
- SYMBOL: Trading pair (default: 'SOLFDUSD')
- RANGE_SIZE: Price range per bar (default: 0.0 for 1R = tickSize; else USDT multiple)
- HIST_DEPTH_SECONDS: Historical fetch depth (default: 600 for 10min)

Run: python this_script.py
Stop: Ctrl+C
"""

import logging
import os
import signal
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from binance import ThreadedWebsocketManager
from binance.client import Client
from binance.exceptions import BinanceAPIException
from pytz import timezone
from rich import box
from rich.console import Console
from rich.progress import Progress, SpinnerColumn
from rich.text import Text

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Config
SYMBOL = "SOLFDUSD"  # Change for other pairs
RANGE_SIZE = 0.0  # 0.0 means auto-set to tickSize for true 1R (TradingView style)
HIST_DEPTH_SECONDS = 600  # 10min in seconds for 1s klines

# Globals
console = Console()
client = Client()  # Public client
kenya_tz = timezone("Africa/Nairobi")
csv_file = "historic_df_omega.csv"
df = pd.DataFrame()  # In-memory storage
running = True
tick_size = 0.0
range_size = 0.0
tws = None  # WebSocket manager

# WebSocket data storage
klines_buffer = []
last_kline_time = 0


def signal_handler(sig, frame):
    global running
    running = False
    console.print("\n🛑 Stopping gracefully...", style="bold yellow")
    if tws:
        tws.stop()
    save_csv()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def archive_csv():
    """Archive existing CSV with UUID if present."""
    if os.path.exists(csv_file):
        short_uuid = uuid.uuid4().hex[:8]
        archive_name = (
            f"{csv_file.rsplit('.', 1)[0]}_{short_uuid}.{csv_file.rsplit('.', 1)[1]}"
        )
        os.rename(csv_file, archive_name)
        console.print(f"🗄️ Archived previous CSV to {archive_name}", style="green")


def format_timestamp(ts_ms: int) -> str:
    """Convert UTC ms to Kenya TZ string matching sample format."""
    dt_utc = datetime.fromtimestamp(ts_ms / 1000, tz=timezone("UTC"))
    dt_kenya = dt_utc.astimezone(kenya_tz)
    return dt_kenya.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Match sample precision


def get_symbol_ticksize(symbol: str) -> float:
    """Fetch and parse tickSize from symbol info, auto-set RANGE_SIZE if 0."""
    global RANGE_SIZE, tick_size, range_size
    try:
        info = client.get_symbol_info(symbol)
        if not info:
            raise ValueError(f"Symbol {symbol} not found")
        tick_filter = next(
            f for f in info["filters"] if f["filterType"] == "PRICE_FILTER"
        )
        tick_size = float(tick_filter["tickSize"])
        if RANGE_SIZE == 0.0:
            range_size = tick_size  # Auto 1R
            console.print(
                f"🔧 Auto-set 1R range_size to tickSize: {range_size}",
                style="italic blue",
            )
        else:
            range_size = RANGE_SIZE
            if range_size % tick_size != 0:
                logger.warning(
                    f"⚠️ range_size {range_size} not multiple of tickSize {tick_size}; rounding up"
                )
                range_size = ((range_size // tick_size) + 1) * tick_size
        return tick_size
    except Exception as e:
        console.print(f"❌ Error fetching symbol info: {e}", style="red")
        sys.exit(1)


def kline_message_handler(msg):
    """Handle incoming WebSocket 1s kline messages."""
    global klines_buffer, last_kline_time

    if "e" in msg and msg["e"] == "kline" and msg["k"]["x"]:  # kline closed
        k = msg["k"]
        kline_data = {
            "Open Time": format_timestamp(k["t"]),
            "Open": float(k["o"]),
            "High": float(k["h"]),
            "Low": float(k["l"]),
            "Close": float(k["c"]),
            "Volume": float(k["v"]),
            "Close Time": format_timestamp(k["T"]),
            "Quote Asset Volume": float(k["q"]),
            "Number of Trades": k["n"],
            "Taker Buy Base Asset Volume": float(k["V"]),
            "Taker Buy Quote Asset Volume": float(k["Q"]),
            "Ignore": 0,
        }
        klines_buffer.append(kline_data)
        last_kline_time = time.time()


def aggregate_range_bars(klines_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate 1s klines into range bars using accumulated High/Low for triggers."""
    global tick_size, range_size
    if not klines_list:
        return []

    bars = []
    current_bar = None

    for kline in klines_list:
        if current_bar is None:
            # Start new bar with this kline
            current_bar = kline.copy()
            continue

        # Update aggregates and extremes
        current_bar["High"] = max(current_bar["High"], kline["High"])
        current_bar["Low"] = min(current_bar["Low"], kline["Low"])
        agg_cols = [
            "Volume",
            "Quote Asset Volume",
            "Number of Trades",
            "Taker Buy Base Asset Volume",
            "Taker Buy Quote Asset Volume",
        ]
        for agg_col in agg_cols:
            current_bar[agg_col] += kline[agg_col]

        # Check trigger: if accumulated High or Low exceeds range from open
        open_price = float(current_bar["Open"])
        up_trigger = current_bar["High"] >= open_price + range_size
        down_trigger = current_bar["Low"] <= open_price - range_size

        if up_trigger or down_trigger:
            # Close current bar at the extreme that triggered
            if up_trigger:
                current_bar["Close"] = current_bar["High"]
            else:
                current_bar["Close"] = current_bar["Low"]
            current_bar["Close Time"] = kline["Close Time"]
            bars.append(current_bar)

            # Start new bar at the trigger price (reversal)
            new_open = current_bar["High"] if up_trigger else current_bar["Low"]
            current_bar = kline.copy()
            current_bar["Open Time"] = kline["Close Time"]
            current_bar["Open"] = new_open
            current_bar["High"] = max(new_open, kline["High"])
            current_bar["Low"] = min(new_open, kline["Low"])

    # Append final bar if it has meaningful data
    if current_bar and len(klines_list) > 0:
        # Only append if current_bar is different from the last processed kline
        last_kline = klines_list[-1]
        if (
            current_bar["Open"] != last_kline["Open"]
            or current_bar["Close"] != last_kline["Close"]
            or current_bar["High"] != last_kline["High"]
            or current_bar["Low"] != last_kline["Low"]
        ):
            bars.append(current_bar)

    return bars


def print_status():
    """Print the status line."""
    if df.empty:
        return
    now_hhmmss = datetime.now(kenya_tz).strftime("%H:%M:%S")
    last_close_time_str = df["Close Time"].iloc[-1]
    last_hhmmss = last_close_time_str.split()[1][:8]
    last_price = float(df["Close"].iloc[-1])
    last_trades = int(df["Number of Trades"].iloc[-1])
    curr_range_pct = 0.0
    rows = len(df)

    # Check WebSocket health
    ws_health = (
        "✅ OK"
        if last_kline_time and (time.time() - last_kline_time) < 10
        else "⚠️ STALE"
    )
    ws_state = (
        "CONNECTED"
        if last_kline_time and (time.time() - last_kline_time) < 10
        else "RECONNECTING"
    )

    console.print(
        f"[{now_hhmmss}] 📊 Status: {SYMBOL} @ {last_price:.2f}, Range: {curr_range_pct:.1f}%, Trades: {last_trades}, Health: {ws_health}, Rows: {rows}, Last trade: {last_hhmmss}, State: {ws_state}",
        style="cyan",
    )


def load_historical() -> None:
    """Bootstrap with historical data using REST API."""
    global df
    now_utc = int(datetime.now(timezone("UTC")).timestamp() * 1000)
    start_time = now_utc - HIST_DEPTH_SECONDS * 1000

    with Progress(SpinnerColumn(), console=console) as progress:
        task = progress.add_task("📥 Fetching historical 1s data...", total=None)
        try:
            klines = client.get_klines(
                symbol=SYMBOL,
                interval=Client.KLINE_INTERVAL_1SECOND,
                startTime=start_time,
                endTime=now_utc,
                limit=1000,
            )

            if klines:
                df_klines = pd.DataFrame(
                    klines,
                    columns=[
                        "Open Time",
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume",
                        "Close Time",
                        "Quote Asset Volume",
                        "Number of Trades",
                        "Taker Buy Base Asset Volume",
                        "Taker Buy Quote Asset Volume",
                        "Ignore",
                    ],
                )
                # Convert types
                for col in [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    "Quote Asset Volume",
                    "Number of Trades",
                    "Taker Buy Base Asset Volume",
                    "Taker Buy Quote Asset Volume",
                ]:
                    df_klines[col] = pd.to_numeric(df_klines[col])
                df_klines["Ignore"] = 0
                # Format timestamps
                df_klines["Open Time"] = [
                    format_timestamp(int(ts)) for ts in df_klines["Open Time"]
                ]
                df_klines["Close Time"] = [
                    format_timestamp(int(ts)) for ts in df_klines["Close Time"]
                ]

                # Convert to list of dicts for aggregation
                klines_list = df_klines.to_dict("records")
                bars = aggregate_range_bars(klines_list)
                if bars:
                    df = pd.DataFrame(bars)
                    save_csv()
                    console.print(
                        f"✅ Loaded {len(bars)} 1R range bars from history (range: {range_size}) 📊",
                        style="green",
                    )
            else:
                console.print("🚀 Fresh start! No historical data.", style="italic cyan")

        except Exception as e:
            console.print(f"❌ Historical data error: {e}", style="red")
        finally:
            progress.remove_task(task)


def process_new_klines() -> bool:
    """Process new 1s klines from WebSocket into range bars."""
    global df, klines_buffer

    if not klines_buffer:
        return False

    # Process all buffered klines
    bars = aggregate_range_bars(klines_buffer)
    updated = False

    if bars:
        for bar in bars:
            close_time_str = bar["Close Time"]
            hhmmss = close_time_str.split()[1][:8]
            price = float(bar["Close"])
            range_val = abs(price - float(bar["Open"]))
            trades = int(bar["Number of Trades"])
            console.print(
                f"[{hhmmss}] 📊 CSV updated - Range bar closed - Price: {price:.6f} - Trades: {trades}",
                style="bold green",
            )

        new_df = pd.DataFrame(bars)
        df = pd.concat([df, new_df], ignore_index=True)
        updated = True
        save_csv()

    # Clear processed klines
    klines_buffer.clear()
    return updated


def save_csv():
    """Save DF to CSV atomically."""
    if df.empty:
        return
    temp_file = csv_file + ".tmp"
    df.to_csv(temp_file, index=False)
    os.replace(temp_file, csv_file)


def start_websocket():
    """Start WebSocket connection for 1s klines."""
    global tws
    tws = ThreadedWebsocketManager()
    tws.start()

    # Start kline socket for 1-second intervals
    tws.start_kline_socket(symbol=SYMBOL, interval="1s", callback=kline_message_handler)

    console.print(f"🔌 WebSocket started for {SYMBOL}@kline_1s", style="bold green")
    return tws


def main():
    """Main loop."""
    global running, tws

    archive_csv()
    get_symbol_ticksize(SYMBOL)  # Validate early

    # Load historical data for bootstrap
    load_historical()

    # Start WebSocket
    console.print("🔌 Starting WebSocket connection...", style="bold blue")
    start_websocket()

    if not df.empty:
        print_status()

    try:
        # Main processing loop
        while running:
            updated = process_new_klines()
            if updated or not df.empty:
                print_status()

            # Check WebSocket health and reconnect if needed
            if last_kline_time and (time.time() - last_kline_time) > 30:
                console.print("⚠️ WebSocket stale, reconnecting...", style="yellow")
                tws.stop()
                time.sleep(2)
                start_websocket()

            time.sleep(1)  # Check for new data every second

    except Exception as e:
        console.print(f"🛑 Connection error: {e}. Retrying in 5s...", style="bold red")
        time.sleep(5)
        if tws:
            tws.stop()
        running = True
        main()  # Restart

    console.print("👋 Session ended. Check historic_df_omega.csv!", style="bold blue")


if __name__ == "__main__":
    main()
