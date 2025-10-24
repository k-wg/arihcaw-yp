"""
Binance Range Bar Aggregator
============================
Fetches 1s klines from Binance REST API (for precise 1R aggregation), aggregates into configurable range bars,
saves to CSV with archiving, and displays a live dashboard with Rich.

Requirements:
pip install python-binance pandas pytz rich

Config:
- SYMBOL: Trading pair (default: 'SOLFDUSD')
- RANGE_SIZE: Price range per bar (default: 0.0 for 1R = tickSize; else USDT multiple)
- HIST_DEPTH_SECONDS: Historical fetch depth (default: 600 for 10min)
- POLL_INTERVAL_SEC: Polling frequency (default: 5 for finer real-time)

Run: python this_script.py
Stop: Ctrl+C
"""

import logging
import os
import signal
import sys
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
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
POLL_INTERVAL_SEC = 5  # Faster poll for 1s data

# Globals
console = Console()
client = Client()  # Public client
kenya_tz = timezone("Africa/Nairobi")
csv_file = "historic_df_alpha.csv"
df = pd.DataFrame()  # In-memory storage
last_processed_time: Optional[int] = None  # UTC ms
running = True
tick_size = 0.0
range_size = 0.0


def signal_handler(sig, frame):
    global running
    running = False
    console.print("\n🛑 Stopping gracefully...", style="bold yellow")
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


def fetch_klines(start_time: int, end_time: Optional[int] = None) -> pd.DataFrame:
    """Fetch 1s klines via REST, convert to DF with columns matching sample."""
    try:
        klines = client.get_klines(
            symbol=SYMBOL,
            interval=Client.KLINE_INTERVAL_1SECOND,
            startTime=start_time,
            endTime=end_time,
            limit=1000,  # Covers ~16min
        )
        if not klines:
            return pd.DataFrame()

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
        # Format timestamps to strings
        df_klines["Open Time"] = [
            format_timestamp(int(ts)) for ts in df_klines["Open Time"]
        ]
        df_klines["Close Time"] = [
            format_timestamp(int(ts)) for ts in df_klines["Close Time"]
        ]
        return df_klines
    except BinanceAPIException as e:
        if e.code == -1003:  # Rate limit
            time.sleep(5)
            return fetch_klines(start_time, end_time)
        console.print(f"❌ API Error: {e}", style="red")
        raise
    except Exception as e:
        console.print(f"❌ Fetch error: {e}", style="red")
        raise


def aggregate_range_bars(klines_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Aggregate 1s klines into range bars using High/Low for triggers."""
    global tick_size, range_size
    if klines_df.empty:
        return []

    bars = []
    current_bar = None
    for _, row in klines_df.iterrows():
        open_price = float(current_bar["Open"]) if current_bar else None
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])

        if current_bar is None:
            # Start new bar with this kline
            current_bar = {
                "Open Time": row["Open Time"],
                "Open": row["Open"],
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": row["Volume"],
                "Close Time": row["Close Time"],
                "Quote Asset Volume": row["Quote Asset Volume"],
                "Number of Trades": row["Number of Trades"],
                "Taker Buy Base Asset Volume": row["Taker Buy Base Asset Volume"],
                "Taker Buy Quote Asset Volume": row["Taker Buy Quote Asset Volume"],
                "Ignore": 0,
            }
            continue

        # Update aggregates and extremes
        current_bar["High"] = max(current_bar["High"], high)
        current_bar["Low"] = min(current_bar["Low"], low)
        agg_cols = [
            "Volume",
            "Quote Asset Volume",
            "Number of Trades",
            "Taker Buy Base Asset Volume",
            "Taker Buy Quote Asset Volume",
        ]
        for agg_col in agg_cols:
            current_bar[agg_col] += row[agg_col]

        # Check trigger: if High or Low exceeds range from open
        up_trigger = high >= open_price + range_size
        down_trigger = low <= open_price - range_size
        if up_trigger or down_trigger:
            # Close current bar at the extreme that triggered
            if up_trigger:
                current_bar["Close"] = high
            else:
                current_bar["Close"] = low
            current_bar["Close Time"] = row["Close Time"]
            bars.append(current_bar)

            # Start new bar at the trigger price (reversal)
            new_open = high if up_trigger else low
            current_bar = {
                "Open Time": row["Close Time"],
                "Open": new_open,
                "High": max(new_open, high),
                "Low": min(new_open, low),
                "Close": close,
                "Volume": row["Volume"],
                "Close Time": row["Close Time"],
                "Quote Asset Volume": row["Quote Asset Volume"],
                "Number of Trades": row["Number of Trades"],
                "Taker Buy Base Asset Volume": row["Taker Buy Base Asset Volume"],
                "Taker Buy Quote Asset Volume": row["Taker Buy Quote Asset Volume"],
                "Ignore": 0,
            }

    # Append final bar
    if current_bar:
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
    console.print(
        f"[{now_hhmmss}] 📊 Status: {SYMBOL} @ {last_price:.2f}, Range: {curr_range_pct:.1f}%, Trades: {last_trades}, Health: ✅ OK, Rows: {rows}, Last trade: {last_hhmmss}, State: CONNECTED",
        style="cyan",
    )


def load_historical() -> None:
    """Bootstrap with historical data."""
    global df, last_processed_time
    now_utc = int(datetime.now(timezone("UTC")).timestamp() * 1000)
    start_time = now_utc - HIST_DEPTH_SECONDS * 1000

    with Progress(SpinnerColumn(), console=console) as progress:
        task = progress.add_task("📥 Fetching historical 1s data...", total=None)
        klines_df = fetch_klines(start_time, now_utc)
        progress.remove_task(task)

    if klines_df.empty:
        console.print("🚀 Fresh start! No historical data.", style="italic cyan")
        return

    bars = aggregate_range_bars(klines_df)
    if bars:
        df = pd.DataFrame(bars)
        last_processed_time = int(
            datetime.strptime(
                klines_df.iloc[-1]["Close Time"], "%Y-%m-%d %H:%M:%S.%f"
            ).timestamp()
            * 1000
        )
        save_csv()
        console.print(
            f"✅ Loaded {len(bars)} 1R range bars from history (range: {range_size}) 📊",
            style="green",
        )


def poll_new_data() -> bool:
    """Poll for new 1s klines since last processed."""
    global df, last_processed_time
    if last_processed_time is None:
        return False

    now_utc = int(datetime.now(timezone("UTC")).timestamp() * 1000)
    klines_df = fetch_klines(last_processed_time + 1, now_utc)

    if klines_df.empty:
        return False

    # Update last processed
    new_last_str = klines_df.iloc[-1]["Close Time"]
    new_last = int(
        datetime.strptime(new_last_str, "%Y-%m-%d %H:%M:%S.%f").timestamp() * 1000
    )

    bars = aggregate_range_bars(klines_df)
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

    last_processed_time = new_last
    return updated


def save_csv():
    """Save DF to CSV atomically."""
    if df.empty:
        return
    temp_file = csv_file + ".tmp"
    df.to_csv(temp_file, index=False)
    os.replace(temp_file, csv_file)


def main():
    """Main loop."""
    archive_csv()
    get_symbol_ticksize(SYMBOL)  # Validate early

    load_historical()
    if not df.empty:
        print_status()

    while running:
        while True:
            try:
                updated = poll_new_data()
                print_status()
                break
            except Exception as e:
                console.print(
                    f"🛑 Connection error: {e}. Retrying in 5s...", style="bold red"
                )
                time.sleep(5)
        time.sleep(POLL_INTERVAL_SEC)

    console.print("👋 Session ended. Check historic_df_alpha.csv!", style="bold blue")


if __name__ == "__main__":
    main()
