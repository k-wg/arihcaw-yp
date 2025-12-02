import asyncio
import logging
import os
import random
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional, Tuple

import pandas as pd
from binance import ThreadedWebsocketManager
from binance.client import Client
from binance.exceptions import BinanceAPIException
from pytz import timezone
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("historic_df_alpha.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Config
SYMBOL = "SOLFDUSD"
RANGE_SIZE = 0.0
PREFETCH_HOURS = 2  # Hours of historical data to prefetch
PREFETCH_SECONDS = PREFETCH_HOURS * 3600  # Convert to seconds
GAP_DETECTION_THRESHOLD = 300  # 5 minutes threshold for gap detection
KLINE_INTERVAL_SECONDS = 1  # 1-second klines
MAX_KLINES_PER_REQUEST = 1000  # Binance API limit

# Globals
console = Console()
client = Client()
tws = None
kenya_tz = timezone("Africa/Nairobi")
csv_file = "historic_df_alpha.csv"
df = pd.DataFrame()
running = True
tick_size = 0.0
range_size = 0.0

# WebSocket data storage
klines_data: Deque[Dict[str, Any]] = deque(maxlen=10000)
last_kline_time: Optional[datetime] = None
data_lock = threading.Lock()

# Range bar tracking
current_bar: Optional[Dict[str, Any]] = None
bar_klines: List[Dict[str, Any]] = []  # Track klines for current bar

# WebSocket management
ws_connected = False
ws_reconnect_delay = 5
ws_max_reconnect_delay = 60

# State tracking
data_gap_filling = False
prefetch_completed = False  # Start with False since we'll prefetch
gap_filling_in_progress = False

# Coin flip tracking
update_counter = 0

# Gap detection
last_data_update_time: Optional[datetime] = None
last_gap_check_time = time.time()
GAP_CHECK_INTERVAL = 60  # Check for gaps every 60 seconds


def signal_handler(sig, frame):
    global running
    running = False
    console.print("\n🛑 Stopping gracefully...", style="bold yellow")
    if tws:
        try:
            tws.stop()
        except:
            pass
    save_csv()
    os._exit(0)  # Use os._exit instead of sys.exit


signal.signal(signal.SIGINT, signal_handler)


def perform_coin_flip_clear():
    """Perform a coin flip and clear screen if heads."""
    global update_counter

    update_counter += 1

    # Check if it's the 777th update
    if update_counter % 777 == 0:
        # Perform coin flip
        flip = random.choice(["heads", "tails"])

        if flip == "heads":
            # Clear screen seamlessly
            subprocess.run(["clear"], shell=False)
            logger.debug("Screen cleared after coin flip (heads)")
            return True

    return False


def archive_csv():
    """Archive existing CSV with UUID if present."""
    if os.path.exists(csv_file):
        short_uuid = uuid.uuid4().hex[:8]
        archive_name = (
            f"{csv_file.rsplit('.', 1)[0]}_{short_uuid}.{csv_file.rsplit('.', 1)[1]}"
        )
        os.rename(csv_file, archive_name)
        console.print(f"🗄️ Archived previous CSV to {archive_name}", style="green")
        logger.info(f"Archived previous CSV to {archive_name}")


def format_timestamp(ts_ms: int) -> str:
    """Convert UTC ms to Kenya TZ string matching sample format."""
    dt_utc = datetime.fromtimestamp(ts_ms / 1000, tz=timezone("UTC"))
    dt_kenya = dt_utc.astimezone(kenya_tz)
    return dt_kenya.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse timestamp string back to datetime object."""
    return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f").replace(
        tzinfo=kenya_tz
    )


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
            range_size = tick_size
            console.print(
                f"🔧 Auto-set 1R range_size to tickSize: {range_size}",
                style="italic blue",
            )
            logger.info(f"Auto-set range_size to tickSize: {range_size}")
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
        logger.error(f"Error fetching symbol info: {e}")
        os._exit(1)  # Use os._exit instead of sys.exit


def fetch_historical_klines(start_time: int, end_time: int) -> List[Dict[str, Any]]:
    """Fetch historical klines from Binance API."""
    try:
        console.print(
            f"📥 Fetching historical klines from {format_timestamp(start_time)} to {format_timestamp(end_time)}...",
            style="yellow",
        )
        logger.info(f"Fetching historical klines from {start_time} to {end_time}")
        
        klines = client.get_klines(
            symbol=SYMBOL,
            interval=Client.KLINE_INTERVAL_1SECOND,
            startTime=start_time,
            endTime=end_time,
            limit=1000
        )
        
        historical_data = []
        for kline in klines:
            historical_data.append({
                "timestamp": kline[0],  # Open time
                "open_time": kline[0],
                "close_time": kline[6],
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5]),
                "quote_volume": float(kline[7]),
                "trades": kline[8],
                "interval": "1s",
                "is_final": True,
            })
        
        console.print(
            f"✅ Fetched {len(historical_data)} historical klines", 
            style="green"
        )
        logger.info(f"Fetched {len(historical_data)} historical klines")
        return historical_data
        
    except Exception as e:
        console.print(f"❌ Error fetching historical klines: {e}", style="red")
        logger.error(f"Error fetching historical klines: {e}")
        return []


def prefetch_historical_data():
    """Prefetch 2 hours of historical data before starting real-time updates."""
    global prefetch_completed, klines_data, data_gap_filling
    
    data_gap_filling = True
    console.print("🔄 Prefetching 2 hours of historical data...", style="yellow")
    
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = end_time - (PREFETCH_HOURS * 3600 * 1000)
    
    # Calculate chunk size based on 1000 candle limit for 1-second klines
    # 1000 candles * 1000 ms/second = 1,000,000 ms per chunk
    chunk_size_ms = MAX_KLINES_PER_REQUEST * KLINE_INTERVAL_SECONDS * 1000
    
    all_historical_data = []
    
    current_start = start_time
    while current_start < end_time:
        current_end = min(current_start + chunk_size_ms, end_time)
        chunk_data = fetch_historical_klines(current_start, current_end)
        all_historical_data.extend(chunk_data)
        current_start = current_end + 1  # +1 to avoid overlap
        time.sleep(0.1)  # Rate limiting
    
    # Add historical data to klines_data
    with data_lock:
        for kline in all_historical_data:
            klines_data.append(kline)
    
    console.print(
        f"✅ Prefetch complete: {len(all_historical_data)} klines loaded", 
        style="bold green"
    )
    logger.info(f"Prefetch complete: {len(all_historical_data)} klines loaded")
    
    data_gap_filling = False
    prefetch_completed = True


def kline_handler(msg):
    """Handle incoming WebSocket kline messages."""
    global last_kline_time, ws_connected, last_data_update_time

    if "e" in msg and msg["e"] == "kline":
        kline = msg["k"]
        # Process both open and closed klines to get maximum data
        with data_lock:
            klines_data.append(
                {
                    "timestamp": kline["t"],  # Open time
                    "open_time": kline["t"],
                    "close_time": kline["T"],
                    "open": float(kline["o"]),
                    "high": float(kline["h"]),
                    "low": float(kline["l"]),
                    "close": float(kline["c"]),
                    "volume": float(kline["v"]),
                    "quote_volume": float(kline["q"]),
                    "trades": kline["n"],
                    "interval": kline["i"],
                    "is_final": kline["x"],
                }
            )
            last_kline_time = datetime.now()
            last_data_update_time = datetime.now()
            ws_connected = True

        # Debug: Print first kline to verify connection
        if len(klines_data) == 1 and not data_gap_filling:
            console.print(
                f"✅ First kline received: {kline['c']} at {datetime.now()}",
                style="green",
            )
            logger.info(f"First kline received: {kline['c']}")


def check_range_completion(bar: Dict[str, Any]) -> bool:
    """Check if bar should complete based on span (high - low) >= range_size."""
    high = float(bar["High"])
    low = float(bar["Low"])
    span = high - low
    return span >= range_size


def create_completed_bar(
    open_price: float,
    high: float,
    low: float,
    close_price: float,
    klines: List[Dict[str, Any]],
    open_time: int,
    close_time: int,
) -> Dict[str, Any]:
    """Create a completed range bar from accumulated klines."""
    # Aggregate volume and trades from all klines in this range bar
    total_volume = sum(kline["volume"] for kline in klines)
    total_quote_volume = sum(kline["quote_volume"] for kline in klines)
    total_trades = sum(kline["trades"] for kline in klines)

    # For kline data, we don't have taker buy volumes, so we'll use estimates
    # or set them to 0. You might want to modify this based on your needs
    taker_buy_base = total_volume * 0.5  # Estimate - adjust as needed
    taker_buy_quote = total_quote_volume * 0.5  # Estimate - adjust as needed

    return {
        "Open Time": format_timestamp(open_time),
        "Open": open_price,
        "High": high,
        "Low": low,
        "Close": close_price,
        "Volume": total_volume,
        "Close Time": format_timestamp(close_time),
        "Quote Asset Volume": total_quote_volume,
        "Number of Trades": total_trades,
        "Taker Buy Base Asset Volume": taker_buy_base,
        "Taker Buy Quote Asset Volume": taker_buy_quote,
        "Ignore": 0,
    }


def handle_large_price_jump(
    price: float, timestamp: int, kline_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Handle true TradingView behavior for large price jumps using kline data.
    Only creates bars for actual price movements, no artificial intermediate bars.
    """
    global current_bar, bar_klines

    if current_bar is None:
        return []

    completed_bars = []
    open_price = current_bar["open_price"]

    # Calculate how many range bars we can create from this jump
    price_move = abs(price - open_price)
    possible_bars = int(price_move // range_size)

    if possible_bars >= 2:
        # Large jump detected - complete current bar and potentially create one more
        console.print(
            f"⚠️ Large price jump: {price_move:.4f} ({possible_bars} possible bars)",
            style="yellow",
        )
        logger.info(
            f"Large price jump: {price_move:.4f} ({possible_bars} possible bars)"
        )

        # Complete the current bar first (using the extreme that triggers completion)
        if price > open_price:
            # Up movement - complete bar at the high extreme
            completed_high = open_price + range_size
            completed_low = current_bar["low"]
            completed_close = completed_high
        else:
            # Down movement - complete bar at the low extreme
            completed_high = current_bar["high"]
            completed_low = open_price - range_size
            completed_close = completed_low

        # Create the completed bar with accumulated klines
        completed_bar = create_completed_bar(
            open_price,
            completed_high,
            completed_low,
            completed_close,
            bar_klines,
            current_bar["open_time"],
            timestamp,
        )
        completed_bars.append(completed_bar)

        # Check if remaining price movement can create another immediate bar
        remaining_move = abs(price - completed_close)
        if remaining_move >= range_size:
            # Create one additional bar from the remaining movement
            if price > completed_close:
                # Additional up bar
                additional_open = completed_close
                additional_high = price
                additional_low = additional_open
                additional_close = price
            else:
                # Additional down bar
                additional_open = completed_close
                additional_high = additional_open
                additional_low = price
                additional_close = price

            # Create bar with just this kline
            additional_klines = [kline_data]

            additional_bar = create_completed_bar(
                additional_open,
                additional_high,
                additional_low,
                additional_close,
                additional_klines,
                timestamp,
                timestamp,
            )
            completed_bars.append(additional_bar)

            # Reset for next bar starting from the final close
            current_bar = {
                "open_price": additional_close,
                "high": additional_close,
                "low": additional_close,
                "close_price": additional_close,
                "open_time": timestamp,
                "close_time": timestamp,
            }
            bar_klines = []
        else:
            # Start new bar with remaining price movement
            current_bar = {
                "open_price": completed_close,
                "high": max(completed_close, price),
                "low": min(completed_close, price),
                "close_price": price,
                "open_time": timestamp,
                "close_time": timestamp,
            }
            bar_klines = [kline_data]

    return completed_bars


def process_kline_into_bar(kline_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Process a single kline into the current range bar.
    Returns completed bar if range condition met, None otherwise.
    """
    global current_bar, bar_klines

    price = kline_data["close"]  # Use close price for range bar processing
    timestamp = kline_data["close_time"]
    high = kline_data["high"]
    low = kline_data["low"]

    # Add kline to current bar's kline list
    bar_klines.append(kline_data)

    # Initialize or update current bar
    if current_bar is None:
        current_bar = {
            "open_price": price,
            "high": high,
            "low": low,
            "close_price": price,
            "open_time": kline_data["open_time"],
            "close_time": timestamp,
        }
        return None

    # Update extremes using the kline's high/low
    current_bar["high"] = max(current_bar["high"], high)
    current_bar["low"] = min(current_bar["low"], low)
    current_bar["close_price"] = price
    current_bar["close_time"] = timestamp

    # Check for range completion
    if check_range_completion({"High": current_bar["high"], "Low": current_bar["low"]}):
        # Bar completed - create the finalized bar
        completed_bar = create_completed_bar(
            current_bar["open_price"],
            current_bar["high"],
            current_bar["low"],
            current_bar["close_price"],
            bar_klines,
            current_bar["open_time"],
            current_bar["close_time"],
        )

        # Reset for next bar - start from close of completed bar
        current_bar = {
            "open_price": current_bar["close_price"],
            "high": current_bar["close_price"],
            "low": current_bar["close_price"],
            "close_price": current_bar["close_price"],
            "open_time": timestamp,
            "close_time": timestamp,
        }
        bar_klines = []  # Start fresh

        return completed_bar

    return None


def create_range_bar_from_klines() -> Tuple[List[Dict[str, Any]], bool]:
    """
    Process all pending klines into true TradingView-style range bars.
    Returns: (completed_bars, has_new_bars)
    """
    global current_bar

    with data_lock:
        if not klines_data:
            return [], False

        all_completed_bars = []

        for kline in list(klines_data):
            price = kline["close"]
            timestamp = kline["close_time"]
            high = kline["high"]
            low = kline["low"]

            # First check for large jumps
            if current_bar is not None:
                open_price = current_bar["open_price"]
                price_move = abs(price - open_price)
                possible_bars = int(price_move // range_size)

                if possible_bars >= 2:
                    large_jump_bars = handle_large_price_jump(price, timestamp, kline)
                    if large_jump_bars:
                        all_completed_bars.extend(large_jump_bars)
                        continue  # Skip normal processing for this kline

            # Normal processing - single bar completion check
            completed_bar = process_kline_into_bar(kline)
            if completed_bar:
                all_completed_bars.append(completed_bar)

        # Clear processed klines
        klines_data.clear()

        return all_completed_bars, len(all_completed_bars) > 0


def print_status():
    """Print the status line."""
    global current_bar, data_gap_filling, prefetch_completed, gap_filling_in_progress

    now_hhmmss = datetime.now(kenya_tz).strftime("%H:%M:%S")

    if df.empty and current_bar is None:
        console.print(
            f"[{now_hhmmss}] 📊 Status: Waiting for first kline...", style="yellow"
        )
        return

    # Get last completed bar info
    if not df.empty:
        last_close_time_str = df["Close Time"].iloc[-1]
        last_hhmmss = last_close_time_str.split()[1][:8]
        last_price = float(df["Close"].iloc[-1])
        last_trades = int(df["Number of Trades"].iloc[-1])
        last_high = float(df["High"].iloc[-1])
        last_low = float(df["Low"].iloc[-1])
        last_span = last_high - last_low
        rows = len(df)
    else:
        last_hhmmss = "N/A"
        last_price = 0.0
        last_trades = 0
        last_span = 0.0
        rows = 0

    # Get current bar progress
    if current_bar:
        current_open = float(current_bar["open_price"])
        current_high = float(current_bar["high"])
        current_low = float(current_bar["low"])
        current_span = current_high - current_low
        range_progress = (current_span / range_size) * 100
        current_klines = len(bar_klines)
    else:
        range_progress = 0.0
        current_klines = 0
        current_span = 0.0

    ws_status = "✅ CONNECTED" if ws_connected else "⚠️ RECONNECTING"

    # Add status indicators
    status_indicators = []
    if data_gap_filling:
        status_indicators.append("🔄 FILLING GAPS")
    if gap_filling_in_progress:
        status_indicators.append("📥 GAP FILLING")
    if not prefetch_completed:
        status_indicators.append("⏳ PREFETCHING")
    
    status_str = " | ".join(status_indicators) if status_indicators else ""

    console.print(
        f"[{now_hhmmss}] 📊 {SYMBOL} | Last: {last_price:.4f} | Span: {current_span:.4f}/{range_size} ({range_progress:.1f}%) | Klines: {current_klines} | Health: {ws_status} | Bars: {rows} {status_str}",
        style="cyan",
    )


def check_for_data_gaps() -> bool:
    """
    Check if there are data gaps and fill them if needed.
    Returns True if gap filling was performed, False otherwise.
    """
    global last_data_update_time, gap_filling_in_progress, data_gap_filling
    
    if not prefetch_completed or gap_filling_in_progress:
        return False
    
    # Check if we have data update time
    if last_data_update_time is None:
        return False
    
    # Check if data is stale (more than threshold seconds since last update)
    current_time = datetime.now()
    time_since_last_update = (current_time - last_data_update_time).total_seconds()
    
    if time_since_last_update > GAP_DETECTION_THRESHOLD:
        console.print(
            f"⚠️ Data gap detected: {time_since_last_update:.0f} seconds since last update", 
            style="yellow"
        )
        logger.warning(f"Data gap detected: {time_since_last_update:.0f} seconds")
        
        gap_filling_in_progress = True
        
        # Calculate gap start and end times
        gap_start_ms = int(last_data_update_time.timestamp() * 1000) + 1000  # +1 second
        gap_end_ms = int(current_time.timestamp() * 1000)
        
        if gap_end_ms - gap_start_ms > 1000:  # Only fill if gap is > 1 second
            console.print(
                f"🔄 Filling gap from {format_timestamp(gap_start_ms)} to {format_timestamp(gap_end_ms)}...",
                style="yellow"
            )
            
            data_gap_filling = True
            
            # Fetch gap data in chunks respecting the 1000 candle limit
            chunk_size_ms = MAX_KLINES_PER_REQUEST * KLINE_INTERVAL_SECONDS * 1000
            gap_data = []
            
            current_chunk_start = gap_start_ms
            while current_chunk_start < gap_end_ms:
                current_chunk_end = min(current_chunk_start + chunk_size_ms, gap_end_ms)
                chunk_data = fetch_historical_klines(current_chunk_start, current_chunk_end)
                gap_data.extend(chunk_data)
                current_chunk_start = current_chunk_end + 1
                time.sleep(0.1)  # Rate limiting
            
            if gap_data:
                with data_lock:
                    # Add gap data to the beginning of klines_data to maintain chronological order
                    klines_data.extendleft(reversed(gap_data))
                
                console.print(
                    f"✅ Gap filled: {len(gap_data)} klines added", 
                    style="green"
                )
                logger.info(f"Gap filled: {len(gap_data)} klines added")
            else:
                console.print("⚠️ No data available for gap period", style="yellow")
        
        gap_filling_in_progress = False
        data_gap_filling = False
        last_data_update_time = current_time  # Update after filling
        
        return True
    
    return False


def process_new_data() -> bool:
    """Process new kline data from WebSocket into range bars."""
    global df, last_data_update_time

    completed_bars, has_new_bars = create_range_bar_from_klines()
    updated = False

    if has_new_bars and completed_bars:
        for i, bar in enumerate(completed_bars):
            close_time_str = bar["Close Time"]
            hhmmss = close_time_str.split()[1][:8]
            open_price = float(bar["Open"])
            high = float(bar["High"])
            low = float(bar["Low"])
            close = float(bar["Close"])
            span = high - low
            trades = int(bar["Number of Trades"])
            volume = float(bar["Volume"])
            direction = "🟢" if close > open_price else "🔴"

            multi_bar_indicator = (
                f" [{i+1}/{len(completed_bars)}]" if len(completed_bars) > 1 else ""
            )

            # Determine source indicator
            source_indicator = "[LIVE]" if not data_gap_filling else "[GAP FILL]"

            console.print(
                f"[{hhmmss}] {direction} {source_indicator} Range bar closed{multi_bar_indicator} | O:{open_price:.4f} H:{high:.4f} L:{low:.4f} C:{close:.4f} | Span:{span:.4f} | Trades:{trades} Vol:{volume:.2f}",
                style="bold green" if close > open_price else "bold red",
            )
            logger.info(
                f"Range bar closed: O:{open_price:.4f} H:{high:.4f} L:{low:.4f} C:{close:.4f}"
            )

        new_df = pd.DataFrame(completed_bars)
        df = pd.concat([df, new_df], ignore_index=True)
        updated = True
        save_csv()
        
        # Update last data update time
        last_data_update_time = datetime.now()

    return updated


def save_csv():
    """Save DF to CSV atomically."""
    if df.empty:
        return
    try:
        temp_file = csv_file + ".tmp"
        df.to_csv(temp_file, index=False)
        os.replace(temp_file, csv_file)
        logger.debug("CSV saved successfully")
    except Exception as e:
        console.print(f"❌ Error saving CSV: {e}", style="red")
        logger.error(f"Error saving CSV: {e}")


def start_websocket():
    """Start WebSocket connection using ThreadedWebsocketManager for klines."""
    global tws, ws_connected, ws_reconnect_delay

    try:
        # Stop existing connection if any
        if tws:
            try:
                tws.stop()
                time.sleep(1)
            except Exception as e:
                logger.warning(f"Error stopping existing WebSocket: {e}")

        # Reset connection status
        ws_connected = False

        # Initialize the websocket manager
        tws = ThreadedWebsocketManager()
        tws.start()

        # Start kline socket with 1s interval
        tws.start_kline_socket(
            symbol=SYMBOL,
            callback=kline_handler,
            interval=Client.KLINE_INTERVAL_1SECOND,
        )

        console.print(
            f"🔌 WebSocket started for {SYMBOL} 1s kline stream", style="bold green"
        )
        logger.info(f"WebSocket started for {SYMBOL}")
        ws_reconnect_delay = 5  # Reset reconnect delay on success
        return tws

    except BinanceAPIException as e:
        console.print(f"❌ Binance API error starting WebSocket: {e}", style="red")
        logger.error(f"Binance API error starting WebSocket: {e}")

        # Exponential backoff for reconnection
        ws_reconnect_delay = min(ws_reconnect_delay * 2, ws_max_reconnect_delay)
        console.print(f"🔄 Retrying in {ws_reconnect_delay} seconds...", style="yellow")
        time.sleep(ws_reconnect_delay)
        return start_websocket()  # Retry
    except Exception as e:
        console.print(f"❌ Error starting WebSocket: {e}", style="red")
        logger.error(f"Error starting WebSocket: {e}")

        # Exponential backoff for reconnection
        ws_reconnect_delay = min(ws_reconnect_delay * 2, ws_max_reconnect_delay)
        console.print(f"🔄 Retrying in {ws_reconnect_delay} seconds...", style="yellow")
        time.sleep(ws_reconnect_delay)
        return start_websocket()  # Retry


def check_websocket_health():
    """Check if WebSocket connection is healthy and reconnect if needed."""
    global ws_connected

    with data_lock:
        if last_kline_time is None:
            # No data ever received
            if ws_connected:
                ws_connected = False
                logger.warning("WebSocket health check: No data ever received")
            return False
        elif (datetime.now() - last_kline_time).seconds > 30:
            # Stale connection
            if ws_connected:
                console.print(
                    "⚠️ WebSocket connection stale, reconnecting...", style="yellow"
                )
                logger.warning("WebSocket connection stale")
                ws_connected = False
            return False
        else:
            # Healthy connection
            if not ws_connected:
                ws_connected = True
                console.print("✅ WebSocket connection healthy", style="green")
                logger.info("WebSocket connection healthy")
            return True


def main():
    """Main loop."""
    global running, tws, prefetch_completed

    console.print(
        Panel.fit(
            f"🚀 Starting Historic DF Alpha for {SYMBOL}\n"
            f"📊 Range Size: {RANGE_SIZE if RANGE_SIZE > 0 else 'Auto (tickSize)'}\n"
            f"⏰ Prefetch: {PREFETCH_HOURS} hours\n"
            f"🔍 Gap Detection: {GAP_DETECTION_THRESHOLD}s threshold\n"
            f"📦 Max Chunk: {MAX_KLINES_PER_REQUEST} candles ({MAX_KLINES_PER_REQUEST * KLINE_INTERVAL_SECONDS / 60:.1f} min)\n"
            f"💾 CSV: {csv_file}",
            title="Historic DF Alpha - Enhanced",
            style="bold blue",
        )
    )
    logger.info(f"Starting Historic DF Alpha for {SYMBOL} - Enhanced")

    archive_csv()
    get_symbol_ticksize(SYMBOL)

    # Prefetch historical data first
    prefetch_historical_data()

    # Start WebSocket connection after prefetch
    console.print("🔌 Starting WebSocket connection...", style="bold blue")
    start_websocket()

    # Wait a bit for initial connection
    time.sleep(3)

    last_health_check = time.time()
    health_check_interval = 10  # Check health every 10 seconds
    last_gap_check = time.time()
    gap_check_interval = GAP_CHECK_INTERVAL  # Check for gaps every interval

    try:
        while running:
            # Process any new data
            updated = process_new_data()

            # Check for coin flip clear on every iteration
            perform_coin_flip_clear()

            # Check for data gaps periodically
            current_time = time.time()
            if current_time - last_gap_check >= gap_check_interval:
                if check_for_data_gaps():
                    # If gap was filled, process the new data immediately
                    process_new_data()
                last_gap_check = current_time

            # Check WebSocket health periodically
            if current_time - last_health_check >= health_check_interval:
                if not check_websocket_health():
                    # Health check failed, reconnect
                    console.print(
                        "🔄 Reconnecting WebSocket due to health check failure...",
                        style="yellow",
                    )
                    logger.warning("WebSocket health check failed, reconnecting")
                    start_websocket()
                    time.sleep(3)  # Wait for reconnection

                last_health_check = current_time
                print_status()

            time.sleep(0.1)

    except KeyboardInterrupt:
        console.print("\n🛑 Keyboard interrupt received", style="yellow")
        logger.info("Keyboard interrupt received")
    except BinanceAPIException as e:
        console.print(f"🛑 Binance API error in main loop: {e}", style="bold red")
        logger.error(f"Binance API error in main loop: {e}")
    except Exception as e:
        console.print(f"🛑 Unexpected error in main loop: {e}", style="bold red")
        logger.error(f"Unexpected error in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if tws:
            try:
                tws.stop()
                logger.info("WebSocket stopped")
            except Exception as e:
                logger.warning(f"Error stopping WebSocket: {e}")
        save_csv()
        console.print(
            "👋 Session ended. Check historic_df_alpha.csv!", style="bold blue"
        )
        logger.info("Session ended")


if __name__ == "__main__":
    main()