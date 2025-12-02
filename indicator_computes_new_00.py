import logging
import os
import queue
import random
import subprocess
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from binance import ThreadedWebsocketManager
from binance.client import Client
from colorama import Back, Fore, Style, init
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from tabulate import tabulate

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Initialize Rich console
console = Console()

# Configuration
UPDATE_INTERVAL = 1  # seconds - adjustable update interval
DISPLAY_ROWS = 15  # number of rows to display in summary table
SYMBOL = "SOLFDUSD"  # Change to your trading pair
timeout_que = 3  # Time to choose option 1 - 3 before the default 1 activates

# Global variables for tracking script runtime
SCRIPT_START_TIME = None
LAST_UPDATE_TIME = None

# Token watchlist with icons and pairings
TOKEN_WATCHLIST = {
    "BTC": {"icon": "₿", "pairings": ["FDUSD", "USDT", "USDC"]},
    "ETH": {"icon": "🔷", "pairings": ["FDUSD", "USDT", "USDC"]},
    "BNB": {"icon": "⚡", "pairings": ["FDUSD", "USDT", "USDC"]},
    "DOGE": {"icon": "🐕", "pairings": ["FDUSD", "USDT", "USDC"]},
    "XRP": {"icon": "✖️", "pairings": ["FDUSD", "USDT", "USDC"]},
    "LINK": {"icon": "🔗", "pairings": ["FDUSD", "USDT", "USDC"]},
    "PAXG": {
        "icon": "🏆",
        "pairings": ["USDT", "USDC", "FDUSD"],
    },  # Changed order for PAXG
    "ASTER": {
        "icon": "🌟",
        "pairings": ["USDT", "USDC", "FDUSD"],
    },  # Changed order for ASTER
    "SOL": {"icon": "🔶", "pairings": ["FDUSD", "USDT", "USDC"]},
}

# WebSocket market data storage
token_price_data = {}
token_price_lock = threading.Lock()
websocket_manager = None
last_token_data_update = 0
TOKEN_UPDATE_INTERVAL = 30  # Update token data every 30 seconds

# Global decimal precision storage - NO MORE ROUNDING
SYMBOL_DECIMAL_PRECISION = 8  # Default high precision

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def get_symbol_info(client, symbol):
    """
    Get symbol information including tick size for decimal precision
    Now returns maximum precision without rounding
    """
    try:
        info = client.get_exchange_info()
        for sym_info in info["symbols"]:
            if sym_info["symbol"] == symbol:
                # Return maximum precision for calculations
                return 8  # Always use maximum precision, no rounding
        return 8  # Default high precision
    except Exception as e:
        logger.error(f"Error getting symbol info for {symbol}: {e}")
        return 8  # Default high precision


class WebSocketMarketData:
    """
    WebSocket-based market data manager for real-time token prices
    """

    def __init__(self):
        self.token_data = {}
        self.websocket_manager = None
        self.connected = False
        self.last_update_time = 0
        self.price_0300_cache = {}  # Cache for 03:00 prices
        self.cache_initialized = False
        self.last_cache_update_date = None  # Track when cache was last updated

    def start_websocket(self):
        """Start WebSocket connection for all tokens"""
        try:
            self.websocket_manager = ThreadedWebsocketManager()
            self.websocket_manager.start()

            # Create streams for all tokens with all pairings
            streams = []
            for token, data in TOKEN_WATCHLIST.items():
                for pairing in data["pairings"]:
                    symbol = f"{token}{pairing}"
                    streams.append(f"{symbol.lower()}@ticker")

            # Start combined stream
            self.websocket_manager.start_multiplex_socket(
                streams=streams, callback=self._handle_websocket_message
            )

            self.connected = True
            console.print("🔌 WebSocket market data connected", style="bold green")
            return True

        except Exception as e:
            console.print(f"❌ WebSocket connection failed: {e}", style="red")
            return False

    def _handle_websocket_message(self, msg):
        """Handle incoming WebSocket messages"""
        try:
            if "data" in msg:
                data = msg["data"]
                symbol = data["s"]

                # Extract token and pairing from symbol
                for token, token_data in TOKEN_WATCHLIST.items():
                    for pairing in token_data["pairings"]:
                        if symbol == f"{token}{pairing}":
                            # Update token data
                            with token_price_lock:
                                self.token_data[token] = {
                                    "symbol": symbol,
                                    "current_price": float(data["c"]),
                                    "price_change_percent": float(data["P"]),
                                    "high_price": float(data["h"]),
                                    "low_price": float(data["l"]),
                                    "volume": float(data["v"]),
                                    "quote_volume": float(data["q"]),
                                    "last_update": time.time(),
                                    "pairing": pairing,
                                }
                            break

        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")

    def should_refresh_0300_cache(self):
        """Check if we should refresh the 03:00 cache based on current time"""
        if self.last_cache_update_date is None:
            return True

        # Get current Kenyan time
        utc_now = datetime.now(timezone.utc)
        kenyan_time = utc_now + timedelta(hours=3)
        current_date = kenyan_time.date()

        # Determine which date's 03:00 price we need
        if kenyan_time.hour >= 3:
            # After 03:00, we need today's 03:00 price
            required_date = current_date
        else:
            # Before 03:00, we need yesterday's 03:00 price
            required_date = current_date - timedelta(days=1)

        # Check if our cache has the required date
        cache_data = self.price_0300_cache.get("SOL")
        if cache_data and "cache_date" in cache_data:
            cache_date = cache_data["cache_date"]
            if cache_date != required_date:
                logger.info(f"🔄 Cache refresh needed: cache_date={cache_date}, required_date={required_date}")
                return True  # Cache has wrong date, need refresh
        else:
            return True  # No cache or no date info
        
        # Also refresh if it's been more than 24 hours since last update (safety check)
        utc_now = datetime.now(timezone.utc)
        if self.last_cache_update_date:
            hours_since_update = (utc_now - self.last_cache_update_date).total_seconds() / 3600
            if hours_since_update > 24:
                logger.info(f"🔄 Cache refresh needed: {hours_since_update:.1f} hours since last update")
                return True

        return False

    def initialize_0300_prices(self, client):
        """Initialize 03:00 prices and refresh when needed"""
        # Check if we need to refresh cache
        if not self.should_refresh_0300_cache() and self.cache_initialized:
            logger.info("✅ Cache is up to date, skipping refresh")
            return

        console.print("🕒 Initializing/Refreshing 03:00 price cache...", style="yellow")
        with Progress(SpinnerColumn(), console=console) as progress:
            task = progress.add_task(
                "Fetching 03:00 prices...", total=len(TOKEN_WATCHLIST)
            )

            for token in TOKEN_WATCHLIST.keys():
                try:
                    # Try each pairing in order of preference
                    price_0300 = None
                    used_pairing = None
                    cache_date = None

                    for pairing in TOKEN_WATCHLIST[token]["pairings"]:
                        symbol = f"{token}{pairing}"
                        try:
                            price_0300, cache_date = get_price_at_kenyan_0300_1sec(client, symbol)
                            if price_0300 is not None:
                                used_pairing = pairing
                                break
                        except Exception as e:
                            logger.debug(f"Failed to get 03:00 price for {symbol}: {e}")
                            continue

                    if price_0300 is not None:
                        self.price_0300_cache[token] = {
                            "price": price_0300,
                            "pairing": used_pairing,
                            "timestamp": datetime.now(timezone.utc),
                            "cache_date": cache_date,  # Store the date of the cached price
                        }
                        logger.info(
                            f"✅ Cached 03:00 price for {token}{used_pairing} on {cache_date}: {price_0300:.8f}"
                        )
                    else:
                        logger.warning(
                            f"❌ Could not get 03:00 price for {token} with any pairing"
                        )

                except Exception as e:
                    logger.error(f"Error getting 03:00 price for {token}: {e}")

                progress.advance(task)

        # Update cache timestamp
        self.last_cache_update_date = datetime.now(timezone.utc)
        self.cache_initialized = True
        
        # Log cache summary
        utc_now = datetime.now(timezone.utc)
        kenyan_time = utc_now + timedelta(hours=3)
        logger.info(f"✅ 03:00 price cache initialized/refreshed at {kenyan_time.strftime('%Y-%m-%d %H:%M:%S')} EAT")
        console.print("✅ 03:00 price cache initialized/refreshed", style="green")

    def get_token_performance(self):
        """Get token performance data from WebSocket"""
        global last_token_data_update

        current_time = time.time()

        # Only update token data periodically to reduce CPU usage
        if current_time - last_token_data_update < TOKEN_UPDATE_INTERVAL:
            return self._get_cached_token_performance()

        last_token_data_update = current_time
        token_data = {}

        with token_price_lock:
            for token, token_info in TOKEN_WATCHLIST.items():
                if token in self.token_data:
                    ws_data = self.token_data[token]
                    cache_data = self.price_0300_cache.get(token)

                    if cache_data and "price" in cache_data:
                        price_0300 = cache_data["price"]
                        current_price = ws_data["current_price"]
                        diff = current_price - price_0300
                        percent_change = (diff / price_0300) * 100
                        value_change = diff  # Actual value difference

                        token_data[token] = {
                            "icon": token_info["icon"],
                            "symbol": ws_data["symbol"],
                            "change": percent_change,
                            "value_change": value_change,
                            "current_price": current_price,
                            "price_0300": price_0300,
                            "pairing": ws_data.get("pairing", "N/A"),
                            "status": "OK",
                        }
                    else:
                        token_data[token] = {
                            "icon": token_info["icon"],
                            "symbol": f"{token}???",
                            "change": 0.0,
                            "value_change": 0.0,
                            "current_price": 0.0,
                            "status": "No 03:00 price",
                        }
                else:
                    token_data[token] = {
                        "icon": token_info["icon"],
                        "symbol": f"{token}???",
                        "change": 0.0,
                        "value_change": 0.0,
                        "current_price": 0.0,
                        "status": "No WebSocket data",
                    }

        return token_data

    def _get_cached_token_performance(self):
        """Get cached token performance data"""
        token_data = {}
        with token_price_lock:
            for token, token_info in TOKEN_WATCHLIST.items():
                if token in self.token_data:
                    ws_data = self.token_data[token]
                    cache_data = self.price_0300_cache.get(token)

                    if cache_data and "price" in cache_data:
                        price_0300 = cache_data["price"]
                        current_price = ws_data["current_price"]
                        diff = current_price - price_0300
                        percent_change = (diff / price_0300) * 100
                        value_change = diff  # Actual value difference

                        token_data[token] = {
                            "icon": token_info["icon"],
                            "symbol": ws_data["symbol"],
                            "change": percent_change,
                            "value_change": value_change,
                            "current_price": current_price,
                            "price_0300": price_0300,
                            "pairing": ws_data.get("pairing", "N/A"),
                            "status": "OK",
                        }
                    else:
                        token_data[token] = {
                            "icon": token_info["icon"],
                            "symbol": f"{token}???",
                            "change": 0.0,
                            "value_change": 0.0,
                            "current_price": 0.0,
                            "status": "No 03:00 price",
                        }
                else:
                    token_data[token] = {
                        "icon": token_info["icon"],
                        "symbol": f"{token}???",
                        "change": 0.0,
                        "value_change": 0.0,
                        "current_price": 0.0,
                        "status": "No WebSocket data",
                    }

        return token_data

    def stop(self):
        """Stop WebSocket connection"""
        if self.websocket_manager:
            self.websocket_manager.stop()
            self.connected = False
            console.print("🔌 WebSocket market data disconnected", style="yellow")


# Global WebSocket market data manager
market_data_manager = WebSocketMarketData()


def get_price_at_kenyan_0300_1sec(client, symbol):
    """
    Get the price at 03:00 Kenyan time (00:00 UTC) for any symbol using 1-second klines
    Returns price and the date of the price
    """
    try:
        # Get current UTC time
        utc_now = datetime.now(timezone.utc)

        # Convert to Kenyan time (UTC+3)
        kenyan_time = utc_now + timedelta(hours=3)

        # Set target time to 03:00:00 Kenyan time (which is 00:00:00 UTC)
        target_kenyan_time = kenyan_time.replace(
            hour=3, minute=0, second=0, microsecond=0
        )

        # Get the date of the target time
        target_date = target_kenyan_time.date()

        # If current Kenyan time is before 03:00, use yesterday's 03:00
        if kenyan_time < target_kenyan_time:
            target_kenyan_time = target_kenyan_time - timedelta(days=1)
            target_date = target_kenyan_time.date()

        # Convert back to UTC for Binance API
        target_utc = target_kenyan_time - timedelta(hours=3)
        timestamp = int(target_utc.timestamp() * 1000)

        # Get the kline for 03:00 Kenyan time using 1-second interval
        # Binance uses '1s' for 1-second intervals
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=Client.KLINE_INTERVAL_1SECOND,
            start_str=timestamp,
            end_str=timestamp + 1000,  # 1 second later
            limit=1,
        )

        if klines:
            price = float(klines[0][4])
            logger.debug(f"📊 Got 03:00 price for {symbol} on {target_date}: {price:.8f}")
            return price, target_date  # Return the close price and date
        else:
            # Fallback to 1-minute if 1-second fails
            klines = client.get_historical_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_1MINUTE,
                start_str=timestamp,
                end_str=timestamp + 60000,
                limit=1,
            )
            if klines:
                price = float(klines[0][4])
                logger.debug(f"📊 Got 03:00 price (1-min fallback) for {symbol} on {target_date}: {price:.8f}")
                return price, target_date
            else:
                logger.warning(f"❌ No kline data for {symbol} at 03:00 on {target_date}")
                return None, None

    except Exception as e:
        logger.error(f"❌ Error getting price at 03:00 for {symbol}: {e}")
        return None, None


def get_price_at_kenyan_0300_for_solfdusd(client):
    """
    Get the price at 03:00 Kenyan time (00:00 UTC) specifically for SOLFDUSD
    This is used for the daily difference in the table (not animated)
    """
    try:
        # Get current UTC time
        utc_now = datetime.now(timezone.utc)

        # Convert to Kenyan time (UTC+3)
        kenyan_time = utc_now + timedelta(hours=3)

        # Set target time to 03:00:00 Kenyan time (which is 00:00:00 UTC)
        target_kenyan_time = kenyan_time.replace(
            hour=3, minute=0, second=0, microsecond=0
        )

        # Get the date of the target time
        target_date = target_kenyan_time.date()

        # If current Kenyan time is before 03:00, use yesterday's 03:00
        if kenyan_time < target_kenyan_time:
            target_kenyan_time = target_kenyan_time - timedelta(days=1)
            target_date = target_kenyan_time.date()

        # Convert back to UTC for Binance API
        target_utc = target_kenyan_time - timedelta(hours=3)
        timestamp = int(target_utc.timestamp() * 1000)

        # Get the kline for 03:00 Kenyan time using 1-second interval
        try:
            klines = client.get_historical_klines(
                symbol="SOLFDUSD",
                interval=Client.KLINE_INTERVAL_1SECOND,
                start_str=timestamp,
                end_str=timestamp + 1000,  # 1 second later
                limit=1,
            )
        except Exception as e:
            logger.warning(f"1-second kline failed for SOLFDUSD, falling back to 1-minute: {e}")
            klines = None

        if klines and len(klines) > 0:
            price = float(klines[0][4])
            logger.info(f"📊 Got SOLFDUSD 03:00 price on {target_date}: {price:.8f} (1-second)")
            return price, target_date
        else:
            # Fallback to 1-minute
            klines = client.get_historical_klines(
                symbol="SOLFDUSD",
                interval=Client.KLINE_INTERVAL_1MINUTE,
                start_str=timestamp,
                end_str=timestamp + 60000,
                limit=1,
            )
            if klines and len(klines) > 0:
                price = float(klines[0][4])
                logger.info(f"📊 Got SOLFDUSD 03:00 price on {target_date}: {price:.8f} (1-minute fallback)")
                return price, target_date
            else:
                logger.error(f"❌ No kline data for SOLFDUSD at 03:00 on {target_date}")
                return None, None

    except Exception as e:
        logger.error(f"❌ Error getting price at 03:00 for SOLFDUSD: {e}")
        return None, None


def get_daily_difference_kenyan_time_solfdusd(client, cached_data=None):
    """
    Calculate daily difference for SOLFDUSD specifically (for table display)
    Uses cached data if available and date matches requirement
    """
    try:
        # Get current price from WebSocket or REST API
        current_price = None
        
        # First try WebSocket
        with token_price_lock:
            if "SOL" in market_data_manager.token_data:
                sol_data = market_data_manager.token_data["SOL"]
                # Only use if it's SOLFDUSD pairing
                if sol_data.get("pairing") == "FDUSD":
                    current_price = sol_data["current_price"]
                    logger.debug(f"📊 Using WebSocket SOLFDUSD price: {current_price:.8f}")
        
        # If not available from WebSocket, use REST API
        if current_price is None:
            try:
                ticker = client.get_symbol_ticker(symbol="SOLFDUSD")
                current_price = float(ticker["price"])
                logger.debug(f"📊 Using REST API SOLFDUSD price: {current_price:.8f}")
            except Exception as e:
                logger.error(f"❌ Error getting current price for SOLFDUSD: {e}")
                return "N/A"

        # Get current Kenyan time to determine which date's 03:00 price we need
        utc_now = datetime.now(timezone.utc)
        kenyan_time = utc_now + timedelta(hours=3)
        kenyan_hour = kenyan_time.hour
        
        if kenyan_hour >= 3:
            # After 03:00, we need today's 03:00 price
            required_date = kenyan_time.date()
            time_period = "today"
        else:
            # Before 03:00, we need yesterday's 03:00 price
            required_date = kenyan_time.date() - timedelta(days=1)
            time_period = "yesterday"
        
        logger.debug(f"🕒 Kenyan time: {kenyan_time.strftime('%Y-%m-%d %H:%M:%S')}, hour={kenyan_hour}")
        logger.debug(f"📅 Need {time_period}'s 03:00 price (date: {required_date})")

        # Check cache
        price_0300 = None
        cache_source = "none"
        
        if cached_data and "cache_date" in cached_data and "price" in cached_data:
            cache_date = cached_data["cache_date"]
            if cache_date == required_date:
                price_0300 = cached_data["price"]
                cache_source = "cache"
                logger.debug(f"✅ Using cached 03:00 price for {required_date}: {price_0300:.8f}")
            else:
                logger.debug(f"🔄 Cache mismatch: cache_date={cache_date}, required_date={required_date}")
        
        # If cache not valid, fetch fresh
        if price_0300 is None:
            price_0300, fetched_date = get_price_at_kenyan_0300_for_solfdusd(client)
            if price_0300 is None:
                logger.error(f"❌ Could not fetch 03:00 price for SOLFDUSD")
                return "N/A"
            cache_source = "fresh_fetch"
            logger.info(f"📊 Fetched fresh 03:00 price for SOLFDUSD on {fetched_date}: {price_0300:.8f}")
        
        # Calculate difference
        diff = current_price - price_0300
        percent = (diff / price_0300) * 100

        logger.info(f"📈 SOLFDUSD Daily Diff: Current={current_price:.8f}, 03:00={price_0300:.8f}, Diff={diff:+.8f}, %={percent:+.2f}% ({cache_source})")
        
        # Format without adding 0.08
        if percent > 0:
            return f"+{percent:.2f}%"
        else:
            return f"{percent:.2f}%"

    except Exception as e:
        logger.error(f"❌ Error calculating daily difference for SOLFDUSD: {e}")
        return "N/A"


def get_daily_difference_kenyan_time():
    """
    Calculate daily difference using Kenyan time (UTC+3)
    Now uses WebSocket data for current price instead of REST API
    This is for the animated market overview
    """
    try:
        # Get current price from WebSocket
        with token_price_lock:
            if "SOL" in market_data_manager.token_data:
                current_price = market_data_manager.token_data["SOL"]["current_price"]
            else:
                # Fallback to REST API if WebSocket not available
                client = Client()
                # Try different pairings for SOL
                price_found = False
                current_price = 0.0
                for pairing in ["FDUSD", "USDT", "USDC"]:
                    try:
                        symbol = f"SOL{pairing}"
                        ticker = client.get_symbol_ticker(symbol=symbol)
                        current_price = float(ticker["price"])
                        price_found = True
                        break
                    except:
                        continue

                if not price_found:
                    return "N/A"

        # Get 03:00 price from cache
        cache_data = market_data_manager.price_0300_cache.get("SOL")
        if cache_data and "price" in cache_data:
            price_0300 = cache_data["price"]
        else:
            return "N/A"

        # Calculate difference
        diff = current_price - price_0300
        percent = (diff / price_0300) * 100

        # Format without adding 0.08
        if percent > 0:
            return f"+{percent:.2f}%"
        else:
            return f"{percent:.2f}%"

    except Exception as e:
        logger.error(f"Error calculating daily difference: {e}")
        return "N/A"


def clear_screen():
    """Clear screen instantly without any delay"""
    try:
        subprocess.run("clear", shell=True, check=True)
    except Exception as e:
        logger.debug(f"Error clearing screen: {e}")


def should_clear_screen():
    """Coin flip to decide if we should clear the screen"""
    return random.choice([True, False])


class StatefulRSICalculator:
    """
    Stateful RSI calculator using Wilder's smoothing
    Maintains state between updates for incremental calculation
    """

    def __init__(self, length=14):
        self.length = length
        self.alpha = 1.0 / length
        self.previous_avg_gain = None
        self.previous_avg_loss = None
        self.previous_price = None
        self.initialized = False
        self.initial_prices = deque(maxlen=length + 1)

    def initialize(self, prices):
        """Initialize RSI with historical data"""
        if len(prices) < self.length + 1:
            return None

        changes = np.diff(prices)
        gains = np.where(changes >= 0, changes, 0.0)
        losses = np.where(changes < 0, -changes, 0.0)

        # Use SMA for initialization
        self.previous_avg_gain = np.mean(gains[: self.length])
        self.previous_avg_loss = np.mean(losses[: self.length])
        self.previous_price = prices[-1]
        self.initialized = True

        # Calculate first RSI value
        if self.previous_avg_loss == 0:
            return 100.0
        elif self.previous_avg_gain == 0:
            return 0.0
        else:
            rs = self.previous_avg_gain / self.previous_avg_loss
            return 100 - (100 / (1 + rs))

    def update(self, new_price):
        """Update RSI with new price incrementally using Wilder's smoothing"""
        if not self.initialized:
            self.initial_prices.append(new_price)
            if len(self.initial_prices) >= self.length + 1:
                return self.initialize(list(self.initial_prices))
            return None

        # Calculate price change
        change = new_price - self.previous_price
        gain = change if change > 0 else 0.0
        loss = -change if change < 0 else 0.0

        # Wilder's smoothing
        self.previous_avg_gain = (
            self.previous_avg_gain * (self.length - 1) + gain
        ) / self.length
        self.previous_avg_loss = (
            self.previous_avg_loss * (self.length - 1) + loss
        ) / self.length
        self.previous_price = new_price

        # Calculate RSI
        if self.previous_avg_loss == 0:
            return 100.0
        elif self.previous_avg_gain == 0:
            return 0.0
        else:
            rs = self.previous_avg_gain / self.previous_avg_loss
            return 100 - (100 / (1 + rs))

    def is_ready(self):
        return self.initialized


class StatefulSMA:
    """
    Simple Moving Average - Very reliable
    """

    def __init__(self, length=50):
        self.length = length
        self.values = deque(maxlen=length)
        self.initialized = False

    def update(self, new_value):
        if pd.isna(new_value):
            return np.nan

        self.values.append(new_value)
        
        if len(self.values) < self.length:
            return np.nan

        try:
            result = float(np.mean(self.values))
            self.initialized = True
            return result
        except:
            return np.nan

    def is_ready(self):
        return self.initialized


class StatefulMovingAverage:
    """Simple and reliable moving average"""

    def __init__(self, length):
        self.length = length
        self.prices = deque(maxlen=length)
        self.initialized = False

    def update(self, new_price):
        self.prices.append(new_price)

        if len(self.prices) < self.length:
            return np.nan

        try:
            ma_value = sum(self.prices) / len(self.prices)
            self.initialized = True
            return ma_value
        except:
            return np.nan

    def is_ready(self):
        return self.initialized


class StatefulFibonacci:
    """Simple Fibonacci calculator"""

    def __init__(self, length=5000):
        self.length = length
        self.prices = deque(maxlen=length)
        self.initialized = False

    def update(self, new_price):
        self.prices.append(new_price)

        if len(self.prices) < self.length:
            return {
                "level_100": np.nan,
                "level_764": np.nan,
                "level_618": np.nan,
                "level_500": np.nan,
                "level_382": np.nan,
                "level_236": np.nan,
                "level_000": np.nan,
            }

        try:
            maxr = max(self.prices)
            minr = min(self.prices)
            ranr = maxr - minr

            result = {
                "level_100": maxr,
                "level_764": maxr - 0.236 * ranr,
                "level_618": maxr - 0.382 * ranr,
                "level_500": maxr - 0.50 * ranr,
                "level_382": minr + 0.382 * ranr,
                "level_236": minr + 0.236 * ranr,
                "level_000": minr,
            }
            
            self.initialized = True
            return result
        except:
            return {
                "level_100": np.nan,
                "level_764": np.nan,
                "level_618": np.nan,
                "level_500": np.nan,
                "level_382": np.nan,
                "level_236": np.nan,
                "level_000": np.nan,
            }

    def is_ready(self):
        return self.initialized


class StatefulIndicatorEngine:
    """
    SIMPLIFIED indicator engine - Focus on reliability over complexity
    """

    def __init__(self, client=None):
        self.initialized = False
        self.last_processed_time = None
        self.current_df = pd.DataFrame()
        self.client = client
        self.last_solfdusd_cache = None  # Cache for SOLFDUSD daily difference

        # Initialize all calculators with SIMPLE implementations
        self.rsi = StatefulRSICalculator(length=14)
        self.rsi_ma50 = StatefulSMA(length=36)

        # Price moving averages
        self.ma2 = StatefulMovingAverage(length=2)
        self.ma7 = StatefulMovingAverage(length=7)
        self.ma14 = StatefulMovingAverage(length=14)
        self.ma50 = StatefulMovingAverage(length=50)
        self.ma100 = StatefulMovingAverage(length=100)
        self.ma200 = StatefulMovingAverage(length=200)
        self.ma350 = StatefulMovingAverage(length=350)
        self.ma500 = StatefulMovingAverage(length=500)

        # Fibonacci
        self.fibonacci = StatefulFibonacci(length=5000)

        # Track all calculators for easy access
        self.ma_calculators = {
            "short002": self.ma2,
            "short007": self.ma7,
            "short21": self.ma14,
            "short50": self.ma50,
            "long100": self.ma100,
            "long200": self.ma200,
            "long350": self.ma350,
            "long500": self.ma500,
        }

    def initialize_from_history(self, historical_df):
        """Initialize from historical data - SIMPLE AND RELIABLE"""
        if historical_df.empty:
            return False

        self.current_df = historical_df.copy()

        console.print(
            f"🔄 Initializing indicators from {len(self.current_df)} bars...", 
            style="yellow"
        )

        # Sequential computation
        rsi_list = []
        rsi_ma50_list = []
        ma_lists = {name: [] for name in self.ma_calculators.keys()}
        fib_lists = {
            "level_100": [], "level_764": [], "level_618": [], "level_500": [],
            "level_382": [], "level_236": [], "level_000": [],
        }
        
        # Get daily difference for each bar
        daily_diff_list = []

        for _, row in self.current_df.iterrows():
            close = row["Close"]

            # RSI with Wilder's smoothing
            rsi_val = self.rsi.update(close)
            rsi_list.append(rsi_val)

            # RSI MA50
            if pd.isna(rsi_val):
                rsi_ma50_val = np.nan
            else:
                rsi_ma50_val = self.rsi_ma50.update(rsi_val)
            rsi_ma50_list.append(rsi_ma50_val)

            # MAs
            for name, calc in self.ma_calculators.items():
                ma_val = calc.update(close)
                ma_lists[name].append(ma_val)

            # Fib
            fib_levels = self.fibonacci.update(close)
            for level, val in fib_levels.items():
                fib_lists[level].append(val)
            
            # Daily difference for SOLFDUSD
            daily_diff = get_daily_difference_kenyan_time_solfdusd(self.client, self.last_solfdusd_cache)
            daily_diff_list.append(daily_diff)

        # Assign to df
        self.current_df["rsi"] = rsi_list
        self.current_df["rsi_ma50"] = rsi_ma50_list
        for name, vals in ma_lists.items():
            self.current_df[name] = vals
        for level, vals in fib_lists.items():
            self.current_df[level] = vals
        self.current_df["daily_diff"] = daily_diff_list

        self.initialized = True
        if not self.current_df.empty:
            self.last_processed_time = self.current_df["Open Time"].iloc[-1]

        # Check which indicators are ready
        ready_indicators = []
        if self.rsi.is_ready():
            ready_indicators.append("RSI")
        if self.rsi_ma50.is_ready():
            ready_indicators.append("RSI_MA50")
        for name, calc in self.ma_calculators.items():
            if calc.is_ready():
                ready_indicators.append(name)
        if self.fibonacci.is_ready():
            ready_indicators.append("Fibonacci")

        console.print(
            f"✅ Indicators initialized! Ready: {', '.join(ready_indicators) if ready_indicators else 'None'}",
            style="green"
        )

        return True

    def process_new_data(self, new_rows_df):
        """Process new data - SIMPLE AND RELIABLE"""
        if not self.initialized or new_rows_df.empty:
            return False

        new_data = []

        for _, row in new_rows_df.iterrows():
            if (self.last_processed_time and row["Open Time"] <= self.last_processed_time):
                continue

            close = row["Close"]
            timestamp = row["Open Time"]

            # Calculate all indicators
            rsi_val = self.rsi.update(close)
            rsi_ma50_val = self.rsi_ma50.update(rsi_val) if not pd.isna(rsi_val) else np.nan

            ma_values = {}
            for name, calc in self.ma_calculators.items():
                ma_val = calc.update(close)
                ma_values[name] = ma_val

            fib_levels = self.fibonacci.update(close)
            
            # Daily difference for SOLFDUSD
            daily_diff = get_daily_difference_kenyan_time_solfdusd(self.client, self.last_solfdusd_cache)

            new_row = row.to_dict()
            new_row.update({
                "daily_diff": daily_diff,
                "rsi": rsi_val,
                "rsi_ma50": rsi_ma50_val,
                **ma_values,
                **fib_levels,
            })

            new_data.append(new_row)
            self.last_processed_time = timestamp

        if new_data:
            new_df = pd.DataFrame(new_data)
            self.current_df = pd.concat([self.current_df, new_df], ignore_index=True)
            return True

        return False

    def get_current_data(self):
        return self.current_df.copy()

    def update_solfdusd_cache(self, cache_data):
        """Update the SOLFDUSD cache for daily difference calculation"""
        self.last_solfdusd_cache = cache_data
        if cache_data and "cache_date" in cache_data:
            logger.debug(f"🔄 Updated SOLFDUSD cache with date: {cache_data['cache_date']}")


def create_market_overview_panel(token_data):
    """
    Create a market overview panel with all tokens in individual tabs on one line
    """
    # Create a table for horizontal layout
    market_table = Table(
        show_header=False,
        show_lines=False,
        box=box.ROUNDED,
        padding=(0, 1),
        expand=True,
    )

    # Add columns for each token
    for token in TOKEN_WATCHLIST.keys():
        market_table.add_column(justify="center", width=16)

    # Create token display cells
    token_cells = []
    for token in TOKEN_WATCHLIST.keys():
        if token in token_data and token_data[token]["status"] == "OK":
            data = token_data[token]
            change = data["change"]
            value_change = data["value_change"]
            current_price = data["current_price"]

            # Determine color and style based on performance
            if change > 0:
                color = "green"
                style = "bold green"
                trend = "▲"
            elif change < 0:
                color = "red"
                style = "bold red"
                trend = "▼"
            else:
                color = "white"
                style = "white"
                trend = "●"

            # Determine price color based on value (you can customize this logic)
            price_color = (
                "cyan"
                if current_price > 100
                else "yellow"
                if current_price > 10
                else "white"
            )

            # Create individual token tab with full symbol and both value & % change
            token_display = (
                f"{data['icon']} {data['symbol']}\n"
                f"[bold magenta]Price:[/bold magenta] [{price_color}]{current_price:.4f}[/{price_color}]\n"
                f"[{style}]{trend} {value_change:+.4f} ({change:+.2f}%)[/{style}]"
            )
            token_cells.append(token_display)
        else:
            status = (
                token_data.get(token, {}).get("status", "N/A")
                if token in token_data
                else "N/A"
            )
            token_cells.append(
                f"{TOKEN_WATCHLIST[token]['icon']} {token}???\n[white]{status}[/white]"
            )

    # Add single row with all tokens
    market_table.add_row(*token_cells)

    # Calculate market sentiment
    if token_data:
        positive_tokens = sum(
            1
            for data in token_data.values()
            if data.get("status") == "OK" and data["change"] > 0
        )
        total_tokens = len(
            [data for data in token_data.values() if data.get("status") == "OK"]
        )
        sentiment = (
            f"Market Sentiment: {positive_tokens}/{total_tokens} tokens positive"
        )

        # Add sentiment to title
        title = f"🔄 MARKET OVERVIEW • {sentiment} • 03:00 EAT Base • 📡 WebSocket"
    else:
        title = "🔄 MARKET OVERVIEW • 03:00 EAT Base • 📡 WebSocket"

    # Create centered panel without using Align on the table
    return Panel(
        market_table, title=title, title_align="center", style="cyan", padding=(1, 0)
    )


def analyze_ma_fib_relationship(df):
    """Analyze which MAs are below fib_0.236 while others are above"""
    if df.empty:
        return {"below_fib_236": "No data", "status": "N/A"}

    latest = df.iloc[-1]

    # Check if we have all required columns
    required_ma_columns = [
        "short002",
        "short007",
        "short21",
        "short50",
        "long100",
        "long200",
        "long350",
        "long500",
    ]
    required_fib_column = "level_236"

    missing_columns = [
        col
        for col in required_ma_columns + [required_fib_column]
        if col not in df.columns
    ]
    if missing_columns:
        return {"below_fib_236": f"Missing: {missing_columns}", "status": "N/A"}

    fib_236 = latest[required_fib_column]

    if pd.isna(fib_236):
        return {"below_fib_236": "Fib 23.6% not available", "status": "N/A"}

    # Check each MA against fib_236
    ma_status = {}
    for ma_col in required_ma_columns:
        ma_value = latest[ma_col]
        if pd.isna(ma_value):
            ma_status[ma_col] = "N/A"
        else:
            ma_status[ma_col] = "below" if ma_value < fib_236 else "above"

    # Find which MAs are below fib_236
    below_fib = [ma for ma, status in ma_status.items() if status == "below"]
    above_fib = [ma for ma, status in ma_status.items() if status == "above"]

    # Determine the special case: only one MA below while others above
    if len(below_fib) == 1 and len(above_fib) == len(required_ma_columns) - 1:
        status = f"🚨 {below_fib[0]} below Fib 23.6%"
    elif len(below_fib) > 0:
        status = f"⚠️ {len(below_fib)} MAs below Fib 23.6%: {', '.join(below_fib)}"
    else:
        status = "✅ All MAs above Fib 23.6%"

    return {
        "below_fib_236": below_fib,
        "above_fib_236": above_fib,
        "status": status,
        "fib_236_value": fib_236,
    }


def save_indicators_to_csv(df, filename="indicators_output.csv"):
    """Save calculated indicators to CSV file"""
    try:
        # Select only the calculated indicators and essential OHLC data
        columns_to_save = [
            "Open Time",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "daily_diff",
            "rsi",
            "rsi_ma50",
            "short002",
            "short007",
            "short21",
            "short50",
            "long100",
            "long200",
            "long350",
            "long500",
            "level_100",
            "level_764",
            "level_618",
            "level_500",
            "level_382",
            "level_236",
            "level_000",
        ]

        # Filter to only include columns that exist in the DataFrame
        available_columns = [col for col in columns_to_save if col in df.columns]

        # Save to CSV
        df[available_columns].to_csv(filename, index=False)
        logger.info(f"Indicators saved to {filename}")
        logger.info(f"Saved {len(df)} rows with {len(available_columns)} columns")

        return filename

    except Exception as e:
        logger.error(f"Error saving indicators to CSV: {e}")
        raise


def load_range_bars_only(input_filename):
    """Load range bar data without calculating indicators"""
    try:
        df = pd.read_csv(input_filename)

        if "Open Time" in df.columns:
            try:
                df["Open Time"] = pd.to_datetime(df["Open Time"], format="mixed")
            except:
                try:
                    df["Open Time"] = pd.to_datetime(
                        df["Open Time"], infer_datetime_format=True
                    )
                except:
                    df["Open Time"] = pd.to_datetime(df["Open Time"])

        # Ensure numeric columns
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    except Exception as e:
        logger.error(f"Error loading range bars: {e}")
        return pd.DataFrame()


def get_rsi_color_and_emoji(rsi_value):
    """Get color and emoji for RSI value based on common trading levels"""
    if pd.isna(rsi_value):
        return "white", "⚪", "N/A"
    elif rsi_value >= 75:
        return "red", "🔥", "Overbought+"
    elif rsi_value >= 65:
        return "magenta", "🔴", "Overbought"
    elif rsi_value >= 55:
        return "yellow", "🟡", "Bullish"
    elif rsi_value >= 45:
        return "green", "🟢", "Neutral"
    elif rsi_value >= 35:
        return "cyan", "🔵", "Bearish"
    elif rsi_value >= 25:
        return "blue", "🟦", "Oversold"
    else:
        return "bright_blue", "❄️", "Oversold+"


def display_rich_indicators_table(
    summary_df, rows=15, update_count=None, token_data=None, clear_first=False
):
    """Display a rich tabulated view of indicators for the last N rows"""

    # Clear screen instantly if requested (coin flip heads)
    if clear_first:
        clear_screen()

    if summary_df is None or summary_df.empty:
        console.print(
            Panel(
                "❌ No data available for indicators table",
                style="red",
                title="Error",
                title_align="center",
            )
        )
        return

    # Get last N rows
    display_data = summary_df.tail(rows).copy()

    if display_data.empty:
        console.print(
            Panel(
                "⚠️ No data to display in indicators table",
                style="yellow",
                title="Warning",
                title_align="center",
            )
        )
        return

    # Create main indicators table
    main_table = Table(
        title=f"📊 COMPLETE INDICATORS TABLE - LAST {len(display_data)} BARS",
        title_style="bold magenta",
        box=box.DOUBLE_EDGE,
        header_style="bold cyan",
        show_header=True,
        show_lines=True,
    )

    # Define columns for display - INCLUDING ALL MAs
    columns_config = [
        ("Open Time", "Time", "left"),
        ("Close", "Close", "right"),
        ("daily_diff", "Daily Diff %", "center"),
        ("rsi", "RSI", "right"),
        ("rsi_ma50", "RSI MA50", "right"),
        ("short002", "MA2", "right"),
        ("short007", "MA7", "right"),
        ("short21", "MA14", "right"),
        ("short50", "MA50", "right"),
        ("long100", "MA100", "right"),
        ("long200", "MA200", "right"),
        ("long350", "MA350", "right"),
        ("long500", "MA500", "right"),
        ("level_100", "Fib 100%", "right"),
        ("level_764", "Fib 76.4%", "right"),
        ("level_618", "Fib 61.8%", "right"),
        ("level_500", "Fib 50%", "right"),
        ("level_382", "Fib 38.2%", "right"),
        ("level_236", "Fib 23.6%", "right"),
        ("level_000", "Fib 0%", "right"),
    ]

    # Add columns to table
    for col_name, display_name, justify in columns_config:
        if col_name in display_data.columns:
            main_table.add_column(display_name, justify=justify, style="white")

    # Add rows to table
    for _, row in display_data.iterrows():
        row_values = []
        for col_name, display_name, justify in columns_config:
            if col_name not in display_data.columns:
                continue

            value = row.get(col_name, np.nan)

            if col_name == "Open Time":
                if pd.notna(value):
                    if isinstance(value, str):
                        formatted_value = value[11:19]  # Extract HH:MM:SS
                    else:
                        formatted_value = value.strftime("%H:%M:%S")
                else:
                    formatted_value = "N/A"
                style = "white"
            elif col_name == "daily_diff":
                formatted_value = value if pd.notna(value) else "N/A"
                if isinstance(formatted_value, str) and "+" in formatted_value:
                    style = "green"
                elif isinstance(formatted_value, str) and "-" in formatted_value:
                    style = "red"
                else:
                    style = "white"
            elif col_name == "rsi":
                if pd.notna(value):
                    formatted_value = f"{value:.6f}"  # Full precision for RSI
                    color, emoji, _ = get_rsi_color_and_emoji(value)
                    style = color
                else:
                    formatted_value = "N/A"
                    style = "white"
            elif col_name == "rsi_ma50":
                if pd.notna(value):
                    formatted_value = f"{value:.6f}"  # Full precision for RSI MA50
                    style = "yellow"
                else:
                    formatted_value = "N/A"
                    style = "white"
            elif col_name in [
                "Close",
                "short002",
                "short007",
                "short21",
                "short50",
                "long100",
                "long200",
                "long350",
                "long500",
            ]:
                if pd.notna(value):
                    # Full precision formatting - NO ROUNDING
                    formatted_value = f"{value:.8f}".rstrip("0").rstrip(".")
                    if formatted_value == "":
                        formatted_value = "0"
                    # Color code based on price position for MAs
                    if col_name == "Close":
                        style = "bold white"
                    else:
                        close_price = row.get("Close", np.nan)
                        if pd.notna(close_price):
                            if close_price > value:
                                style = "green"
                            else:
                                style = "red"
                        else:
                            style = "white"
                else:
                    formatted_value = "N/A"
                    style = "white"
            elif col_name.startswith("level_"):
                if pd.notna(value):
                    # Full precision formatting - NO ROUNDING
                    formatted_value = f"{value:.8f}".rstrip("0").rstrip(".")
                    if formatted_value == "":
                        formatted_value = "0"
                    # Color code Fibonacci levels
                    close_price = row.get("Close", np.nan)
                    if pd.notna(close_price):
                        if close_price >= value:
                            style = "green"
                        else:
                            style = "red"
                    else:
                        style = "white"
                else:
                    formatted_value = "N/A"
                    style = "white"
            else:
                if pd.notna(value) and isinstance(value, (int, float)):
                    formatted_value = f"{value:.6f}"  # Full precision
                else:
                    formatted_value = "N/A"
                style = "white"

            row_values.append((formatted_value, style))

        # Add row to table
        main_table.add_row(*[f"[{style}]{val}[/{style}]" for val, style in row_values])

    # Display main table
    console.print(main_table)

    # Create horizontally arranged quick stats table with MA-Fib analysis
    latest_row = display_data.iloc[-1]

    stats_table = Table(
        title="📊 QUICK STATS",
        title_style="bold green",
        box=box.ROUNDED,
        header_style="bold green",
        show_header=False,
        show_lines=False,
        expand=True,
    )

    # Add columns for horizontal layout
    stats_table.add_column("Metric 1", style="cyan", justify="center")
    stats_table.add_column("Value 1", style="white", justify="center")
    stats_table.add_column("Metric 2", style="cyan", justify="center")
    stats_table.add_column("Value 2", style="white", justify="center")
    stats_table.add_column("Metric 3", style="cyan", justify="center")
    stats_table.add_column("Value 3", style="white", justify="center")

    # Calculate running time
    running_time = format_running_time(SCRIPT_START_TIME)

    # Analyze MA-Fib relationship
    ma_fib_analysis = analyze_ma_fib_relationship(summary_df)

    # Create horizontal row with all stats
    stats_table.add_row(
        "Total Bars",
        str(len(summary_df)),
        "Update #",
        f"{update_count}" if update_count is not None else "Single Run",
        "Running Time",
        f"⏱️ {running_time}",
    )

    # Create MA-Fib analysis panel with centered content
    ma_fib_text = Text()
    ma_fib_text.append(ma_fib_analysis["status"], style="bold yellow")

    ma_fib_panel = Panel(
        Align.center(ma_fib_text),  # Center the content
        style="magenta",
        box=box.DOUBLE,
        padding=(1, 2),
        title="MA vs Fibonacci 23.6% Analysis",
        title_align="center",
    )

    # Display stats in panels
    console.print(
        Panel(
            stats_table,
            style="green",
            title="Statistics",
            title_align="center",
        )
    )

    # Display the MA-Fib analysis with the cute style and centered content
    console.print(ma_fib_panel)

    # Display Market Overview
    if token_data:
        market_panel = create_market_overview_panel(token_data)
        console.print(market_panel)
    else:
        # Fallback if no token data available
        console.print(
            Panel(
                "🔄 Market data unavailable - check Binance connection",
                style="yellow",
                title="Market Overview",
                title_align="center",
            )
        )


def format_running_time(start_time):
    """Format running time into human readable format"""
    if start_time is None:
        return "0s"

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def check_file_updated(filepath, last_modified_time):
    """Check if file has been updated since last check"""
    try:
        if not os.path.exists(filepath):
            return False, last_modified_time

        current_modified_time = os.path.getmtime(filepath)

        if last_modified_time is None or current_modified_time > last_modified_time:
            return True, current_modified_time
        else:
            return False, last_modified_time

    except Exception as e:
        logger.error(f"Error checking file modification time: {e}")
        return False, last_modified_time


def continuous_indicator_calculator():
    """Continuously monitor and update indicators using stateful engine and WebSocket data"""
    global SCRIPT_START_TIME, LAST_UPDATE_TIME, SYMBOL_DECIMAL_PRECISION

    INPUT_FILE = "historic_df_alpha.csv"
    OUTPUT_FILE = "pinescript_indicators.csv"

    if os.path.exists(OUTPUT_FILE):
        unique_id = str(uuid.uuid4())
        archive_name = f"pinescript_indicators_{unique_id}.csv"
        os.rename(OUTPUT_FILE, archive_name)
        logger.info(f"Archived existing indicators file to {archive_name}")

    last_modified_time = None
    update_count = 0

    # Initialize Binance client
    try:
        client = Client()
        logger.info("✅ Binance client initialized")
    except Exception as e:
        console.print(f"❌ Failed to initialize Binance client: {e}", style="red")
        return

    # Initialize stateful engine with client
    stateful_engine = StatefulIndicatorEngine(client)
    engine_initialized = False

    # Initialize WebSocket market data
    websocket_started = market_data_manager.start_websocket()

    # Initialize Binance client for 03:00 prices (one-time) and symbol precision
    try:
        # Detect symbol decimal precision (though we don't round anymore)
        SYMBOL_DECIMAL_PRECISION = get_symbol_info(client, SYMBOL)
        console.print(
            f"🎯 Using maximum precision for {SYMBOL} (no rounding)", style="bold green"
        )

        market_data_manager.initialize_0300_prices(client)
        logger.info("03:00 price cache initialized")
        
        # Get SOLFDUSD cache for daily difference
        sol_cache = market_data_manager.price_0300_cache.get("SOL")
        if sol_cache:
            stateful_engine.update_solfdusd_cache(sol_cache)
            logger.info(f"✅ SOLFDUSD cache initialized with date: {sol_cache.get('cache_date')}")
            
    except Exception as e:
        logger.warning(f"Could not initialize 03:00 prices: {e}")

    # Set script start time
    SCRIPT_START_TIME = time.time()
    LAST_UPDATE_TIME = SCRIPT_START_TIME

    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        console.print(
            Panel(
                f"❌ Input file {INPUT_FILE} not found!",
                style="red",
                title="Error",
                title_align="center",
            )
        )
        market_data_manager.stop()
        return

    # Display startup banner
    startup_info = Table.grid(padding=1)
    startup_info.add_column(style="green", justify="center")
    startup_info.add_row("🚀 STATEFUL + WEBSOCKET Indicator Calculator 🚀")
    startup_info.add_row("")
    startup_info.add_row(f"📁 Input: {INPUT_FILE}")
    startup_info.add_row(f"💾 Output: {OUTPUT_FILE}")
    startup_info.add_row(f"⏱️  Update Interval: {UPDATE_INTERVAL} seconds")
    startup_info.add_row(f"📊 Display Rows: {DISPLAY_ROWS}")
    startup_info.add_row(f"🎯 Precision: MAXIMUM (No Rounding)")
    startup_info.add_row(f"📡 WebSocket: {'✅ Connected' if websocket_started else '❌ Failed'}")
    startup_info.add_row(f"🕒 Daily Diff: SOLFDUSD (1-second klines)")
    startup_info.add_row("")
    startup_info.add_row("🔄 Press Ctrl+C to stop")

    console.print(
        Panel(
            startup_info,
            style="green",
            box=box.DOUBLE,
            padding=(1, 2),
        )
    )

    try:
        while True:
            # Check if we need to refresh 03:00 cache (daily reset)
            if client and market_data_manager.should_refresh_0300_cache():
                logger.info("🔄 Refreshing 03:00 cache due to date/time change")
                market_data_manager.initialize_0300_prices(client)
                # Update SOLFDUSD cache in engine
                sol_cache = market_data_manager.price_0300_cache.get("SOL")
                if sol_cache:
                    stateful_engine.update_solfdusd_cache(sol_cache)
                    logger.info(f"🔄 Updated SOLFDUSD cache with date: {sol_cache.get('cache_date')}")

            # Check if file has been updated
            is_updated, last_modified_time = check_file_updated(
                INPUT_FILE, last_modified_time
            )

            if is_updated or not engine_initialized:
                update_count += 1
                LAST_UPDATE_TIME = time.time()

                try:
                    # Load the data
                    start_time = time.time()
                    df = load_range_bars_only(INPUT_FILE)

                    if df.empty:
                        continue

                    # Initialize or update stateful engine
                    if not engine_initialized:
                        engine_initialized = stateful_engine.initialize_from_history(df)
                        processing_time = time.time() - start_time
                        
                        if engine_initialized:
                            # Save initial results
                            df_with_indicators = stateful_engine.get_current_data()
                            save_indicators_to_csv(df_with_indicators, OUTPUT_FILE)
                            
                            # Get token performance data from WebSocket
                            token_data = market_data_manager.get_token_performance()

                            # Display initial table
                            display_rich_indicators_table(
                                df_with_indicators,
                                DISPLAY_ROWS,
                                update_count,
                                token_data,
                                clear_first=True,
                            )
                            
                            console.print(
                                Panel(
                                    f"✅ Stateful engine initialized with {len(df)} historical bars in {processing_time:.2f}s",
                                    style="green",
                                    title="Initialization Complete",
                                    title_align="center",
                                )
                            )
                        else:
                            console.print(
                                Panel(
                                    "❌ Failed to initialize stateful engine",
                                    style="red",
                                    title="Initialization Failed",
                                    title_align="center",
                                )
                            )
                    else:
                        # Process only new data incrementally
                        last_time = stateful_engine.last_processed_time
                        if last_time:
                            new_data = df[df["Open Time"] > last_time]
                        else:
                            new_data = df

                        if not new_data.empty:
                            success = stateful_engine.process_new_data(new_data)
                            processing_time = time.time() - start_time

                            if success:
                                # Get updated data
                                df_with_indicators = stateful_engine.get_current_data()

                                # Save results
                                save_indicators_to_csv(df_with_indicators, OUTPUT_FILE)

                                # Get token performance data from WebSocket
                                token_data = market_data_manager.get_token_performance()

                                # Coin flip for screen clearing
                                clear_screen_flag = should_clear_screen()
                                if clear_screen_flag:
                                    logger.info(
                                        f"🎯 Update #{update_count}: Coin flip - HEADS (screen cleared)"
                                    )
                                else:
                                    logger.info(
                                        f"🎯 Update #{update_count}: Coin flip - TAILS (screen not cleared)"
                                    )

                                # Display rich table with optional screen clearing
                                display_rich_indicators_table(
                                    df_with_indicators,
                                    DISPLAY_ROWS,
                                    update_count,
                                    token_data,
                                    clear_first=clear_screen_flag,
                                )

                                logger.info(
                                    f"Incremental update #{update_count} completed in {processing_time:.2f}s | New rows: {len(new_data)}"
                                )

                except Exception as e:
                    error_msg = f"❌ Error during update #{update_count}: {e}"
                    console.print(
                        Panel(
                            error_msg,
                            style="red",
                            title="Error",
                            title_align="center",
                        )
                    )
                    logger.error(error_msg)

            else:
                # File not updated, show waiting message
                if update_count == 0 or (time.time() % 30 < UPDATE_INTERVAL):
                    running_time = format_running_time(SCRIPT_START_TIME)
                    status = (
                        "Initializing..."
                        if not engine_initialized
                        else "Waiting for updates..."
                    )
                    ws_status = "✅" if market_data_manager.connected else "❌"
                    
                    # Show current cache status
                    sol_cache = market_data_manager.price_0300_cache.get("SOL")
                    cache_status = "No cache"
                    if sol_cache and "cache_date" in sol_cache:
                        cache_status = f"Cache date: {sol_cache['cache_date']}"
                    
                    console.print(
                        f"[yellow]⏳ {status} (Update #{update_count}) | WebSocket: {ws_status} | {cache_status} | Running: {running_time}[/yellow]"
                    )

            time.sleep(UPDATE_INTERVAL)

    except KeyboardInterrupt:
        running_time = format_running_time(SCRIPT_START_TIME)
        console.print(
            Panel(
                f"🛑 Stopping calculator...\n⏱️ Total running time: {running_time}",
                style="yellow",
                title="Info",
                title_align="center",
            )
        )
    except Exception as e:
        running_time = format_running_time(SCRIPT_START_TIME)
        error_msg = f"❌ Critical error: {e}"
        console.print(
            Panel(
                error_msg,
                style="red",
                title="Critical Error",
                title_align="center",
            )
        )
    finally:
        market_data_manager.stop()


def configure_settings():
    """Allow user to configure update interval and display rows"""
    global UPDATE_INTERVAL, DISPLAY_ROWS

    settings_table = Table(
        title="⚙️ CURRENT CONFIGURATION SETTINGS ⚙️",
        title_style="bold blue",
        box=box.ROUNDED,
        header_style="bold blue",
    )

    settings_table.add_column("Setting", style="cyan")
    settings_table.add_column("Current Value", style="green")

    settings_table.add_row("📍 Update Interval", f"{UPDATE_INTERVAL} seconds")
    settings_table.add_row("📊 Display Rows", f"{DISPLAY_ROWS}")

    console.print(
        Panel(
            settings_table,
            style="blue",
            title="Configuration",
            title_align="center",
        )
    )

    try:
        # Update interval
        new_interval = Prompt.ask(
            f"🕐 Enter new update interval in seconds", default=str(UPDATE_INTERVAL)
        )
        if new_interval and new_interval.replace(".", "").isdigit():
            UPDATE_INTERVAL = float(new_interval)
            console.print(
                f"✅ [green]Update interval set to {UPDATE_INTERVAL} seconds[/green]"
            )

        # Display rows
        new_rows = Prompt.ask(
            f"📊 Enter number of rows to display", default=str(DISPLAY_ROWS)
        )
        if new_rows and new_rows.isdigit():
            DISPLAY_ROWS = int(new_rows)
            console.print(f"✅ [green]Display rows set to {DISPLAY_ROWS}[/green]")

        console.print(
            Panel(
                "🎯 Configuration updated successfully!",
                style="green",
                title="Success",
                title_align="center",
            )
        )

    except Exception as e:
        console.print(
            Panel(
                f"❌ Configuration error: {e}",
                style="red",
                title="Error",
                title_align="center",
            )
        )


def get_choice_with_timeout():
    def input_thread():
        try:
            choice = console.input(
                f"\n[cyan]Enter your choice (1/2/3): [/cyan]"
            ).strip()
            q.put(choice)
        except:
            pass

    q = queue.Queue()
    t = threading.Thread(target=input_thread)
    t.daemon = True
    t.start()
    try:
        choice = q.get(timeout=timeout_que)
    except queue.Empty:
        console.print(f"\n[yellow]Timeout, defaulting to continuous mode (1)[/yellow]")
        choice = "1"
    return choice


def main():
    """Main function with enhanced menu system"""
    global SCRIPT_START_TIME

    # Display welcome banner
    banner_text = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║           🚀 STATEFUL + WEBSOCKET INDICATOR CALCULATOR 🚀                   ║
║                                                                              ║
║         Real-time Performance • No Rate Limits • Instant Updates            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    console.print(
        Panel(
            Align.center(banner_text),
            style="bold magenta",
            box=box.DOUBLE,
        )
    )

    # Display menu options
    menu_table = Table(show_header=False, box=box.ROUNDED, show_lines=False, width=80)

    menu_table.add_column(justify="center")

    menu_options = [
        "[1] 🔄 Continuous Mode - Real-time monitoring with WebSocket data",
        "[2] 📊 Single Run - One-time calculation and display",
        "[3] ⚙️  Settings - Configure update interval and display options",
    ]

    for option in menu_options:
        menu_table.add_row(option)

    console.print(
        Panel(
            menu_table,
            style="green",
            title="Choose an Option",
            title_align="center",
        )
    )

    # Get user choice with timeout
    choice = get_choice_with_timeout()

    if choice == "1":
        # Continuous mode
        console.print(
            Panel(
                "🔄 Starting Continuous Mode with WebSocket Data...",
                style="bold cyan",
                box=box.DOUBLE,
            )
        )
        continuous_indicator_calculator()

    elif choice == "2":
        # Single run mode
        console.print(
            Panel(
                "📊 Starting Single Run Analysis...",
                style="bold cyan",
                box=box.DOUBLE,
            )
        )
        try:
            # For single run, use the traditional approach
            client = Client()

            # Detect symbol decimal precision
            global SYMBOL_DECIMAL_PRECISION
            SYMBOL_DECIMAL_PRECISION = get_symbol_info(client, SYMBOL)
            console.print(
                f"🎯 Using maximum precision for {SYMBOL} (no rounding)",
                style="bold green",
            )

            # Load and process data
            df = load_range_bars_only("historic_df_alpha.csv")
            if not df.empty:
                # Use stateful engine for single run too
                stateful_engine = StatefulIndicatorEngine(client)
                success = stateful_engine.initialize_from_history(df)
                
                if success:
                    df_with_indicators = stateful_engine.get_current_data()
                    save_indicators_to_csv(df_with_indicators, "pinescript_indicators.csv")

                    # Get token data using traditional method for single run
                    token_data = {}
                    try:
                        for token, token_info in TOKEN_WATCHLIST.items():
                            price_0300 = None
                            used_pairing = None
                            current_price = 0.0

                            # Try each pairing in order of preference
                            for pairing in token_info["pairings"]:
                                symbol = f"{token}{pairing}"
                                try:
                                    price_0300, cache_date = get_price_at_kenyan_0300_1sec(client, symbol)
                                    if price_0300 is not None:
                                        ticker = client.get_symbol_ticker(symbol=symbol)
                                        current_price = float(ticker["price"])
                                        used_pairing = pairing
                                        break
                                except:
                                    continue

                            if price_0300 is not None:
                                diff = current_price - price_0300
                                percent_change = (diff / price_0300) * 100
                                value_change = diff

                                token_data[token] = {
                                    "icon": token_info["icon"],
                                    "symbol": f"{token}{used_pairing}",
                                    "change": percent_change,
                                    "value_change": value_change,
                                    "current_price": current_price,
                                    "price_0300": price_0300,
                                    "pairing": used_pairing,
                                    "status": "OK",
                                }
                            else:
                                token_data[token] = {
                                    "icon": token_info["icon"],
                                    "symbol": f"{token}???",
                                    "change": 0.0,
                                    "value_change": 0.0,
                                    "current_price": 0.0,
                                    "status": "N/A",
                                }
                    except Exception as e:
                        logger.warning(f"Could not fetch token data for single run: {e}")

                    # Coin flip for screen clearing in single run
                    clear_screen_flag = should_clear_screen()
                    if clear_screen_flag:
                        logger.info("🎯 Single Run: Coin flip - HEADS (screen cleared)")
                    else:
                        logger.info("🎯 Single Run: Coin flip - TAILS (screen not cleared)")

                    display_rich_indicators_table(
                        df_with_indicators,
                        DISPLAY_ROWS,
                        token_data=token_data,
                        clear_first=clear_screen_flag,
                    )
                else:
                    console.print(
                        Panel(
                            "❌ Failed to initialize indicators",
                            style="red",
                            title="Error",
                            title_align="center",
                        )
                    )
        except Exception as e:
            console.print(
                Panel(
                    f"❌ Error during single run: {e}",
                    style="red",
                    title="Error",
                    title_align="center",
                )
            )

    elif choice == "3":
        # Settings configuration
        configure_settings()
        console.print(
            "\n[yellow]Returning to main menu...[/yellow]",
            style="bold yellow",
        )
        time.sleep(2)
        main()

    else:
        console.print(
            Panel(
                "❌ Invalid choice! Please select 1, 2, or 3.",
                style="red",
                title="Error",
                title_align="center",
            )
        )
        time.sleep(2)
        main()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print(
            "\n[yellow]👋 Script terminated by user. Goodbye![/yellow]",
            style="bold yellow",
        )
        market_data_manager.stop()
    except Exception as e:
        console.print(
            Panel(
                f"❌ Unexpected error: {e}",
                style="red",
                title="Critical Error",
                title_align="center",
            )
        )
        logger.exception("Unexpected error in main execution")
        market_data_manager.stop()