import logging
import os
import queue
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
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

# Token watchlist with icons
TOKEN_WATCHLIST = {
    "BTC": "₿",
    "ETH": "🔷", 
    "BNB": "⚡",
    "DOGE": "🐕",
    "XRP": "✖️",
    "LINK": "🔗",
    "PAXG": "🏆",
    "ASTER": "🌟"
}

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class PinescriptRSICalculator:
    """
    Exact Pinescript RSI implementation with Wilder's smoothing
    Matches TradingView's ta.rsi() calculation precisely
    """

    def __init__(self, length=14):
        self.length = length
        self.alpha = 1.0 / length  # Wilder's smoothing alpha

    def calculate_rsi(self, prices):
        """
        Calculate RSI using Wilder's smoothing method (matches TradingView exactly)

        Args:
            prices: Series of close prices

        Returns:
            Series of RSI values
        """
        if len(prices) < self.length + 1:
            return pd.Series([np.nan] * len(prices), index=prices.index)

        # Calculate price changes
        changes = prices.diff()

        # Separate gains and losses
        gains = changes.where(changes >= 0, 0.0)
        losses = -changes.where(changes < 0, 0.0)  # Make losses positive

        # Initialize RSI series
        rsi = pd.Series(index=prices.index, dtype=float)

        # Calculate first RSI value using SMA
        first_avg_gain = gains.iloc[1 : self.length + 1].mean()
        first_avg_loss = losses.iloc[1 : self.length + 1].mean()

        if first_avg_loss == 0:
            rsi.iloc[self.length] = 100.0
        elif first_avg_gain == 0:
            rsi.iloc[self.length] = 0.0
        else:
            rs = first_avg_gain / first_avg_loss
            rsi.iloc[self.length] = 100 - (100 / (1 + rs))

        # Calculate subsequent RSI values using Wilder's smoothing
        avg_gain = first_avg_gain
        avg_loss = first_avg_loss

        for i in range(self.length + 1, len(prices)):
            # Wilder's smoothing formula
            avg_gain = (avg_gain * (self.length - 1) + gains.iloc[i]) / self.length
            avg_loss = (avg_loss * (self.length - 1) + losses.iloc[i]) / self.length

            if avg_loss == 0:
                rsi.iloc[i] = 100.0
            elif avg_gain == 0:
                rsi.iloc[i] = 0.0
            else:
                rs = avg_gain / avg_loss
                rsi.iloc[i] = 100 - (100 / (1 + rs))

        return rsi


class MovingAverageCalculator:
    """Simple Moving Average calculator matching TradingView's ta.sma()"""

    @staticmethod
    def sma(data, length):
        """Calculate Simple Moving Average"""
        return data.rolling(window=length, min_periods=length).mean()


class FibonacciLevels:
    """Calculate Fibonacci retracement levels"""

    def __init__(self, length=1500):
        self.length = length

    def calculate_levels(self, prices):
        """
        Calculate Fibonacci levels over specified period
        Returns NaN for all levels if insufficient data available

        Args:
            prices: Series of prices to calculate Fibonacci levels from

        Returns:
            Dictionary of Fibonacci levels (returns NaN if insufficient data)
        """
        # Return NaN for all levels if insufficient data
        if len(prices) < self.length:
            return {
                "level_100": np.nan,
                "level_764": np.nan,
                "level_618": np.nan,
                "level_500": np.nan,
                "level_382": np.nan,
                "level_236": np.nan,
                "level_000": np.nan,
            }

        # Calculate with proper rolling window when sufficient data exists
        high = (
            prices.rolling(window=self.length, min_periods=self.length).max().iloc[-1]
        )
        low = prices.rolling(window=self.length, min_periods=self.length).min().iloc[-1]

        range_val = high - low

        levels = {
            "level_100": high,
            "level_764": high - 0.236 * range_val,  # 76.4% retracement
            "level_618": high - 0.382 * range_val,  # 61.8% retracement
            "level_500": high - 0.50 * range_val,  # 50% retracement
            "level_382": low + 0.382 * range_val,  # 38.2% retracement
            "level_236": low + 0.236 * range_val,  # 23.6% retracement
            "level_000": low,
        }

        return levels


def get_token_performance(client):
    """
    Get 24h performance for watchlist tokens
    
    Args:
        client: Binance client instance
        
    Returns:
        Dictionary with token performance data
    """
    token_data = {}
    
    try:
        # Get 24hr ticker for all symbols
        tickers = client.get_ticker()
        
        for token, icon in TOKEN_WATCHLIST.items():
            symbol = f"{token}USDT"
            
            # Find the ticker for this symbol
            ticker_data = next((t for t in tickers if t['symbol'] == symbol), None)
            
            if ticker_data:
                price_change_percent = float(ticker_data['priceChangePercent'])
                token_data[token] = {
                    'icon': icon,
                    'change': price_change_percent,
                    'current_price': float(ticker_data['lastPrice'])
                }
            else:
                token_data[token] = {
                    'icon': icon,
                    'change': 0.0,
                    'current_price': 0.0
                }
                
    except Exception as e:
        logger.error(f"Error fetching token performance: {e}")
        # Return empty data on error
        for token, icon in TOKEN_WATCHLIST.items():
            token_data[token] = {
                'icon': icon,
                'change': 0.0,
                'current_price': 0.0
            }
    
    return token_data


def create_market_overview_panel(token_data):
    """
    Create a market overview panel with all tokens in individual tabs on one line
    
    Args:
        token_data: Dictionary with token performance data
        
    Returns:
        Rich Panel object
    """
    # Create a table for horizontal layout
    market_table = Table(
        show_header=False,
        show_lines=False,
        box=box.ROUNDED,
        padding=(0, 1),
        expand=True
    )
    
    # Add columns for each token
    for token in TOKEN_WATCHLIST.keys():
        market_table.add_column(justify="center", width=12)
    
    # Create token display cells
    token_cells = []
    for token in TOKEN_WATCHLIST.keys():
        if token in token_data:
            data = token_data[token]
            change = data['change']
            
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
            
            # Create individual token tab
            token_display = f"{data['icon']} {token}\n[{style}]{trend} {change:+.2f}%[/{style}]"
            token_cells.append(token_display)
        else:
            token_cells.append(f"{TOKEN_WATCHLIST[token]} {token}\n[white]N/A[/white]")
    
    # Add single row with all tokens
    market_table.add_row(*token_cells)
    
    # Calculate market sentiment
    if token_data:
        positive_tokens = sum(1 for data in token_data.values() if data['change'] > 0)
        total_tokens = len(token_data)
        sentiment = f"Market Sentiment: {positive_tokens}/{total_tokens} tokens positive"
        
        # Add sentiment to title
        title = f"🔄 MARKET OVERVIEW • {sentiment}"
    else:
        title = "🔄 MARKET OVERVIEW"
    
    return Panel(
        market_table,
        title=title,
        title_align="center",
        style="cyan",
        padding=(1, 0)
    )


def calculate_all_indicators(df, rsi_length=14, rsi_source="Close"):
    """
    Calculate all indicators from the Pinescript code

    Args:
        df: DataFrame with OHLCV data
        rsi_length: RSI calculation period
        rsi_source: Column to use for RSI calculation

    Returns:
        DataFrame with all calculated indicators
    """

    # Ensure we have required columns
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Create a copy to avoid modifying original data
    result_df = df.copy()

    # Initialize calculators
    rsi_calc = PinescriptRSICalculator(length=rsi_length)

    # Calculate RSI
    logger.info(f"Calculating RSI with length {rsi_length}")
    result_df["rsi"] = rsi_calc.calculate_rsi(result_df[rsi_source])

    # Calculate RSI MA50
    logger.info("Calculating RSI MA50")
    result_df["rsi_ma50"] = MovingAverageCalculator.sma(result_df["rsi"], 35)

    # Calculate Moving Averages (from Pinescript)
    logger.info("Calculating moving averages")
    result_df["short002"] = MovingAverageCalculator.sma(result_df["Close"], 2)
    result_df["short007"] = MovingAverageCalculator.sma(result_df["Close"], 7)
    result_df["short21"] = MovingAverageCalculator.sma(
        result_df["Close"], 14
    )  # Note: Pinescript shows 14, not 21
    result_df["short50"] = MovingAverageCalculator.sma(result_df["Close"], 50)
    result_df["long100"] = MovingAverageCalculator.sma(result_df["Close"], 100)
    result_df["long200"] = MovingAverageCalculator.sma(result_df["Close"], 200)
    result_df["long350"] = MovingAverageCalculator.sma(result_df["Close"], 350)
    result_df["long500"] = MovingAverageCalculator.sma(result_df["Close"], 500)

    # Calculate Fibonacci Levels (dynamic for each row)
    logger.info("Calculating Fibonacci levels")
    fib_calc = FibonacciLevels(length=1500)

    # For simplicity, calculate Fibonacci levels for the entire dataset
    # In practice, you might want to calculate these dynamically for each bar
    fib_levels = fib_calc.calculate_levels(result_df["Close"])

    for level_name, level_value in fib_levels.items():
        result_df[level_name] = level_value

    logger.info("All indicators calculated successfully")
    return result_df


def save_indicators_to_csv(df, filename="indicators_output.csv"):
    """
    Save calculated indicators to CSV file

    Args:
        df: DataFrame with calculated indicators
        filename: Output filename
    """
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

        # Log summary statistics for key indicators
        if "rsi" in df.columns:
            rsi_stats = df["rsi"].describe()
            logger.info(
                f"RSI Statistics: Min={rsi_stats['min']:.2f}, Max={rsi_stats['max']:.2f}, Mean={rsi_stats['mean']:.2f}"
            )

        return filename

    except Exception as e:
        logger.error(f"Error saving indicators to CSV: {e}")
        raise


def load_and_process_range_bars(input_filename="historic_df_alpha.csv"):
    """
    Load range bar data and calculate all Pinescript indicators

    Args:
        input_filename: Path to the range bar CSV file

    Returns:
        DataFrame with all calculated indicators
    """

    try:
        logger.info(f"Loading data from {input_filename}")

        # Load the range bar data
        df = pd.read_csv(input_filename)
        logger.info(f"Loaded {len(df)} bars from {input_filename}")

        # Fetch daily difference using price_monitor approach
        try:
            client = Client()
            utc_now = datetime.now(timezone.utc)
            target_time = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
            timestamp = int(target_time.timestamp() * 1000)

            klines = client.get_historical_klines(
                symbol=SYMBOL,
                interval=Client.KLINE_INTERVAL_1MINUTE,
                start_str=timestamp,
                end_str=timestamp + 60000,
                limit=1,
            )

            if klines:
                price_0300 = float(klines[0][4])
            else:
                price_0300 = None

            if price_0300 is None:
                df["daily_diff"] = "N/A"
                logger.warning("Could not retrieve 03:00 price.")
            else:
                ticker = client.get_symbol_ticker(symbol=SYMBOL)
                current_price = float(ticker["price"])
                diff = current_price - price_0300
                percent = (diff / price_0300) * 100

                # MODIFICATION: Add 0.08 to both positive AND negative daily difference
                if percent > 0:
                    percent += 0.08
                    formatted_change = f"+{percent:.2f}%"
                else:
                    # Add 0.08 to negative values as well
                    percent += 0.08
                    if percent >= 0:
                        formatted_change = f"+{percent:.2f}%"
                    else:
                        formatted_change = f"{percent:.2f}%"

                df["daily_diff"] = formatted_change
                logger.info(f"Daily diff fetched for {SYMBOL}: {formatted_change}")
        except Exception as e:
            logger.error(f"Error fetching daily diff: {e}")
            df["daily_diff"] = "N/A"

        # Enhanced datetime parsing to handle mixed formats
        if "Open Time" in df.columns:
            try:
                # First try with mixed format parsing
                df["Open Time"] = pd.to_datetime(df["Open Time"], format="mixed")
            except:
                try:
                    # Fallback to infer format
                    df["Open Time"] = pd.to_datetime(
                        df["Open Time"], infer_datetime_format=True
                    )
                except:
                    # Final fallback - let pandas auto-detect
                    df["Open Time"] = pd.to_datetime(df["Open Time"])

        if "Close Time" in df.columns:
            try:
                # First try with mixed format parsing
                df["Close Time"] = pd.to_datetime(df["Close Time"], format="mixed")
            except:
                try:
                    # Fallback to infer format
                    df["Close Time"] = pd.to_datetime(
                        df["Close Time"], infer_datetime_format=True
                    )
                except:
                    # Final fallback - let pandas auto-detect
                    df["Close Time"] = pd.to_datetime(df["Close Time"])

        # Verify we have enough data for calculations
        min_required_bars = 500  # Need at least 500 bars for long-term MAs
        if len(df) < min_required_bars:
            logger.warning(
                f"Only {len(df)} bars available. Some long-term indicators may not be calculated."
            )

        # Calculate all indicators
        logger.info("Starting indicator calculations...")
        df_with_indicators = calculate_all_indicators(df)

        logger.info("Indicator calculations completed")
        return df_with_indicators

    except Exception as e:
        logger.error(f"Error processing range bars: {e}")
        raise


def get_rsi_color_and_emoji(rsi_value):
    """
    Get color and emoji for RSI value based on common trading levels

    Args:
        rsi_value: RSI value (0-100)

    Returns:
        tuple: (color, emoji, description)
    """
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


def get_price_trend_emoji(current, previous):
    """Get emoji for price movement"""
    if pd.isna(current) or pd.isna(previous):
        return "⚪"
    elif current > previous:
        return "📈"
    elif current < previous:
        return "📉"
    else:
        return "➡️"


def get_ma_position_color(price, ma_value):
    """Get color based on price position relative to MA"""
    if pd.isna(ma_value) or pd.isna(price):
        return "white"
    elif price > ma_value:
        return "green"  # Price above MA (bullish)
    else:
        return "red"  # Price below MA (bearish)


def format_number(value, decimals=2):
    """Format number with appropriate decimal places"""
    if pd.isna(value):
        return "N/A".center(8)
    return f"{value:.{decimals}f}".center(8)


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


def display_rich_indicators_table(summary_df, rows=15, update_count=None, token_data=None):
    """
    Display a rich tabulated view of indicators for the last N rows

    Args:
        summary_df: DataFrame with calculated indicators
        rows: Number of rows to display
        update_count: Current update count for display
        token_data: Token performance data for market overview
    """
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
        ("short21", "MA14", "right"),  # Added MA14
        ("short50", "MA50", "right"),
        ("long100", "MA100", "right"),  # Added MA100
        ("long200", "MA200", "right"),
        ("long350", "MA350", "right"),  # Added MA350
        ("long500", "MA500", "right"),  # Added MA500
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
                    formatted_value = f"{value:.2f}"
                    color, emoji, _ = get_rsi_color_and_emoji(value)
                    style = color
                else:
                    formatted_value = "N/A"
                    style = "white"
            elif col_name == "rsi_ma50":
                if pd.notna(value):
                    formatted_value = f"{value:.2f}"
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
                    formatted_value = f"{value:.4f}"
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
                    formatted_value = f"{value:.4f}"
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
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = "N/A"
                style = "white"

            row_values.append((formatted_value, style))

        # Add row to table
        main_table.add_row(*[f"[{style}]{val}[/{style}]" for val, style in row_values])

    # Display main table
    console.print(main_table)

    # MODIFICATION: Create horizontally arranged quick stats table
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

    # Create horizontal row with all stats
    stats_table.add_row(
        "Total Bars",
        str(len(summary_df)),
        "Update #",
        f"{update_count}" if update_count is not None else "Single Run",
        "Running Time",
        f"⏱️ {running_time}",
    )

    # Display stats in a panel
    console.print(
        Panel(
            stats_table,
            style="green",
            title="Statistics",
            title_align="center",
        )
    )

    # REPLACEMENT: Display Market Overview instead of Processing Complete
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


def print_rich_update_header(update_count, total_bars):
    """Print rich update header"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header_text = Text()
    header_text.append("🔄 UPDATE #", style="bold white")
    header_text.append(f"{update_count}", style="bold yellow")
    header_text.append(" - ", style="bold white")
    header_text.append(f"{current_time}", style="bold cyan")
    header_text.append(" - Total Bars: ", style="bold white")
    header_text.append(f"{total_bars}", style="bold green")
    header_text.append(" 🔄", style="bold white")

    console.print(
        Panel(
            header_text,
            style="magenta",
            box=box.DOUBLE,
            padding=(1, 2),
        )
    )


def validate_rsi_calculation(df, sample_size=10):
    """
    Validate RSI calculation by showing sample values and checking ranges

    Args:
        df: DataFrame with calculated RSI
        sample_size: Number of sample values to display
    """

    if "rsi" not in df.columns:
        logger.error("RSI column not found in DataFrame")
        return

    # Remove NaN values for validation
    rsi_values = df["rsi"].dropna()

    if len(rsi_values) == 0:
        logger.error("No valid RSI values found")
        return

    # Create validation table
    validation_table = Table(
        title="RSI Validation Results",
        title_style="bold blue",
        box=box.ROUNDED,
        header_style="bold blue",
    )

    validation_table.add_column("Metric", style="cyan")
    validation_table.add_column("Value", style="white")

    validation_data = [
        ("Total RSI values calculated", str(len(rsi_values))),
        ("RSI range", f"{rsi_values.min():.4f} to {rsi_values.max():.4f}"),
        ("RSI mean", f"{rsi_values.mean():.4f}"),
        ("RSI std", f"{rsi_values.std():.4f}"),
        (
            "Range check",
            "✅ Within expected range (0-100)"
            if rsi_values.min() >= 0 and rsi_values.max() <= 100
            else "⚠️ Outside expected range",
        ),
    ]

    for metric, value in validation_data:
        validation_table.add_row(metric, value)

    console.print(
        Panel(
            validation_table,
            style="blue",
            title="RSI Validation",
            title_align="center",
        )
    )

    # Show sample values
    sample_table = Table(
        title=f"Sample RSI Values (last {sample_size})",
        title_style="bold yellow",
        box=box.SIMPLE,
        header_style="bold yellow",
    )

    sample_table.add_column("Bar Index", style="cyan")
    sample_table.add_column("Close Price", style="white")
    sample_table.add_column("RSI Value", style="green")

    for i, (idx, rsi_val) in enumerate(rsi_values.tail(sample_size).items()):
        close_price = df.loc[idx, "Close"] if "Close" in df.columns else "N/A"
        if close_price != "N/A":
            close_price = f"{close_price:.4f}"
        sample_table.add_row(str(idx), close_price, f"{rsi_val:.4f}")

    console.print(
        Panel(
            sample_table,
            style="yellow",
            title="Sample Data",
            title_align="center",
        )
    )


def check_file_updated(filepath, last_modified_time):
    """
    Check if file has been updated since last check

    Args:
        filepath: Path to file to check
        last_modified_time: Last known modification time

    Returns:
        tuple: (is_updated, new_modification_time)
    """
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
    """
    Continuously monitor and update indicators as the CSV file changes
    """
    global SCRIPT_START_TIME, LAST_UPDATE_TIME

    INPUT_FILE = "historic_df_alpha.csv"
    OUTPUT_FILE = "pinescript_indicators.csv"

    if os.path.exists(OUTPUT_FILE):
        unique_id = str(uuid.uuid4())
        archive_name = f"pinescript_indicators_{unique_id}.csv"
        os.rename(OUTPUT_FILE, archive_name)
        logger.info(f"Archived existing indicators file to {archive_name}")

    last_modified_time = None
    update_count = 0

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
        console.print(
            Panel(
                "Please ensure your range bar data is saved as 'historic_df_alpha.csv'",
                style="yellow",
                title="Info",
                title_align="center",
            )
        )
        return

    # Display startup banner
    startup_info = Table.grid(padding=1)
    startup_info.add_column(style="green", justify="center")
    startup_info.add_row("🚀 Starting Continuous Indicator Calculator 🚀")
    startup_info.add_row("")
    startup_info.add_row(f"📁 Input: {INPUT_FILE}")
    startup_info.add_row(f"💾 Output: {OUTPUT_FILE}")
    startup_info.add_row(f"⏱️  Update Interval: {UPDATE_INTERVAL} seconds")
    startup_info.add_row(f"📊 Display Rows: {DISPLAY_ROWS}")
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

    # Initialize Binance client for token data
    try:
        client = Client()
        logger.info("Binance client initialized for token data")
    except Exception as e:
        logger.warning(f"Could not initialize Binance client: {e}")
        client = None

    try:
        while True:
            # Check if file has been updated
            is_updated, last_modified_time = check_file_updated(
                INPUT_FILE, last_modified_time
            )

            if is_updated:
                update_count += 1
                LAST_UPDATE_TIME = time.time()

                try:
                    # Load and process the data
                    start_time = time.time()
                    df_with_indicators = load_and_process_range_bars(INPUT_FILE)
                    processing_time = time.time() - start_time

                    # Save results to CSV
                    save_indicators_to_csv(df_with_indicators, OUTPUT_FILE)

                    # Get token performance data
                    token_data = {}
                    if client:
                        try:
                            token_data = get_token_performance(client)
                        except Exception as e:
                            logger.error(f"Error getting token data: {e}")
                            token_data = {}

                    # Print update header
                    print_rich_update_header(update_count, len(df_with_indicators))

                    # Display rich table with token data
                    display_rich_indicators_table(
                        df_with_indicators, DISPLAY_ROWS, update_count, token_data
                    )

                    # Log processing stats (now in background only)
                    total_bars = len(df_with_indicators)
                    valid_rsi = df_with_indicators["rsi"].notna().sum()
                    valid_rsi_ma50 = df_with_indicators["rsi_ma50"].notna().sum()

                    logger.info(
                        f"Update #{update_count} completed in {processing_time:.2f}s | "
                        f"Bars: {total_bars} | RSI: {valid_rsi} | RSI MA50: {valid_rsi_ma50}"
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
                # File not updated, show waiting message every 30 seconds
                if update_count == 0 or (time.time() % 30 < UPDATE_INTERVAL):
                    running_time = format_running_time(SCRIPT_START_TIME)
                    console.print(
                        f"[yellow]⏳ Waiting for {INPUT_FILE} to update... (Update #{update_count + 1}) | Running: {running_time}[/yellow]"
                    )

            # Wait for next check
            time.sleep(UPDATE_INTERVAL)

    except KeyboardInterrupt:
        running_time = format_running_time(SCRIPT_START_TIME)
        console.print(
            Panel(
                f"🛑 Stopping continuous calculator...\n⏱️ Total running time: {running_time}",
                style="yellow",
                title="Info",
                title_align="center",
            )
        )
        logger.info(f"Continuous calculator stopped by user after {running_time}")
    except Exception as e:
        running_time = format_running_time(SCRIPT_START_TIME)
        error_msg = f"❌ Critical error in continuous calculator: {e}\n⏱️ Running time: {running_time}"
        console.print(
            Panel(
                error_msg,
                style="red",
                title="Critical Error",
                title_align="center",
            )
        )
        logger.error(error_msg)


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
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║                  🚀 PINESCRIPT INDICATOR CALCULATOR 🚀                      ║
    ║                                                                              ║
    ║           Advanced Technical Analysis with Real-time Updates                ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """

    console.print(Panel(banner, style="bold magenta", box=box.DOUBLE))

    # Display menu options
    menu_table = Table(
        title="📋 MAIN MENU OPTIONS",
        title_style="bold green",
        box=box.ROUNDED,
        header_style="bold green",
        show_header=False,
    )

    menu_table.add_column("Option", style="cyan", width=8)
    menu_table.add_column("Description", style="white")

    menu_options = [
        ("[1]", "🔄 Continuous Mode - Real-time monitoring with auto-updates"),
        ("[2]", "📊 Single Run - One-time calculation and display"),
        ("[3]", "⚙️  Settings - Configure update interval and display options"),
    ]

    for option, description in menu_options:
        menu_table.add_row(option, description)

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
                "🔄 Starting Continuous Mode...",
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
            df_with_indicators = load_and_process_range_bars("historic_df_alpha.csv")
            save_indicators_to_csv(df_with_indicators, "pinescript_indicators.csv")
            
            # Get token data for single run too
            token_data = {}
            try:
                client = Client()
                token_data = get_token_performance(client)
            except Exception as e:
                logger.warning(f"Could not fetch token data for single run: {e}")
            
            display_rich_indicators_table(df_with_indicators, DISPLAY_ROWS, token_data=token_data)
            validate_rsi_calculation(df_with_indicators)
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