import atexit
import hashlib
import os
import queue
import random
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta

import pandas as pd
import psutil
from lightweight_charts import Chart


class PortManager:
    """Manages port allocation and cleanup for web-based GUI"""

    def __init__(self, min_port=8000, max_port=9000):
        self.min_port = min_port
        self.max_port = max_port
        self.allocated_ports = set()
        self.cleanup_registered = False

    def is_port_available(self, port):
        """Check if a port is available for use"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.bind(("localhost", port))
                return True
        except (socket.error, OSError):
            return False

    def find_available_port(self, max_attempts=50):
        """Find an available port in the specified range"""
        attempts = 0
        while attempts < max_attempts:
            port = random.randint(self.min_port, self.max_port)
            if port not in self.allocated_ports and self.is_port_available(port):
                self.allocated_ports.add(port)
                print(f"✅ Found available port: {port}")
                return port
            attempts += 1

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            port = s.getsockname()[1]
            self.allocated_ports.add(port)
            print(f"⚠️ Using system-assigned port: {port}")
            return port

    def release_port(self, port):
        """Release a port (cleanup)"""
        if port in self.allocated_ports:
            self.allocated_ports.remove(port)
            print(f"🔓 Released port: {port}")

    def cleanup_ports(self):
        """Clean up all allocated ports"""
        ports_to_clean = self.allocated_ports.copy()
        for port in ports_to_clean:
            self.release_port(port)
        print(f"🧹 Cleaned up {len(ports_to_clean)} ports")

    def register_cleanup(self):
        """Register cleanup handlers"""
        if not self.cleanup_registered:
            atexit.register(self.cleanup_ports)
            self.cleanup_registered = True


class ProcessManager:
    """Manages Python processes for clean starts and graceful exits"""

    def __init__(self, script_name=None):
        self.script_name = script_name or os.path.basename(__file__)
        self.current_pid = os.getpid()
        self.cleanup_registered = False
        self.original_signal_handlers = {}
        self.port_manager = PortManager()

    def get_script_processes(self):
        """Get all processes running the current script"""
        script_processes = []
        for proc in psutil.process_iter(["pid", "name", "cmdline", "status"]):
            try:
                if (
                    proc.info["cmdline"]
                    and self.script_name in " ".join(proc.info["cmdline"])
                    and proc.info["pid"] != self.current_pid
                ):
                    script_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                continue
        return script_processes

    def kill_script_processes(self, timeout=5):
        """Kill all processes running this script except the current one"""
        print("🔫 Killing existing script processes...")
        processes = self.get_script_processes()

        if not processes:
            print("✅ No existing script processes found")
            return True

        print(f"🔄 Found {len(processes)} existing process(es) to terminate")
        for proc in processes:
            try:
                print(
                    f"⚠️ Terminating process PID {proc.pid} (Status: {proc.status()})..."
                )
                proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        gone, alive = psutil.wait_procs(processes, timeout=timeout)

        for proc in alive:
            try:
                print(f"💀 Force killing stubborn process PID {proc.pid}...")
                proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        time.sleep(1)
        remaining = self.get_script_processes()
        if remaining:
            print(f"❌ {len(remaining)} process(es) still running after kill attempt:")
            for proc in remaining:
                print(f"   - PID {proc.pid}, Status: {proc.status()}")
            return False
        else:
            print("✅ All existing processes terminated successfully")
            return True

    def ensure_clean_start(self):
        """Ensure clean start by killing existing processes"""
        print("🚀 Ensuring clean start...")
        success = self.kill_script_processes()
        time.sleep(2)
        return success

    def register_exit_handlers(self):
        """Register exit handlers for graceful shutdown"""
        if self.cleanup_registered:
            return

        def cleanup_handler(signum=None, frame=None):
            if signum:
                signal_name = signal.Signals(signum).name
                print(
                    f"\n🛑 Received signal {signal_name} ({signum}), initiating graceful shutdown..."
                )
            else:
                print(f"\n🛑 Shutdown initiated, cleaning up...")
            self._cleanup_resources()
            os._exit(0)

        signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGQUIT"):
            signals.append(signal.SIGQUIT)

        for sig in signals:
            self.original_signal_handlers[sig] = signal.signal(sig, cleanup_handler)

        atexit.register(cleanup_handler)
        self.cleanup_registered = True
        print("✅ Exit handlers registered for graceful shutdown")

    def _cleanup_resources(self):
        """Clean up resources before exit"""
        print("🧹 Cleaning up resources...")
        self.port_manager.cleanup_ports()
        print("✅ Cleanup completed")

    def restore_signal_handlers(self):
        """Restore original signal handlers"""
        for sig, handler in self.original_signal_handlers.items():
            if handler:
                signal.signal(sig, handler)
        self.original_signal_handlers = {}


class MSSecondTemporalGenius:
    """Implements the ms/sec y-axis genius for elegant temporal data handling"""

    def __init__(self):
        self.candle_cache = {}

    def _safe_datetime_conversion(self, series):
        """Safely convert datetime with multiple format attempts - MS/SEC GENIUS CORE"""
        if series.empty:
            return series

        # MILLISECOND PRECISION FORMATS - The Genius Foundation
        formats = [
            "%Y-%m-%d %H:%M:%S.%f",  # Microsecond precision
            "%Y-%m-%d %H:%M:%S",  # Second precision
            "%Y-%m-%d %H:%M",  # Minute precision
            "%Y-%m-%d",  # Day precision
            "%m/%d/%Y %H:%M:%S",  # Alternative formats
            "%m/%d/%Y %H:%M",
            "%m/%d/%Y",
        ]

        for fmt in formats:
            try:
                converted = pd.to_datetime(series, format=fmt, errors="raise")
                print(
                    f"✅ MS/SEC Genius: Successfully parsed datetime with format: {fmt}"
                )
                return converted
            except:
                continue

        # Final fallback - let pandas infer with microsecond support
        try:
            converted = pd.to_datetime(series, errors="coerce")
            print(
                "⚠️ MS/SEC Genius: Using pandas auto datetime parsing (microsecond support)"
            )
            return converted
        except Exception as e:
            print(f"❌ MS/SEC Genius: All datetime parsing attempts failed: {e}")
            return pd.Series([pd.NaT] * len(series))

    def _resample_candles(self, df, timeframe):
        """Resample candle data to specified timeframe - MS/SEC GENIUS RESAMPLING"""
        if df.empty:
            return df

        df = df.set_index("time")

        # MS/SEC GENIUS: Timeframe mapping
        rule_map = {
            "1s": "1S",  # 1 second
            "5s": "5S",  # 5 seconds
            "30s": "30S",  # 30 seconds
        }

        if timeframe not in rule_map:
            return df.reset_index()

        rule = rule_map[timeframe]
        resampled = (
            df.resample(rule)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum" if "volume" in df.columns else None,
                }
            )
            .dropna()
        )
        resampled = resampled.reset_index()

        # Handle volume if needed
        if "volume" not in resampled.columns and "volume" in df.columns:
            resampled["volume"] = 0.0

        print(f"✅ MS/SEC Genius: Resampled to {timeframe} using rule {rule}")
        return resampled

    def _calculate_data_hash(self, df):
        """Calculate hash of data for change detection"""
        if df.empty:
            return None
        try:
            return hashlib.md5(
                pd.util.hash_pandas_object(df).values.tobytes()
            ).hexdigest()
        except:
            return str(df.shape) + str(df.iloc[-1:].to_dict() if not df.empty else "")

    def _robust_file_load(self, filepath, max_retries=30, retry_delay=0.1):
        """Robust file loading with retries, now including column validation"""
        for attempt in range(max_retries):
            try:
                if not os.path.exists(filepath):
                    if attempt == max_retries - 1:
                        print(
                            f"❌ File not found after {max_retries} checks: {filepath}"
                        )
                        return None
                    time.sleep(retry_delay)
                    continue

                file_size = os.path.getsize(filepath)
                if file_size == 0:
                    if attempt == max_retries - 1:
                        print(f"❌ File is empty after {max_retries} checks: {filepath}")
                        return None
                    time.sleep(retry_delay)
                    continue

                # Test read to validate columns/headers
                try:
                    test_df = pd.read_csv(filepath, nrows=1)
                    if test_df.empty or test_df.shape[1] == 0:
                        if attempt == max_retries - 1:
                            print(
                                f"❌ File has no valid columns after {max_retries} checks: {filepath}"
                            )
                            return None
                        time.sleep(retry_delay)
                        continue
                except Exception as read_e:
                    print(f"⚠️ Test read error (attempt {attempt + 1}): {read_e}")
                    if attempt == max_retries - 1:
                        print(
                            f"❌ File validation failed after {max_retries} attempts: {filepath}"
                        )
                        return None
                    time.sleep(retry_delay)
                    continue

                return filepath
            except Exception as e:
                print(f"⚠️ File check error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    print(
                        f"❌ File loading failed after {max_retries} attempts: {filepath}"
                    )
                    return None
                time.sleep(retry_delay)
        return None

    def _load_candle_data_with_cache(self, symbol, timeframe, no_resample=False):
        """Load candle data with MS/SEC GENIUS temporal processing"""
        cache_key = f"{symbol}_{timeframe}_noresample{no_resample}"

        try:
            filepath = "historic_df_alpha.csv"
            valid_file = self._robust_file_load(filepath)
            if not valid_file:
                return pd.DataFrame()

            df = pd.read_csv(filepath)
            if df.empty:
                print("❌ Candle CSV file has no data")
                return pd.DataFrame()

            # Column mapping
            column_mapping = {
                "Open Time": "time",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }

            existing_columns = {
                k: v for k, v in column_mapping.items() if k in df.columns
            }
            df = df.rename(columns=existing_columns)

            # MS/SEC GENIUS: Apply high-precision datetime conversion
            if "time" in df.columns:
                df["time"] = self._safe_datetime_conversion(df["time"])
                df = df.dropna(subset=["time"])
            else:
                print("❌ No time column found in candle data")
                return pd.DataFrame()

            # Convert numeric columns
            numeric_columns = ["open", "high", "low", "close", "volume"]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Drop rows with critical NaN values
            critical_columns = ["open", "high", "low", "close"]
            df = df.dropna(subset=critical_columns)

            if df.empty:
                print("❌ No valid data after cleaning")
                return pd.DataFrame()

            # Select required columns
            required_columns = ["time", "open", "high", "low", "close"]
            if "volume" in df.columns:
                required_columns.append("volume")

            df = df[required_columns]
            df = df.sort_values("time").reset_index(drop=True)

            # MS/SEC GENIUS: Apply resampling ONLY if not no_resample
            if not no_resample:
                df = self._resample_candles(df, timeframe)

            if df.empty:
                print("❌ No valid data after resampling")
                return pd.DataFrame()

            # Cache management
            current_hash = self._calculate_data_hash(df)
            cached_data = self.candle_cache.get(cache_key, {})

            if cached_data.get("hash") == current_hash:
                return cached_data["data"]

            # Update cache
            self.candle_cache[cache_key] = {
                "data": df,
                "hash": current_hash,
                "count": len(df),
                "last_timestamp": df.iloc[-1]["time"] if not df.empty else None,
            }

            resample_note = " (no resample for fine grain)" if no_resample else ""
            print(
                f"✅ MS/SEC Genius: Loaded {len(df)} bars for {timeframe}{resample_note}"
            )
            return df

        except Exception as e:
            print(f"❌ Error loading candle data: {e}")
            import traceback

            traceback.print_exc()
            return pd.DataFrame()


class DataFreshnessMonitor:
    """Monitors data freshness and triggers refreshes when data is stale"""

    def __init__(self, live_chart):
        self.live_chart = live_chart
        self.running = True
        self.monitoring_active = True
        self.stale_threshold_minutes = 3
        self.check_interval_seconds = 60
        self.max_refresh_attempts = 7
        self.current_refresh_attempts = 0
        self.csv_monitoring_active = False
        self.last_csv_check_time = None
        self.csv_check_interval = 10

        print(f"🕒 Data Freshness Monitor initialized:")
        print(f"   - Stale threshold: {self.stale_threshold_minutes} minutes")
        print(f"   - Check interval: {self.check_interval_seconds} seconds")
        print(f"   - Max refresh attempts: {self.max_refresh_attempts}")

    def stop_monitoring(self):
        """Stop the freshness monitoring"""
        self.running = False
        self.monitoring_active = False
        print("🛑 Data Freshness Monitor stopped")

    def start_monitoring(self):
        """Start the freshness monitoring in a separate thread"""
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()
        print("🔍 Data Freshness Monitor started")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        print("🔄 Starting data freshness monitoring loop...")

        while self.running:
            try:
                if not self.monitoring_active:
                    time.sleep(self.check_interval_seconds)
                    continue

                if self.csv_monitoring_active:
                    self._monitor_csv_files()
                    time.sleep(self.csv_check_interval)
                    continue

                print(
                    f"🕒 Checking data freshness... (Attempt {self.current_refresh_attempts + 1}/{self.max_refresh_attempts})"
                )

                latest_timestamp = self._get_latest_candle_timestamp()

                if latest_timestamp:
                    is_stale = self._is_data_stale(latest_timestamp)

                    if is_stale:
                        print(f"⚠️ Data is stale! Last candle: {latest_timestamp}")
                        self._handle_stale_data()
                    else:
                        print(f"✅ Data is fresh. Last candle: {latest_timestamp}")
                        self.current_refresh_attempts = 0
                else:
                    print("❌ Could not determine latest candle timestamp")

                time.sleep(self.check_interval_seconds)

            except Exception as e:
                print(f"❌ Error in freshness monitoring: {e}")
                time.sleep(self.check_interval_seconds)

    def _get_latest_candle_timestamp(self):
        """Get the timestamp of the latest candle"""
        try:
            # Use the appropriate method based on current mode
            candle_data = self.live_chart.temporal_genius._load_candle_data_with_cache(
                self.live_chart.current_symbol,
                self.live_chart.current_timeframe,
                no_resample=self.live_chart.mode_300_active,
            )

            if not candle_data.empty and "time" in candle_data.columns:
                latest_time = candle_data["time"].iloc[-1]
                if isinstance(latest_time, str):
                    latest_time = pd.to_datetime(latest_time)
                return latest_time
        except Exception as e:
            print(f"❌ Error getting latest candle timestamp: {e}")
        return None

    def _is_data_stale(self, timestamp):
        """Check if data is stale (older than threshold)"""
        try:
            current_time = datetime.now()
            if isinstance(timestamp, pd.Timestamp):
                timestamp = timestamp.to_pydatetime()

            time_diff = current_time - timestamp
            return time_diff.total_seconds() > (self.stale_threshold_minutes * 60)
        except Exception as e:
            print(f"❌ Error checking data staleness: {e}")
            return True

    def _handle_stale_data(self):
        """Handle stale data by triggering refresh"""
        print(
            f"🔄 Attempting to refresh stale data (Attempt {self.current_refresh_attempts + 1})"
        )
        self._trigger_smooth_refresh()
        self.current_refresh_attempts += 1

        if self.current_refresh_attempts >= self.max_refresh_attempts:
            print(f"🚨 Max refresh attempts reached. Switching to CSV monitoring mode.")
            self.csv_monitoring_active = True
            self.current_refresh_attempts = 0

    def _trigger_smooth_refresh(self):
        """Trigger a smooth refresh of the chart data"""
        try:
            print("✨ Triggering smooth chart refresh...")
            self.live_chart.refresh_chart_data()
            print("✅ Refresh triggered successfully")

        except Exception as e:
            print(f"❌ Error triggering refresh: {e}")

    def _monitor_csv_files(self):
        """Monitor CSV files to check if they're updating"""
        print("📊 Monitoring CSV file updates...")

        try:
            historic_csv = "historic_df_alpha.csv"
            indicators_csv = "pinescript_indicators.csv"

            historic_updating = self._is_file_updating(historic_csv)
            indicators_updating = self._is_file_updating(indicators_csv)

            print(f"📁 Historic CSV updating: {historic_updating}")
            print(f"📁 Indicators CSV updating: {indicators_updating}")

            if historic_updating and indicators_updating:
                print("✅ Both CSV files are updating! Resuming normal monitoring.")
                self.csv_monitoring_active = False
                self._trigger_smooth_refresh()
            else:
                print("⏳ Waiting for CSV files to start updating...")
                if not historic_updating and not indicators_updating:
                    print(
                        "🚨 ALERT: Both data files are not updating! Check data source."
                    )
                elif not historic_updating:
                    print("🚨 ALERT: Historic data file is not updating!")
                elif not indicators_updating:
                    print("🚨 ALERT: Indicators file is not updating!")

        except Exception as e:
            print(f"❌ Error monitoring CSV files: {e}")

    def _is_file_updating(self, filepath):
        """Check if a file is being updated (modified recently)"""
        try:
            if not os.path.exists(filepath):
                print(f"❌ File does not exist: {filepath}")
                return False

            mod_time = os.path.getmtime(filepath)
            current_time = time.time()
            is_recent = (current_time - mod_time) < 120
            has_content = os.path.getsize(filepath) > 0

            return is_recent and has_content

        except Exception as e:
            print(f"❌ Error checking file update status for {filepath}: {e}")
            return False


class LiveChart:
    def __init__(self):
        # Initialize process manager
        self.process_manager = ProcessManager("fluff_pancakes_00.py")
        self.process_manager.ensure_clean_start()
        self.process_manager.register_exit_handlers()

        # Initialize MS/SEC Genius for Fine mode
        self.temporal_genius = MSSecondTemporalGenius()

        # Find port and setup chart
        self.chart_port = self.process_manager.port_manager.find_available_port()
        self.process_manager.port_manager.register_cleanup()
        os.environ["LIGHTWEIGHT_CHARTS_PORT"] = str(self.chart_port)

        print(f"🌐 Starting chart on port: {self.chart_port}")
        self.chart = Chart(toolbox=True, inner_height=0.75)

        # Core state
        self.current_symbol = "SOLFDUSD"
        self.current_timeframe = "1s"
        self.running = True
        self.data_initialized = False
        self.volume_configured = False

        # TRUE 300 MODE integration - exactly like _precise.py
        self.mode_300_active = False  # Default to Fit mode
        self.max_bars_display = 15000
        self.last_300_bars_data = None

        # Chart components for Fit mode
        self.rsi_chart = None
        self.rsi_line = None
        self.rsi_ma_line = None
        self.ma_lines = {}
        self.fib_lines = {}
        self.current_fib_levels = {}

        # Data management
        self.candle_cache = {}
        self.indicators_cache = {}
        self.last_candle_hash = None
        self.last_indicators_hash = None
        self.auto_scroll = True

        # Update coordination
        self.update_queue = queue.Queue()
        self.last_update_time = 0
        self.update_debounce_ms = 500  # Match precise.py for Fine mode responsiveness

        # Tracking
        self.last_processed_time = None
        self.last_candle_count = 0
        self.last_indicators_count = 0

        # Error tracking
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5

        # JS init flag
        self.js_initialized = False

        # Initialize Data Freshness Monitor
        self.freshness_monitor = DataFreshnessMonitor(self)

        # Apply styling and setup
        self.apply_styling()
        self.setup_chart()

        # Start coordinator thread
        self.coordinator_thread = threading.Thread(
            target=self._update_coordinator, daemon=True
        )
        self.coordinator_thread.start()

    def apply_styling(self):
        """Apply beautiful styling - changes watermark based on mode"""
        self.chart.layout(
            background_color="#090008",
            text_color="#FFFFFF",
            font_size=16,
            font_family="Helvetica",
        )

        self.chart.candle_style(
            up_color="#26A69A",
            down_color="#EF5350",
            border_up_color="#26A69A",
            border_down_color="#EF5350",
            wick_up_color="#26A69A",
            wick_down_color="#EF5350",
        )

        # Watermark will be updated based on mode
        self._update_watermark()

        self.chart.crosshair(mode="normal", vert_color="#FFFFFF", vert_style="dotted")
        self.chart.legend(visible=True, font_size=14)

    def _update_watermark(self):
        """Update watermark based on current mode"""
        if self.mode_300_active:
            self.chart.watermark(
                "1500 MODE - MS/SEC GENIUS", color="rgba(180, 180, 240, 0.7)"
            )
        else:
            self.chart.watermark(
                "FIT MODE - FULL FEATURES", color="rgba(180, 180, 240, 0.7)"
            )

    def _get_recent_bars(self, df):
        """Extract the last 1500 bars for 1500 MODE - EXACTLY like _precise.py"""
        if len(df) <= self.max_bars_display:
            return df
        return df.tail(self.max_bars_display)

    def _clear_update_queue(self):
        """Clear all pending updates in the queue to prevent stale updates"""
        cleared = 0
        while not self.update_queue.empty():
            try:
                self.update_queue.get_nowait()
                self.update_queue.task_done()
                cleared += 1
            except queue.Empty:
                break
        if cleared > 0:
            print(f"🧹 Cleared {cleared} stale updates from queue")

    def _update_coordinator(self):
        """Coordinate updates to prevent rapid redraws - adjusted for Fine mode"""
        while self.running:
            try:
                updates_processed = 0
                max_updates_per_cycle = 3  # Match precise.py for Fine mode

                while (
                    not self.update_queue.empty()
                    and updates_processed < max_updates_per_cycle
                ):
                    update_type, data = self.update_queue.get_nowait()

                    try:
                        if not self.js_initialized:
                            print("⚠️ JS not initialized, skipping update")
                            continue

                        if update_type == "new_bars_update":
                            self._apply_new_bars_update(data)
                        elif update_type == "update_last_bar":
                            self._apply_update_last_bar(data)
                        elif update_type == "indicators_update":
                            self._apply_indicators_update(data)
                        elif update_type == "full_refresh":
                            self._apply_full_refresh(data)

                        updates_processed += 1
                        self.consecutive_errors = 0

                    except Exception as e:
                        print(f"❌ Error processing update: {e}")
                        self.consecutive_errors += 1
                        if self.consecutive_errors >= self.max_consecutive_errors:
                            print(
                                "🚨 Too many consecutive errors, pausing updates for 5 seconds"
                            )
                            time.sleep(5)
                            self.consecutive_errors = 0

                    self.update_queue.task_done()
                    time.sleep(0.1)  # Faster cycle like precise.py

                if updates_processed > 0:
                    print(f"🔄 Applied {updates_processed} batched updates")

                time.sleep(0.1)  # Faster sleep like precise.py

            except Exception as e:
                print(f"❌ Error in update coordinator: {e}")
                time.sleep(1)

    def _apply_new_bars_update(self, new_bars):
        """Apply update for new bars - ONLY in Fit mode"""
        if self.mode_300_active:
            return  # No incremental updates in 1500 MODE; use full refresh
        try:
            if new_bars.empty:
                return

            num_new = len(new_bars)
            for _, row in new_bars.iterrows():
                try:
                    self.chart.update(row)
                except Exception as e:
                    print(f"⚠️ Error updating single bar: {e}")
                    continue

            self.last_candle_count += num_new
            if not new_bars.empty:
                self.last_processed_time = new_bars.iloc[-1]["time"]

            print(f"📈 Updated with {num_new} new bars")

        except Exception as e:
            print(f"❌ Error applying new bars update: {e}")

    def _apply_update_last_bar(self, last_bar):
        """Apply update for the last bar - ONLY in Fit mode"""
        if self.mode_300_active:
            return  # No incremental updates in 1500 MODE
        try:
            self.chart.update(last_bar)
            self.last_processed_time = last_bar["time"]
            print("📈 Updated last bar")

        except Exception as e:
            print(f"❌ Error applying last bar update: {e}")

    def _apply_indicators_update(self, data):
        """Apply indicators update - ONLY in Fit mode"""
        if self.mode_300_active:
            return  # No indicators in 1500 MODE

        try:
            indicators_df, candle_df = data

            if not indicators_df.empty:
                self.add_moving_averages(indicators_df)
                self.add_rsi_panel(indicators_df)
                self.add_fibonacci_levels(candle_df, indicators_df)
                print("✅ Indicators updated successfully")
            else:
                print("⚠️ No indicators data to update")

        except Exception as e:
            print(f"❌ Error applying indicators update: {e}")

    def _apply_full_refresh(self, data):
        """Apply full refresh - handles both modes EXACTLY like their respective versions"""
        try:
            if isinstance(data, tuple) and len(data) == 3:
                candle_data, indicators_data, is_initial = data
            else:
                candle_data, indicators_data = data
                is_initial = False

            if self.mode_300_active:
                # 1500 MODE: EXACTLY like _precise.py - clean, minimal, 1500 bars only, no resample for fine grain
                self.last_300_bars_data = self._get_recent_bars(candle_data)
                self.chart.set(self.last_300_bars_data)

                if not self.volume_configured:
                    try:
                        self.chart.volume_config(
                            up_color="#00ff55", down_color="#ed4807"
                        )
                        self.volume_configured = True
                    except Exception as e:
                        print(f"⚠️ Error configuring volume: {e}")

                self.chart.fit()
                print(
                    f"✅ 1500 Mode: Showing {len(self.last_300_bars_data)} bars with MS/SEC Genius (fine grain detail)"
                )

            else:
                # FIT MODE: Full features with all indicators
                self.chart.set(candle_data)

                if not self.volume_configured:
                    try:
                        self.chart.volume_config(
                            up_color="#00ff55", down_color="#ed4807"
                        )
                        self.volume_configured = True
                    except Exception as e:
                        print(f"⚠️ Error configuring volume: {e}")

                if not indicators_data.empty:
                    try:
                        self.add_moving_averages(indicators_data)
                        self.add_rsi_panel(indicators_data)
                        self.add_fibonacci_levels(candle_data, indicators_data)
                    except Exception as e:
                        print(f"⚠️ Error updating indicators: {e}")

                try:
                    self.chart.fit()
                except Exception as e:
                    print(f"⚠️ Error fitting chart: {e}")

                print("🔄 Fit Mode: Full refresh with all indicators")

            # Update watermark to reflect current mode
            self._update_watermark()

        except Exception as e:
            print(f"❌ Error in full refresh: {e}")

    def queue_update(self, update_type, data):
        """Queue an update for coordinated processing - adjusted debounce for Fine mode"""
        if not self.running:
            return

        current_time = time.time() * 1000
        debounce = (
            500 if self.mode_300_active else 1000
        )  # Faster for Fine mode like precise.py
        if current_time - self.last_update_time < debounce:
            return

        self.last_update_time = current_time
        if (
            self.update_queue.qsize() < 10 if self.mode_300_active else 5
        ):  # Larger queue for Fine mode
            self.update_queue.put((update_type, data))
        else:
            print("⚠️ Update queue full, skipping update")

    def _safe_datetime_conversion(self, series):
        """Safely convert datetime with multiple format attempts"""
        return self.temporal_genius._safe_datetime_conversion(series)

    def _calculate_data_hash(self, df):
        """Calculate hash of data to detect changes efficiently"""
        if df.empty:
            return None
        try:
            return hashlib.md5(
                pd.util.hash_pandas_object(df).values.tobytes()
            ).hexdigest()
        except:
            return str(df.shape) + str(df.iloc[-1:].to_dict() if not df.empty else "")

    def _resample_candles(self, df, timeframe):
        """Resample candle data to the specified timeframe"""
        return self.temporal_genius._resample_candles(df, timeframe)

    def _resample_indicators(self, df, timeframe):
        """Resample indicators data to the specified timeframe"""
        if df.empty:
            return df

        df = df.set_index("time")
        rule_map = {"1s": "1S", "5s": "5S", "30s": "30S"}
        if timeframe not in rule_map:
            return df.reset_index()

        rule = rule_map[timeframe]
        resampled = df.resample(rule).last().dropna(how="all")
        return resampled.reset_index()

    def _robust_file_load(self, filepath, max_retries=30, retry_delay=0.1):
        """Robust file loading with retries"""
        return self.temporal_genius._robust_file_load(
            filepath, max_retries, retry_delay
        )

    def _load_candle_data_with_cache(self, symbol, timeframe=None):
        """Load candle data with caching - uses MS/SEC Genius for both modes with no_resample flag"""
        if timeframe is None:
            timeframe = self.current_timeframe

        # Always use temporal_genius, with no_resample based on mode (True for fine: skip resample for ms/sec detail)
        return self.temporal_genius._load_candle_data_with_cache(
            symbol, timeframe, no_resample=self.mode_300_active
        )

    def _load_indicators_data_with_cache(self, symbol):
        """Load indicators data with caching - ONLY for Fit mode"""
        if self.mode_300_active:
            return pd.DataFrame()  # No indicators in 1500 MODE

        cache_key = f"{symbol}_{self.current_timeframe}"

        try:
            filepath = "pinescript_indicators.csv"
            valid_file = self._robust_file_load(filepath)
            if not valid_file:
                print(f"⚠️ Indicators file not ready: {filepath}")
                return pd.DataFrame()

            # Additional try-except around full read as backup for race conditions
            try:
                df = pd.read_csv(filepath)
            except Exception as read_e:
                print(
                    f"⚠️ Failed to read indicators CSV (possible race/malformed): {read_e}"
                )
                print("⏳ Retrying in 1 second...")
                time.sleep(1)
                try:
                    df = pd.read_csv(filepath)
                except Exception as retry_e:
                    print(f"❌ Final read failed: {retry_e}")
                    return pd.DataFrame()

            if df.empty:
                print("⚠️ Indicators CSV file has no data")
                return pd.DataFrame()

            if "Open Time" in df.columns:
                df = df.rename(columns={"Open Time": "time"})
                df["time"] = self._safe_datetime_conversion(df["time"])
                df = df.dropna(subset=["time"])
            else:
                print("❌ No time column found in indicators data")
                return pd.DataFrame()

            indicator_columns = [
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

            for col in indicator_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.sort_values("time").reset_index(drop=True)
            df = self._resample_indicators(df, self.current_timeframe)

            if df.empty:
                print("⚠️ No valid data after resampling for indicators")
                return pd.DataFrame()

            current_hash = self._calculate_data_hash(df)
            cached_data = self.indicators_cache.get(cache_key, {})

            if cached_data.get("hash") == current_hash:
                return cached_data["data"]

            self.indicators_cache[cache_key] = {
                "data": df,
                "hash": current_hash,
                "count": len(df),
                "last_timestamp": df.iloc[-1]["time"] if not df.empty else None,
            }

            print(
                f"📊 Loaded and resampled {len(df)} indicator bars for {self.current_timeframe}"
            )
            return df

        except Exception as e:
            print(f"❌ Error loading indicators data: {e}")
            import traceback

            traceback.print_exc()
            return pd.DataFrame()

    def _get_new_bars_only(self, current_data, last_timestamp, last_count):
        """Extract only new bars since last update - ONLY for Fit mode"""
        if self.mode_300_active:
            return (
                pd.DataFrame(),
                current_data,
                len(current_data),
            )  # Full refresh only in Fine mode

        if current_data.empty:
            return pd.DataFrame(), current_data, len(current_data)

        current_count = len(current_data)

        if current_count > last_count:
            new_bars = current_data.iloc[last_count:]
            print(f"📊 Found {len(new_bars)} new bars (count-based detection)")
            return new_bars, current_data, current_count

        if last_timestamp and not current_data.empty:
            try:
                new_bars = current_data[current_data["time"] > last_timestamp]
                if not new_bars.empty:
                    print(
                        f"📊 Found {len(new_bars)} new bars (timestamp-based detection)"
                    )
                    return new_bars, current_data, current_count
            except Exception as e:
                print(f"⚠️ Timestamp-based detection failed: {e}")

        return pd.DataFrame(), current_data, current_count

    def add_moving_averages(self, indicators_df):
        """Add/update multiple moving averages from indicators CSV - ONLY Fit mode"""
        if self.mode_300_active or indicators_df.empty:
            return

        ma_configs = [
            (2, "#0097a7", "short002"),
            (7, "#00bc0e", "short007"),
            (21, "#ffeb3b", "short21"),
            (50, "#ffffff", "short50"),
            (100, "#f57c00", "long100"),
            (200, "#ff00f7", "long200"),
            (350, "#ff2b2b", "long350"),
            (500, "#009eff", "long500"),
        ]

        ma_updated = 0
        for period, color, col_name in ma_configs:
            if col_name not in indicators_df.columns:
                continue

            ma_series = pd.to_numeric(indicators_df[col_name], errors="coerce")
            sma_data = pd.DataFrame(
                {
                    "time": indicators_df["time"],
                    f"SMA {period}": ma_series,
                }
            ).dropna()

            if not sma_data.empty:
                try:
                    if period not in self.ma_lines:
                        line = self.chart.create_line(
                            f"SMA {period}", color=color, width=1.5
                        )
                        self.ma_lines[period] = line

                    self.ma_lines[period].set(sma_data)
                    ma_updated += 1
                except Exception as e:
                    print(f"❌ Error updating SMA {period}: {e}")

        if ma_updated > 0:
            print(f"✅ Updated {ma_updated} moving averages")

    def add_rsi_panel(self, indicators_df):
        """Add/update RSI panel with RSI 14 and its 50-period MA from CSV - ONLY Fit mode"""
        if (
            self.mode_300_active
            or indicators_df.empty
            or "rsi" not in indicators_df.columns
        ):
            return

        rsi_series = pd.to_numeric(indicators_df["rsi"], errors="coerce")
        rsi_df = pd.DataFrame(
            {"time": indicators_df["time"], "RSI 14": rsi_series}
        ).dropna()

        if rsi_df.empty:
            return

        rsi_ma_df = pd.DataFrame()
        if "rsi_ma50" in indicators_df.columns:
            rsi_ma_series = pd.to_numeric(indicators_df["rsi_ma50"], errors="coerce")
            rsi_ma_df = pd.DataFrame(
                {"time": indicators_df["time"], "RSI MA 50": rsi_ma_series}
            ).dropna()

        try:
            if self.rsi_chart is None:
                self.rsi_chart = self.chart.create_subchart(
                    width=1.0, height=0.25, sync=True, toolbox=False
                )
                self.rsi_chart.layout(
                    background_color="#090008",
                    text_color="#FFFFFF",
                    font_size=12,
                    font_family="Helvetica",
                )
                self.rsi_chart.legend(visible=True, font_size=12)
                self.rsi_chart.crosshair(
                    mode="normal",
                    vert_color="#FFFFFF",
                    vert_style="dotted",
                    horz_color="#FFFFFF",
                    horz_style="dotted",
                )
                self.rsi_chart.horizontal_line(
                    70, color="rgba(255, 0, 0, 0.5)", style="dashed"
                )
                self.rsi_chart.horizontal_line(
                    50, color="rgba(255, 255, 255, 0.3)", style="dotted"
                )
                self.rsi_chart.horizontal_line(
                    30, color="rgba(0, 255, 0, 0.5)", style="dashed"
                )

                self.rsi_line = self.rsi_chart.create_line(
                    "RSI 14", color="#7E57C2", width=2
                )
                print("✅ Created RSI subchart and lines")

            if not rsi_df.empty:
                self.rsi_line.set(rsi_df)

            if not rsi_ma_df.empty:
                if self.rsi_ma_line is None:
                    self.rsi_ma_line = self.rsi_chart.create_line(
                        "RSI MA 50", color="#FF7043", width=1
                    )
                self.rsi_ma_line.set(rsi_ma_df)

            print("✅ RSI panel updated")

        except Exception as e:
            print(f"❌ Error updating RSI panel: {e}")

    def add_fibonacci_levels(self, candle_df, indicators_df):
        """Add/update Fibonacci retracement levels as horizontal lines from CSV - ONLY Fit mode"""
        if self.mode_300_active or indicators_df.empty or candle_df.empty:
            return

        fib_configs = [
            ("Fib 0%", "#ff2b2b", "level_000"),
            ("Fib 23.6%", "#ff679a", "level_236"),
            ("Fib 38.2%", "#6bffaa", "level_382"),
            ("Fib 50%", "#388e3c", "level_500"),
            ("Fib 61.8%", "#26c6da", "level_618"),
            ("Fib 100%", "#ffffff", "level_100"),
        ]

        fib_updated = 0
        for name, color, col_name in fib_configs:
            if col_name not in indicators_df.columns:
                continue

            level_series = pd.to_numeric(indicators_df[col_name], errors="coerce")
            valid_levels = level_series.dropna()
            if valid_levels.empty:
                continue

            level = valid_levels.iloc[-1]

            try:
                if (
                    name in self.fib_lines
                    and self.current_fib_levels.get(name) == level
                ):
                    continue

                if name in self.fib_lines:
                    try:
                        self.fib_lines[name].delete()
                    except:
                        pass
                    del self.fib_lines[name]

                line = self.chart.horizontal_line(
                    level, color=color, width=1, text=name
                )
                self.fib_lines[name] = line
                self.current_fib_levels[name] = level
                fib_updated += 1

            except Exception as e:
                print(f"❌ Error updating {name}: {e}")

        if fib_updated > 0:
            print(f"✅ Updated {fib_updated} Fibonacci levels")

    def setup_chart(self):
        """Setup chart events and controls including TRUE 1500 MODE toggle"""
        self.chart.events.search += self.on_search
        self.chart.events.range_change += self.on_range_change

        self.chart.topbar.textbox("symbol", self.current_symbol)

        # Timeframe switcher - available ONLY in Fit mode
        self.chart.topbar.switcher(
            "timeframe",
            ("1s", "5s", "30s"),
            default=self.current_timeframe,
            func=self.on_timeframe_selection,
        )

        # Live update toggle - available in both modes
        self.chart.topbar.switcher(
            "live_update",
            ("Live: ON", "Live: OFF"),
            default="Live: ON",
            func=self.on_live_update_change,
        )

        # TRUE 1500 MODE toggle - EXACT implementation
        self.chart.topbar.switcher(
            "mode_300",
            ("Fit", "Fine"),
            default="Fit",  # Default to Fit mode
            func=self.on_300_mode_change,
        )

    def on_range_change(self, chart, bars_before, bars_after):
        """Handle visible range changes - ONLY in Fit mode"""
        if self.mode_300_active:
            return  # Auto-scroll always in Fine mode
        self.auto_scroll = bars_after == 0

    def on_300_mode_change(self, chart):
        """Handle 1500 MODE toggle - EXACT implementation from _precise.py"""
        previous_mode = self.mode_300_active
        self.mode_300_active = chart.topbar["mode_300"].value == "Fine"

        # Clear the update queue to prevent stale incremental updates from previous mode
        self._clear_update_queue()

        if self.mode_300_active:
            self.current_timeframe = "1s"  # Fix to 1s like precise.py
            print("🎯 Switching to 1500 MODE - MS/SEC GENIUS")
            print("   ✅ Always shows last 1500 bars")
            print("   ✅ Millisecond-precision datetime parsing")
            print("   ✅ Raw ms/sec data (no resampling for fine grain x-axis detail)")
            print("   ✅ Elegant ms/sec y-axis presentation")
            print("   ✅ Clean, minimal interface")
            print("   ✅ Timeframe fixed to 1s")
        else:
            print("🎯 Switching to FIT MODE - FULL FEATURES")
            print("   ✅ All available data")
            print("   ✅ RSI panel with moving averages")
            print("   ✅ Multiple moving averages")
            print("   ✅ Fibonacci levels")
            print("   ✅ Complete trading toolkit")
            print("   ✅ Timeframe selectable: 1s, 5s, 30s")

        # Clear caches if switching from Fine to Fit (or vice versa) to force reload
        if previous_mode != self.mode_300_active:
            self.candle_cache.clear()
            self.indicators_cache.clear()
            self.last_candle_hash = None
            self.last_indicators_hash = None

        # Refresh chart with new mode
        self.refresh_chart_data()

    def refresh_chart_data(self):
        """Refresh chart data - handles both modes appropriately"""
        print(
            f"🔄 Refreshing {'1500 Mode' if self.mode_300_active else 'Fit Mode'} with MS/SEC Genius..."
        )

        # Use MS/SEC Genius to load data for both modes, but limit bars in Fine, no resample in Fine
        candle_data = self._load_candle_data_with_cache(
            self.current_symbol, self.current_timeframe
        )
        indicators_data = pd.DataFrame()  # No indicators in 1500 MODE; load only in Fit

        if not self.mode_300_active:
            indicators_data = self._load_indicators_data_with_cache(self.current_symbol)

        if candle_data.empty:
            print("❌ No data available for refresh")
            return

        # In Fine mode, always full refresh with 1500 bars limit
        if self.mode_300_active:
            self.last_300_bars_data = self._get_recent_bars(candle_data)

        self.queue_update("full_refresh", (candle_data, indicators_data, False))

        if not candle_data.empty:
            self.last_processed_time = candle_data.iloc[-1]["time"]
            self.last_candle_count = len(candle_data)

    def check_for_updates(self):
        """Check for new data and queue appropriate updates - ONLY incremental in Fit mode"""
        if not self.data_initialized or not self.js_initialized or self.mode_300_active:
            if self.mode_300_active:
                self.refresh_chart_data()  # Full refresh only in Fine mode
            return

        try:
            current_candle_data = self._load_candle_data_with_cache(
                self.current_symbol, self.current_timeframe
            )
            current_indicators_data = self._load_indicators_data_with_cache(
                self.current_symbol
            )

            if current_candle_data.empty:
                return

            current_candle_hash = self._calculate_data_hash(current_candle_data)
            current_indicators_hash = self._calculate_data_hash(current_indicators_data)

            candle_changed = current_candle_hash != self.last_candle_hash
            indicators_changed = current_indicators_hash != self.last_indicators_hash

            new_bars, full_candle_data, new_candle_count = self._get_new_bars_only(
                current_candle_data, self.last_processed_time, self.last_candle_count
            )

            if candle_changed:
                if new_candle_count > self.last_candle_count:
                    self.queue_update("new_bars_update", new_bars)
                    update_type = "new bars"
                else:
                    last_bar = current_candle_data.iloc[-1]
                    self.queue_update("update_last_bar", last_bar)
                    update_type = "last bar"
                self.last_candle_hash = current_candle_hash
                print(f"🔄 Candle {update_type} update queued")

            if indicators_changed:
                self.queue_update(
                    "indicators_update", (current_indicators_data, full_candle_data)
                )
                self.last_indicators_hash = current_indicators_hash

            self.last_candle_count = new_candle_count
            if not new_bars.empty:
                self.last_processed_time = new_bars.iloc[-1]["time"]

        except Exception as e:
            print(f"❌ Error checking for updates: {e}")

    def initialize_chart_data(self):
        """Initialize chart with data on startup - uses MS/SEC Genius for both"""
        print("🔄 Initializing chart data with MS/SEC Genius...")

        # Always use temporal_genius for initial load with no_resample flag
        initial_candle_data = self._load_candle_data_with_cache(
            self.current_symbol, self.current_timeframe
        )
        initial_indicators_data = pd.DataFrame()

        if not self.mode_300_active:
            initial_indicators_data = self._load_indicators_data_with_cache(
                self.current_symbol
            )

        if initial_candle_data.empty:
            print("❌ No initial candle data available.")
            return False

        try:
            # Wait a bit for JS to initialize
            time.sleep(2)
            self.js_initialized = True

            # In Fine mode, limit to 1500 bars immediately
            if self.mode_300_active:
                initial_candle_data = self._get_recent_bars(initial_candle_data)

            self.queue_update(
                "full_refresh", (initial_candle_data, initial_indicators_data, True)
            )
            time.sleep(1)

            if not initial_candle_data.empty:
                self.last_processed_time = initial_candle_data.iloc[-1]["time"]
                self.last_candle_count = len(initial_candle_data)
                self.last_candle_hash = self._calculate_data_hash(initial_candle_data)

            self.last_indicators_hash = self._calculate_data_hash(
                initial_indicators_data
            )
            self.last_indicators_count = len(initial_indicators_data)

            self.data_initialized = True

            print("✅ Chart data initialized successfully")
            return True

        except Exception as e:
            print(f"❌ Error initializing chart: {e}")
            return False

    def live_update_loop(self):
        """Background thread for live updates - faster check in Fine mode"""
        while not self.data_initialized and self.running:
            time.sleep(0.5)

        print("🔄 Live update loop started")

        last_check_time = time.time()
        check_interval = 1 if self.mode_300_active else 2  # Faster like precise.py

        while self.running:
            try:
                if self.chart.topbar["live_update"].value == "Live: ON":
                    current_time = time.time()
                    if current_time - last_check_time >= check_interval:
                        self.check_for_updates()
                        last_check_time = current_time

                time.sleep(
                    0.5 if self.mode_300_active else 1
                )  # Faster sleep in Fine mode

            except Exception as e:
                print(f"❌ Error in update loop: {e}")
                time.sleep(2)

    def on_search(self, chart, searched_string):
        """Handle symbol search - uses MS/SEC Genius in Fine mode"""
        print(f"🔍 Switching to symbol: {searched_string}")
        self.current_symbol = searched_string
        self.auto_scroll = True

        # Use temporal_genius for candles in both modes with no_resample flag
        new_candle_data = self._load_candle_data_with_cache(
            searched_string, self.current_timeframe
        )
        new_indicators_data = pd.DataFrame()

        if not self.mode_300_active:
            new_indicators_data = self._load_indicators_data_with_cache(searched_string)

        if new_candle_data.empty:
            print(f"❌ No data found for symbol: {searched_string}")
            return

        # Limit to 1500 in Fine mode
        if self.mode_300_active:
            new_candle_data = self._get_recent_bars(new_candle_data)

        # Clear queue before new search to avoid stale updates
        self._clear_update_queue()

        chart.topbar["symbol"].set(searched_string)
        self.queue_update("full_refresh", (new_candle_data, new_indicators_data, True))

        if not new_candle_data.empty:
            self.last_processed_time = new_candle_data.iloc[-1]["time"]
            self.last_candle_count = len(new_candle_data)
            self.last_candle_hash = self._calculate_data_hash(new_candle_data)

        self.last_indicators_hash = self._calculate_data_hash(new_indicators_data)
        self.last_indicators_count = len(new_indicators_data)

    def on_timeframe_selection(self, chart):
        """Handle timeframe changes - FIXED in Fine mode; simplified direct load/set for fine grain"""
        if self.mode_300_active:
            print(
                "⏰ Timeframe fixed to 1s in Fine mode - MS/SEC Genius active (raw ms/sec detail)"
            )
            # Simplified direct refresh for fine mode consistency
            self.refresh_chart_data()
            return

        self.current_timeframe = chart.topbar["timeframe"].value
        print(f"⏰ Changing timeframe to: {self.current_timeframe}")
        self.on_search(chart, self.current_symbol)

    def on_live_update_change(self, chart):
        """Handle live update toggle"""
        status = (
            "enabled" if chart.topbar["live_update"].value == "Live: ON" else "disabled"
        )
        print(f"🔄 Live updates {status}")

    def start(self):
        """Start the chart with TRUE dual-mode functionality"""
        print("🚀 Starting Enhanced Live Trading Chart...")
        print(f"🌐 Using port: {self.chart_port}")

        if not self.initialize_chart_data():
            print("❌ Failed to initialize chart data. Exiting.")
            return

        update_thread = threading.Thread(target=self.live_update_loop, daemon=True)
        update_thread.start()

        self.freshness_monitor.start_monitoring()

        print("\n🎯 TRUE DUAL-MODE FUNCTIONALITY:")
        print("   ✅ FIT MODE: Full trading toolkit with all indicators")
        print(
            "   ✅ FINE MODE: 1500 MODE with MS/SEC Genius temporal precision (fixed 1s, raw ms/sec detail)"
        )
        print("\n⚡ CONTROLS:")
        print("   - Symbol search: Type any symbol")
        print("   - Timeframe: 1s, 5s, 30s (Fit mode only; fixed 1s in Fine)")
        print("   - Live updates: ON/OFF toggle")
        print("   - Mode: Fit (full features) / Fine (1500 bars precision)")

        try:
            self.chart.show(block=True)
            self.js_initialized = True
        except Exception as e:
            print(f"❌ Error showing chart: {e}")
        finally:
            self.running = False

        self.freshness_monitor.stop_monitoring()

        print("👋 Chart closed, performing final cleanup...")
        self.process_manager._cleanup_resources()


if __name__ == "__main__":
    try:
        live_chart = LiveChart()
        live_chart.start()
    except KeyboardInterrupt:
        print("\n🛑 Script interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("🏁 Script execution completed")
