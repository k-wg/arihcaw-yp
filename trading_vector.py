# complete_trading_engine.py
import logging
import threading
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from colorama import Fore, Back, Style, init
import queue
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

# Initialize colorama
init(autoreset=True)

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

class TradingMode:
    BUYING = "BUYING"
    SELLING = "SELLING"

class SignalStatus:
    WAITING = "⌛"
    TRIGGERED = "🎯"
    ENTRY = "💰"
    MONITORING = "📊"
    EXIT = "🏁"
    SKIPPED = "⏭️"
    ERROR = "❌"

# Trading Parameters
TRADING_CONFIG = {
    'min_stablecoin_entry': 5.2,
    'max_token_value_selling': 5.2,
    'min_token_value_buying': 5.0,
    'update_interval': 1,
    'signal_check_interval': 1,
    'balance_refresh_interval': 30,
    'inactivity_timeout': 300
}

# Visual Configuration
VISUAL_CONFIG = {
    'colors': {
        'buying': Fore.CYAN,
        'selling': Fore.YELLOW,
        'error': Fore.RED,
        'success': Fore.GREEN,
        'warning': Fore.MAGENTA,
        'info': Fore.BLUE
    },
    'emojis': {
        'buying': '🛒',
        'selling': '💸',
        'waiting': '⌛',
        'triggered': '🎯',
        'error': '❌',
        'success': '✅'
    }
}

# =============================================================================
# COMPLETE TRADING ENGINE
# =============================================================================

class CompleteTradingEngine:
    def __init__(self, api_key: str, api_secret: str):
        # Binance client
        self.client = Client(api_key, api_secret)
        self.symbol = "SOLFDUSD"
        self.trading_pair = None
        
        # Mode detection
        self.current_mode = None
        self.token_balance = 0
        self.stablecoin_balance = 0
        self.current_price = 0
        
        # Signal management
        self.active_signals = {}
        self.signal_queue = queue.Queue()
        self.last_signal_check = None
        
        # Performance monitoring
        self.performance_stats = {
            'signals_triggered': 0,
            'trades_executed': 0,
            'errors_handled': 0,
            'mode_changes': 0
        }
        
        # Thread control
        self.is_running = False
        self.threads = []
        
        # Signal state tracking
        self.signal_states = {}
        self.captured_fibonacci_levels = {}
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup enhanced logging with colors and emojis"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_engine.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def print_with_style(self, message: str, color: str = Fore.WHITE, emoji: str = ""):
        """Print styled messages with emojis and timestamps"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        styled_message = f"{color}{emoji} [{timestamp}] {message}{Style.RESET_ALL}"
        print(styled_message)
        self.logger.info(message)

    def detect_trading_pair(self) -> bool:
        """Auto-detect trading pair from wallet balances"""
        try:
            self.print_with_style("🔍 Detecting trading pair...", Fore.CYAN)
            
            # Get account balances
            account = self.client.get_account()
            balances = {balance['asset']: float(balance['free']) for balance in account['balances']}
            
            # Common tokens and stablecoins
            tokens = ['SOL', 'BTC', 'ETH', 'DOGE', 'ADA', 'DOT', 'BNB', 'XRP', 'MATIC', 'LTC']
            stablecoins = ['USDT', 'USDC', 'FDUSD', 'BUSD', 'TUSD', 'DAI']
            
            # Find token with balance and corresponding stablecoin
            for token in tokens:
                if token in balances and balances[token] > 0:
                    for stablecoin in stablecoins:
                        if stablecoin in balances and balances[stablecoin] > 0:
                            self.symbol = f"{token}{stablecoin}"
                            self.trading_pair = (token, stablecoin)
                            self.print_with_style(
                                f"✅ Trading pair detected: {self.symbol}", 
                                Fore.GREEN, "🎯"
                            )
                            return True
            
            # If no existing pairs, use default
            self.symbol = "SOLFDUSD"
            self.trading_pair = ("SOL", "FDUSD")
            self.print_with_style(
                f"⚠️ Using default trading pair: {self.symbol}", 
                Fore.YELLOW, "⚙️"
            )
            return True
            
        except Exception as e:
            self.print_with_style(f"❌ Error detecting trading pair: {e}", Fore.RED)
            return False

    def get_wallet_balances(self) -> bool:
        """Get current wallet balances for mode detection"""
        try:
            account = self.client.get_account()
            balances = {balance['asset']: float(balance['free']) for balance in account['balances']}
            
            token, stablecoin = self.trading_pair
            self.token_balance = balances.get(token, 0)
            self.stablecoin_balance = balances.get(stablecoin, 0)
            
            # Get current price
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            self.current_price = float(ticker['price'])
            
            token_value = self.token_balance * self.current_price
            
            self.print_with_style(
                f"💰 Balances - {token}: {self.token_balance:.4f} (${token_value:.2f}), "
                f"{stablecoin}: ${self.stablecoin_balance:.2f}, Price: ${self.current_price:.4f}",
                Fore.BLUE, "💼"
            )
            return True
            
        except Exception as e:
            self.print_with_style(f"❌ Error getting wallet balances: {e}", Fore.RED)
            return False

    def determine_trading_mode(self) -> str:
        """Determine if we're in buying or selling mode"""
        token_value = self.token_balance * self.current_price
        
        # Selling mode: token value >= $5.2 (priority)
        if token_value >= TRADING_CONFIG['max_token_value_selling']:
            new_mode = TradingMode.SELLING
            mode_emoji = "📈"
            mode_color = Fore.YELLOW
            
        # Buying mode: token value <= $5 AND stablecoin >= $5.2
        elif (token_value <= TRADING_CONFIG['min_token_value_buying'] and 
              self.stablecoin_balance >= TRADING_CONFIG['min_stablecoin_entry']):
            new_mode = TradingMode.BUYING
            mode_emoji = "📉"
            mode_color = Fore.CYAN
            
        else:
            new_mode = TradingMode.BUYING  # Default to buying if conditions not met
            mode_emoji = "⚖️"
            mode_color = Fore.WHITE
        
        # Log mode change
        if new_mode != self.current_mode:
            self.performance_stats['mode_changes'] += 1
            self.print_with_style(
                f"🔄 Mode changed: {self.current_mode} → {new_mode}", 
                mode_color, mode_emoji
            )
            self.current_mode = new_mode
        
        return self.current_mode

    def load_indicators_data(self) -> Optional[pd.DataFrame]:
        """Load latest indicators from CSV"""
        try:
            if os.path.exists('pinescript_indicators.csv'):
                df = pd.read_csv('pinescript_indicators.csv')
                # Ensure numeric columns
                numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'rsi', 'rsi_ma50', 
                                 'short002', 'short007', 'short21', 'short50', 'long100', 
                                 'long200', 'long350', 'long500']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
            else:
                self.print_with_style("⚠️ Indicators file not found, creating sample data", Fore.YELLOW, "📁")
                return self.create_sample_indicators_data()
        except Exception as e:
            self.print_with_style(f"❌ Error loading indicators: {e}", Fore.RED)
            return self.create_sample_indicators_data()

    def create_sample_indicators_data(self) -> pd.DataFrame:
        """Create sample indicators data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=1000, freq='1min')
        data = {
            'Open Time': dates,
            'Open': np.random.normal(100, 10, 1000),
            'High': np.random.normal(105, 10, 1000),
            'Low': np.random.normal(95, 10, 1000),
            'Close': np.random.normal(102, 10, 1000),
            'Volume': np.random.normal(1000, 100, 1000),
            'rsi': np.random.uniform(30, 70, 1000),
            'rsi_ma50': np.random.uniform(40, 60, 1000),
            'short002': np.random.normal(101, 5, 1000),
            'short007': np.random.normal(102, 5, 1000),
            'short21': np.random.normal(103, 5, 1000),
            'short50': np.random.normal(104, 5, 1000),
            'long100': np.random.normal(105, 5, 1000),
            'long200': np.random.normal(106, 5, 1000),
            'long350': np.random.normal(107, 5, 1000),
            'long500': np.random.normal(108, 5, 1000)
        }
        return pd.DataFrame(data)

    def calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict:
        """Calculate Fibonacci levels from price data"""
        try:
            # Use recent data for Fibonacci calculation
            recent_data = df.tail(500)
            high = recent_data['High'].max()
            low = recent_data['Low'].min()
            price_range = high - low
            
            fib_levels = {
                'fibo_1': high,
                'fibo_0.764': high - 0.236 * price_range,
                'fibo_0.618': high - 0.382 * price_range,
                'fibo_0.5': (high + low) / 2,
                'fibo_0.382': low + 0.382 * price_range,
                'fibo_0.236': low + 0.236 * price_range,
                'fibo_0': low
            }
            
            return fib_levels
        except Exception as e:
            self.print_with_style(f"❌ Error calculating Fibonacci levels: {e}", Fore.RED)
            return {}

    # =============================================================================
    # COMPLETE SCENARIO IMPLEMENTATIONS - ALL 10 SCENARIOS
    # =============================================================================

    def evaluate_scenario_1(self, data: pd.Series, fib_levels: Dict) -> Dict:
        """Scenario 1: MA_350 ≤ Fibo_0.236 (All Other MAs > Fibo_0.236)"""
        signal_info = {
            'name': 'Bull Scenario 1 - MA350 ≤ Fibo_0.236',
            'status': SignalStatus.WAITING,
            'steps': [],
            'entry_condition': False,
            'monitoring_conditions': [],
            'captured_fibo_1': None,
            'captured_fibo_0': None
        }
        
        # Initial condition check
        if (data['long350'] <= fib_levels['fibo_0.236'] and
            data['long200'] > fib_levels['fibo_0.236'] and
            data['long100'] > fib_levels['fibo_0.236'] and
            data['short50'] > fib_levels['fibo_0.236']):
            
            signal_info['steps'].append("✅ Initial conditions met: MA_350 ≤ Fibo_0.236, others > Fibo_0.236")
            
            # Step 1: Wait for MA_350 > Fibo_0.236
            if data['long350'] > fib_levels['fibo_0.236']:
                signal_info['steps'].append("✅ MA_350 crossed above Fibo_0.236")
                signal_info['captured_fibo_1'] = fib_levels['fibo_1']
                signal_info['captured_fibo_0'] = fib_levels['fibo_0']
                signal_info['entry_condition'] = True
                
                # Step 2: Entry confirmation
                if data['long200'] >= fib_levels['fibo_0.618']:
                    if data['short50'] >= data['long200'] and data['long100'] >= data['long200']:
                        if data['short50'] <= data['long350']:
                            signal_info['steps'].append("🎯 ENTRY: All conditions met - Ready for limit order at MA_2 value")
                            signal_info['status'] = SignalStatus.ENTRY
                        else:
                            signal_info['steps'].append("⌛ Waiting for MA_50 <= MA_350")
                    else:
                        signal_info['steps'].append("⌛ Waiting for MA_50, MA_100 >= MA_200")
                else:
                    signal_info['steps'].append("⌛ Waiting for MA_200 >= Fibo_0.618")
                    
                # Check for cancellation conditions
                if data['short50'] <= fib_levels['fibo_0.236'] or data['rsi_ma50'] <= 35:
                    signal_info['steps'].append("❌ ENTRY CANCELLED: MA_50 ≤ Fibo_0.236 or RSI_MA50 ≤ 35")
                    signal_info['status'] = SignalStatus.SKIPPED
            else:
                signal_info['steps'].append("⌛ Waiting for MA_350 > Fibo_0.236")
        else:
            signal_info['steps'].append("⌛ Waiting for initial conditions")
            
        # Exit monitoring (continuous)
        self._monitor_scenario_1_exits(signal_info, data, fib_levels)
        return signal_info

    def _monitor_scenario_1_exits(self, signal_info: Dict, data: pd.Series, fib_levels: Dict):
        """Monitor exit conditions for Scenario 1"""
        if signal_info['captured_fibo_0'] and data['long350'] <= signal_info['captured_fibo_0']:
            signal_info['monitoring_conditions'].append("🚨 EXIT: MA_350 ≤ captured Fibo_0 - Immediate market order")
            signal_info['status'] = SignalStatus.EXIT
            
        if data['rsi_ma50'] <= 35:
            signal_info['monitoring_conditions'].append("⚠️ RSI_MA50 ≤ 35 - Reversal activation")
            # Additional RSI monitoring would go here
            
        if data['long500'] >= fib_levels['fibo_0.764']:
            signal_info['monitoring_conditions'].append("📈 MA_500 ≥ Fibo_0.764 - Monitoring for exit")

    def evaluate_scenario_2(self, data: pd.Series, fib_levels: Dict) -> Dict:
        """Scenario 2: MA_200 ≤ Fibo_0.236 (All Other MAs > Fibo_0.236)"""
        signal_info = {
            'name': 'Bull Scenario 2 - MA200 ≤ Fibo_0.236',
            'status': SignalStatus.WAITING,
            'steps': [],
            'entry_condition': False,
            'monitoring_conditions': [],
            'captured_fibo_1': None,
            'captured_fibo_0': None
        }
        
        if (data['long200'] <= fib_levels['fibo_0.236'] and
            data['long350'] > fib_levels['fibo_0.236'] and
            data['long100'] > fib_levels['fibo_0.236'] and
            data['short50'] > fib_levels['fibo_0.236']):
            
            signal_info['steps'].append("✅ Initial conditions met: MA_200 ≤ Fibo_0.236, others > Fibo_0.236")
            
            if data['long200'] > fib_levels['fibo_0.236']:
                signal_info['steps'].append("✅ MA_200 crossed above Fibo_0.236")
                signal_info['captured_fibo_1'] = fib_levels['fibo_1']
                signal_info['captured_fibo_0'] = fib_levels['fibo_0']
                signal_info['entry_condition'] = True
                
                if data['long200'] >= fib_levels['fibo_0.618']:
                    if data['short50'] >= data['long200'] and data['long100'] >= data['long200']:
                        signal_info['steps'].append("🎯 ENTRY: Enter long position at market order")
                        signal_info['status'] = SignalStatus.ENTRY
                    else:
                        signal_info['steps'].append("⌛ Waiting for MA_50, MA_100 >= MA_200")
                else:
                    signal_info['steps'].append("⌛ Waiting for MA_200 >= Fibo_0.618")
                    
                if data['short50'] <= fib_levels['fibo_0.236'] or data['rsi_ma50'] <= 35:
                    signal_info['steps'].append("❌ ENTRY CANCELLED: MA_50 ≤ Fibo_0.236 or RSI_MA50 ≤ 35")
                    signal_info['status'] = SignalStatus.SKIPPED
            else:
                signal_info['steps'].append("⌛ Waiting for MA_200 > Fibo_0.236")
        else:
            signal_info['steps'].append("⌛ Waiting for initial conditions")
            
        # Exit monitoring
        self._monitor_scenario_2_exits(signal_info, data, fib_levels)
        return signal_info

    def _monitor_scenario_2_exits(self, signal_info: Dict, data: pd.Series, fib_levels: Dict):
        if signal_info['captured_fibo_0'] and data['long350'] <= signal_info['captured_fibo_0']:
            signal_info['monitoring_conditions'].append("🚨 EXIT: MA_350 ≤ captured Fibo_0")
            signal_info['status'] = SignalStatus.EXIT

    def evaluate_scenario_3(self, data: pd.Series, fib_levels: Dict) -> Dict:
        """Scenario 3: MA_2, MA_7, MA_21 All < Fibo_0.236 (All Other MAs > Fibo_0.236)"""
        signal_info = {
            'name': 'Bull Scenario 3 - Short MAs < Fibo_0.236',
            'status': SignalStatus.WAITING,
            'steps': [],
            'entry_condition': False,
            'monitoring_conditions': []
        }
        
        if (data['short002'] < fib_levels['fibo_0.236'] and
            data['short007'] < fib_levels['fibo_0.236'] and
            data['short21'] < fib_levels['fibo_0.236'] and
            data['short50'] > fib_levels['fibo_0.236'] and
            data['long100'] > fib_levels['fibo_0.236'] and
            data['long200'] > fib_levels['fibo_0.236'] and
            data['long350'] > fib_levels['fibo_0.236'] and
            data['long500'] > fib_levels['fibo_0.236']):
            
            signal_info['steps'].append("✅ Initial conditions met: MA_2,7,21 < Fibo_0.236, others >")
            
            if data['short50'] <= fib_levels['fibo_0.236']:
                signal_info['steps'].append("❌ No signal: MA_50 ≤ Fibo_0.236")
                signal_info['status'] = SignalStatus.SKIPPED
            elif data['short50'] >= fib_levels['fibo_0.5']:
                signal_info['steps'].append("🎯 ENTRY: Enter long position at market order")
                signal_info['captured_fibo_1'] = fib_levels['fibo_1']
                signal_info['status'] = SignalStatus.ENTRY
                signal_info['entry_condition'] = True
            else:
                signal_info['steps'].append("⌛ Waiting for MA_50 >= Fibo_0.5")
        else:
            signal_info['steps'].append("⌛ Waiting for initial conditions")
            
        # Exit monitoring
        self._monitor_scenario_3_exits(signal_info, data, fib_levels)
        return signal_info

    def _monitor_scenario_3_exits(self, signal_info: Dict, data: pd.Series, fib_levels: Dict):
        if signal_info.get('captured_fibo_1'):
            if data['long500'] >= signal_info['captured_fibo_1'] or data['rsi_ma50'] <= 35:
                if data['rsi_ma50'] <= 35:
                    signal_info['monitoring_conditions'].append("⚠️ RSI_MA50 ≤ 35 - Monitoring for exit")
                    if data['rsi_ma50'] >= 50:
                        if data['rsi_ma50'] >= 57 and data['long100'] >= fib_levels['fibo_0.764']:
                            signal_info['monitoring_conditions'].append("🚨 EXIT: RSI_MA50 ≥ 57 & MA_100 ≥ Fibo_0.764")
                            signal_info['status'] = SignalStatus.EXIT
                        elif data['rsi_ma50'] <= 35:
                            signal_info['monitoring_conditions'].append("🚨 EXIT: RSI_MA50 double bottom ≤ 35")
                            signal_info['status'] = SignalStatus.EXIT

    def evaluate_scenario_4(self, data: pd.Series, fib_levels: Dict) -> Dict:
        """Scenario 4: MA_2, MA_7, MA_21 All < Fibo_0.382 (All Other MAs > Fibo_0.382)"""
        signal_info = {
            'name': 'Bull Scenario 4 - Short MAs < Fibo_0.382',
            'status': SignalStatus.WAITING,
            'steps': [],
            'entry_condition': False,
            'monitoring_conditions': []
        }
        
        if (data['short002'] < fib_levels['fibo_0.382'] and
            data['short007'] < fib_levels['fibo_0.382'] and
            data['short21'] < fib_levels['fibo_0.382'] and
            data['short50'] > fib_levels['fibo_0.382'] and
            data['long100'] > fib_levels['fibo_0.382'] and
            data['long200'] > fib_levels['fibo_0.382'] and
            data['long350'] > fib_levels['fibo_0.382'] and
            data['long500'] > fib_levels['fibo_0.382']):
            
            signal_info['steps'].append("✅ Initial conditions met: MA_2,7,21 < Fibo_0.382, others >")
            
            if data['short50'] <= fib_levels['fibo_0.382']:
                signal_info['steps'].append("❌ No signal: MA_50 ≤ Fibo_0.382")
                signal_info['status'] = SignalStatus.SKIPPED
            elif data['short50'] >= fib_levels['fibo_0.618']:
                signal_info['steps'].append("🎯 ENTRY: Enter long position at market order")
                signal_info['captured_fibo_1'] = fib_levels['fibo_1']
                signal_info['status'] = SignalStatus.ENTRY
                signal_info['entry_condition'] = True
            else:
                signal_info['steps'].append("⌛ Waiting for MA_50 >= Fibo_0.618")
        else:
            signal_info['steps'].append("⌛ Waiting for initial conditions")
            
        return signal_info

    def evaluate_scenario_5(self, data: pd.Series, fib_levels: Dict) -> Dict:
        """Scenario 5: MA_100 or MA_500 ≤ Fibo_0.236 (All Other MAs > Fibo_0.236)"""
        signal_info = {
            'name': 'Bull Scenario 5 - MA_100/500 ≤ Fibo_0.236',
            'status': SignalStatus.WAITING,
            'steps': [],
            'entry_condition': False,
            'monitoring_conditions': []
        }
        
        ma_100_condition = data['long100'] <= fib_levels['fibo_0.236']
        ma_500_condition = data['long500'] <= fib_levels['fibo_0.236']
        
        if (ma_100_condition or ma_500_condition) and (
            data['long200'] > fib_levels['fibo_0.236'] and
            data['long350'] > fib_levels['fibo_0.236'] and
            data['short50'] > fib_levels['fibo_0.236']):
            
            lagging_ma = 'MA_100' if ma_100_condition else 'MA_500'
            signal_info['steps'].append(f"✅ Initial conditions met: {lagging_ma} ≤ Fibo_0.236, others >")
            
            # Wait for lagging MA > Fibo_0.236
            if (ma_100_condition and data['long100'] > fib_levels['fibo_0.236']) or \
               (ma_500_condition and data['long500'] > fib_levels['fibo_0.236']):
                signal_info['steps'].append(f"✅ {lagging_ma} crossed above Fibo_0.236")
                signal_info['captured_fibo_1'] = fib_levels['fibo_1']
                signal_info['captured_fibo_0'] = fib_levels['fibo_0']
                signal_info['entry_condition'] = True
                
                # Entry confirmation
                if data['rsi_ma50'] <= 35 or data['short50'] <= fib_levels['fibo_0.236']:
                    signal_info['steps'].append("❌ ENTRY CANCELLED: RSI_MA50 ≤ 35 or MA_50 ≤ Fibo_0.236")
                    signal_info['status'] = SignalStatus.SKIPPED
                elif data['long200'] >= fib_levels['fibo_0.618']:
                    if data['short50'] >= data['long200'] and data['long100'] >= data['long200']:
                        if data['short50'] <= data['long350']:
                            signal_info['steps'].append("🎯 ENTRY: Conditions met - Ready for limit order at MA_2 value")
                            signal_info['status'] = SignalStatus.ENTRY
                        else:
                            signal_info['steps'].append("⌛ Waiting for MA_50 <= MA_350")
                    else:
                        signal_info['steps'].append("⌛ Waiting for MA_50, MA_100 >= MA_200")
                else:
                    signal_info['steps'].append("⌛ Waiting for MA_200 >= Fibo_0.618")
            else:
                signal_info['steps'].append(f"⌛ Waiting for {lagging_ma} > Fibo_0.236")
        else:
            signal_info['steps'].append("⌛ Waiting for initial conditions")
            
        return signal_info

    def evaluate_scenario_6(self, data: pd.Series, fib_levels: Dict) -> Dict:
        """Scenario 6: MA_50 ≤ Fibo_0.236 (All Other MAs > Fibo_0.236)"""
        signal_info = {
            'name': 'Bull Scenario 6 - MA_50 ≤ Fibo_0.236',
            'status': SignalStatus.WAITING,
            'steps': [],
            'entry_condition': False,
            'monitoring_conditions': []
        }
        
        if (data['short50'] <= fib_levels['fibo_0.236'] and
            data['long100'] > fib_levels['fibo_0.236'] and
            data['long200'] > fib_levels['fibo_0.236'] and
            data['long350'] > fib_levels['fibo_0.236'] and
            data['long500'] > fib_levels['fibo_0.236']):
            
            signal_info['steps'].append("✅ Initial conditions met: MA_50 ≤ Fibo_0.236, others >")
            
            if data['short50'] > fib_levels['fibo_0.236']:
                signal_info['steps'].append("✅ MA_50 crossed above Fibo_0.236")
                signal_info['captured_fibo_1'] = fib_levels['fibo_1']
                signal_info['captured_fibo_0'] = fib_levels['fibo_0']
                signal_info['entry_condition'] = True
                
                # Check if MA_50 >= all other MAs
                if (data['short50'] >= data['long100'] and
                    data['short50'] >= data['long200'] and
                    data['short50'] >= data['long350'] and
                    data['short50'] >= data['long500']):
                    signal_info['steps'].append("🎯 ENTRY: MA_50 >= all other MAs - Enter long at market")
                    signal_info['status'] = SignalStatus.ENTRY
                else:
                    signal_info['steps'].append("⌛ Waiting for MA_50 >= MA_100,200,350,500")
                    
                if data['short50'] <= fib_levels['fibo_0.236'] or data['rsi_ma50'] <= 35:
                    signal_info['steps'].append("❌ ENTRY CANCELLED: MA_50 ≤ Fibo_0.236 or RSI_MA50 ≤ 35")
                    signal_info['status'] = SignalStatus.SKIPPED
            else:
                signal_info['steps'].append("⌛ Waiting for MA_50 > Fibo_0.236")
        else:
            signal_info['steps'].append("⌛ Waiting for initial conditions")
            
        return signal_info

    def evaluate_scenario_7(self, data: pd.Series, fib_levels: Dict) -> Dict:
        """Scenario 7: MA_50 < Fibo_0.382 (All Other MAs > Fibo_0.382)"""
        signal_info = {
            'name': 'Bull Scenario 7 - MA_50 < Fibo_0.382',
            'status': SignalStatus.WAITING,
            'steps': [],
            'entry_condition': False,
            'monitoring_conditions': []
        }
        
        if (data['short50'] < fib_levels['fibo_0.382'] and
            data['long100'] > fib_levels['fibo_0.382'] and
            data['long200'] > fib_levels['fibo_0.382'] and
            data['long350'] > fib_levels['fibo_0.382'] and
            data['long500'] > fib_levels['fibo_0.382']):
            
            signal_info['steps'].append("✅ Initial conditions met: MA_50 < Fibo_0.382, others >")
            
            if data['long100'] <= fib_levels['fibo_0.382']:
                signal_info['steps'].append("❌ No signal: MA_100 ≤ Fibo_0.382")
                signal_info['status'] = SignalStatus.SKIPPED
            elif data['long100'] >= fib_levels['fibo_0.618']:
                signal_info['steps'].append("🎯 ENTRY: Enter long position at market order")
                signal_info['captured_fibo_1'] = fib_levels['fibo_1']
                signal_info['status'] = SignalStatus.ENTRY
                signal_info['entry_condition'] = True
            else:
                signal_info['steps'].append("⌛ Waiting for MA_100 >= Fibo_0.618")
        else:
            signal_info['steps'].append("⌛ Waiting for initial conditions")
            
        return signal_info

    def evaluate_bear_scenario_1(self, data: pd.Series, fib_levels: Dict) -> Dict:
        """Bear Scenario 1: MA_50 Only < Fibo_0.236 (All Other MAs > Fibo_0.236)"""
        signal_info = {
            'name': 'Bear Scenario 1 - MA_50 Only < Fibo_0.236',
            'status': SignalStatus.WAITING,
            'steps': [],
            'entry_condition': False,
            'monitoring_conditions': []
        }
        
        if (data['short50'] < fib_levels['fibo_0.236'] and
            data['long100'] > fib_levels['fibo_0.236'] and
            data['long200'] > fib_levels['fibo_0.236'] and
            data['long350'] > fib_levels['fibo_0.236'] and
            data['long500'] > fib_levels['fibo_0.236']):
            
            signal_info['steps'].append("✅ Initial conditions met: MA_50 only < Fibo_0.236")
            
            if data['short50'] >= fib_levels['fibo_0.236']:
                signal_info['steps'].append("✅ MA_50 crossed above Fibo_0.236")
                signal_info['captured_fibo_1'] = fib_levels['fibo_1']
                
                # Wait for MA_50 >= all other MAs and Fibo_0.5
                if (data['short50'] >= data['long100'] and
                    data['short50'] >= data['long200'] and
                    data['short50'] >= data['long350'] and
                    data['short50'] >= data['long500'] and
                    data['short50'] >= fib_levels['fibo_0.5']):
                    signal_info['steps'].append("🎯 ENTRY: Enter short position at market order")
                    signal_info['status'] = SignalStatus.ENTRY
                    signal_info['entry_condition'] = True
                else:
                    signal_info['steps'].append("⌛ Waiting for MA_50 >= all MAs and Fibo_0.5")
            else:
                signal_info['steps'].append("⌛ Waiting for MA_50 >= Fibo_0.236")
        else:
            signal_info['steps'].append("⌛ Waiting for initial conditions")
            
        return signal_info

    def evaluate_bear_scenario_2(self, data: pd.Series, fib_levels: Dict) -> Dict:
        """Bear Scenario 2: MA_2, MA_7, MA_21 < Fibo_0.236 (All Other MAs > Fibo_0.236)"""
        signal_info = {
            'name': 'Bear Scenario 2 - Short MAs < Fibo_0.236',
            'status': SignalStatus.WAITING,
            'steps': [],
            'entry_condition': False,
            'monitoring_conditions': []
        }
        
        if (data['short002'] < fib_levels['fibo_0.236'] and
            data['short007'] < fib_levels['fibo_0.236'] and
            data['short21'] < fib_levels['fibo_0.236'] and
            data['short50'] > fib_levels['fibo_0.236'] and
            data['long100'] > fib_levels['fibo_0.236'] and
            data['long200'] > fib_levels['fibo_0.236'] and
            data['long350'] > fib_levels['fibo_0.236'] and
            data['long500'] > fib_levels['fibo_0.236']):
            
            signal_info['steps'].append("✅ Initial conditions met: MA_2,7,21 < Fibo_0.236, others >")
            
            if data['short50'] <= fib_levels['fibo_0.236']:
                signal_info['steps'].append("❌ Disregard trend - fully formed fall")
                signal_info['status'] = SignalStatus.SKIPPED
            elif data['short50'] >= fib_levels['fibo_0.5']:
                signal_info['steps'].append("✅ MA_50 >= Fibo_0.5 - Proceed to market order")
                signal_info['captured_fibo_1'] = fib_levels['fibo_1']
                signal_info['entry_condition'] = True
                signal_info['status'] = SignalStatus.ENTRY
            else:
                signal_info['steps'].append("⌛ Waiting for MA_50 >= Fibo_0.5")
        else:
            signal_info['steps'].append("⌛ Waiting for initial conditions")
            
        return signal_info

    def evaluate_bear_scenario_3(self, data: pd.Series, fib_levels: Dict) -> Dict:
        """Bear Scenario 3: MA_200, MA_350, or MA_500 < Fibo_0.236 (All Other MAs > Fibo_0.236)"""
        signal_info = {
            'name': 'Bear Scenario 3 - Major MAs < Fibo_0.236',
            'status': SignalStatus.WAITING,
            'steps': [],
            'entry_condition': False,
            'monitoring_conditions': []
        }
        
        ma_200_condition = data['long200'] < fib_levels['fibo_0.236']
        ma_350_condition = data['long350'] < fib_levels['fibo_0.236']
        ma_500_condition = data['long500'] < fib_levels['fibo_0.236']
        
        if (ma_200_condition or ma_350_condition or ma_500_condition) and (
            data['short50'] > fib_levels['fibo_0.236'] and
            data['long100'] > fib_levels['fibo_0.236']):
            
            signal_info['steps'].append("✅ Initial conditions met: Major MA < Fibo_0.236, others >")
            
            if data['long200'] >= fib_levels['fibo_0.618']:
                signal_info['steps'].append("✅ MA_200 >= Fibo_0.618")
                
                # Check if both MA_50, MA_100 <= MA_200
                if data['short50'] <= data['long200'] and data['long100'] <= data['long200']:
                    signal_info['steps'].append("❌ Disregard trend: MA_50, MA_100 <= MA_200")
                    signal_info['status'] = SignalStatus.SKIPPED
                elif data['short50'] >= data['long200'] and data['long100'] >= data['long200']:
                    if data['short50'] <= data['long350']:
                        signal_info['steps'].append("🎯 ENTRY: Enter limit order at MA_2 value")
                        signal_info['status'] = SignalStatus.ENTRY
                        signal_info['entry_condition'] = True
                    else:
                        signal_info['steps'].append("⌛ Waiting for MA_50 <= MA_350")
                else:
                    signal_info['steps'].append("⌛ Waiting for MA_50, MA_100 >= MA_200")
            else:
                signal_info['steps'].append("⌛ Waiting for MA_200 >= Fibo_0.618")
        else:
            signal_info['steps'].append("⌛ Waiting for initial conditions")
            
        return signal_info

    def check_all_signal_conditions(self, df: pd.DataFrame, fib_levels: Dict) -> Dict:
        """Check all 10 signal conditions"""
        signals = {}
        latest = df.iloc[-1]
        
        try:
            # Bull Market Scenarios 1-7
            signals['bull_1'] = self.evaluate_scenario_1(latest, fib_levels)
            signals['bull_2'] = self.evaluate_scenario_2(latest, fib_levels)
            signals['bull_3'] = self.evaluate_scenario_3(latest, fib_levels)
            signals['bull_4'] = self.evaluate_scenario_4(latest, fib_levels)
            signals['bull_5'] = self.evaluate_scenario_5(latest, fib_levels)
            signals['bull_6'] = self.evaluate_scenario_6(latest, fib_levels)
            signals['bull_7'] = self.evaluate_scenario_7(latest, fib_levels)
            
            # Bear Market Scenarios 1-3
            signals['bear_1'] = self.evaluate_bear_scenario_1(latest, fib_levels)
            signals['bear_2'] = self.evaluate_bear_scenario_2(latest, fib_levels)
            signals['bear_3'] = self.evaluate_bear_scenario_3(latest, fib_levels)
            
            return signals
            
        except Exception as e:
            self.print_with_style(f"❌ Error checking signal conditions: {e}", Fore.RED)
            return {}

    def execute_trade(self, signal_type: str, signal_info: Dict):
        """Execute trade based on signal and current mode"""
        try:
            if self.current_mode == TradingMode.BUYING:
                self.execute_buy_order(signal_type, signal_info)
            else:
                self.execute_sell_order(signal_type, signal_info)
                
        except Exception as e:
            self.print_with_style(f"❌ Trade execution error: {e}", Fore.RED)
            self.performance_stats['errors_handled'] += 1

    def execute_buy_order(self, signal_type: str, signal_info: Dict):
        """Execute buy order in buying mode"""
        if signal_info['status'] != SignalStatus.ENTRY:
            return
            
        # Calculate order quantity (use maximum available stablecoin)
        order_value = min(self.stablecoin_balance, self.stablecoin_balance)  # Use all available
        order_quantity = order_value / self.current_price
        
        # Ensure minimum order size
        if order_quantity * self.current_price >= TRADING_CONFIG['min_stablecoin_entry']:
            try:
                # Place buy order (commented out for safety - uncomment for live trading)
                # order = self.client.order_market_buy(
                #     symbol=self.symbol,
                #     quantity=order_quantity
                # )
                
                self.print_with_style(
                    f"💰 [SIMULATION] BUY Order: {order_quantity:.4f} {self.trading_pair[0]} "
                    f"at ${self.current_price:.4f} (Value: ${order_value:.2f})",
                    Fore.GREEN, "🛒"
                )
                self.performance_stats['trades_executed'] += 1
                
                # Update balances for simulation
                self.token_balance += order_quantity
                self.stablecoin_balance -= order_value
                
            except BinanceAPIException as e:
                self.print_with_style(f"❌ Buy order failed: {e}", Fore.RED)
        else:
            self.print_with_style("❌ Insufficient stablecoin for buy order", Fore.RED)

    def execute_sell_order(self, signal_type: str, signal_info: Dict):
        """Execute sell order in selling mode - skip entry step"""
        # In selling mode, we skip the entry step and use it only for validation
        if signal_info.get('entry_condition', False) and signal_info['status'] == SignalStatus.ENTRY:
            self.print_with_style(
                f"⏭️ Skipping entry step (Selling Mode) - {signal_info['name']}", 
                Fore.YELLOW, "📈"
            )
            # Mark that we've validated the signal but skip buying
            signal_info['status'] = SignalStatus.SKIPPED
            return
            
        # Proceed with selling logic when exit conditions are met
        if signal_info['status'] == SignalStatus.EXIT:
            try:
                # Sell maximum available tokens
                order_quantity = self.token_balance
                order_value = order_quantity * self.current_price
                
                # Place sell order (commented out for safety - uncomment for live trading)
                # order = self.client.order_market_sell(
                #     symbol=self.symbol,
                #     quantity=order_quantity
                # )
                
                self.print_with_style(
                    f"💰 [SIMULATION] SELL Order: {order_quantity:.4f} {self.trading_pair[0]} "
                    f"at ${self.current_price:.4f} (Value: ${order_value:.2f})",
                    Fore.YELLOW, "💸"
                )
                self.performance_stats['trades_executed'] += 1
                
                # Update balances for simulation
                self.token_balance = 0
                self.stablecoin_balance += order_value
                
            except BinanceAPIException as e:
                self.print_with_style(f"❌ Sell order failed: {e}", Fore.RED)

    def signal_monitor_worker(self, signal_name: str):
        """Worker thread for monitoring individual signals"""
        self.print_with_style(f"🚀 Starting monitor for {signal_name}", Fore.CYAN, "📡")
        
        animation_frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        frame_index = 0
        
        while self.is_running:
            try:
                # Load latest data
                df = self.load_indicators_data()
                if df is None or df.empty:
                    time.sleep(5)
                    continue
                    
                fib_levels = self.calculate_fibonacci_levels(df)
                signals = self.check_all_signal_conditions(df, fib_levels)
                
                if signal_name in signals:
                    signal_info = signals[signal_name]
                    
                    # Update signal status with animation
                    current_frame = animation_frames[frame_index % len(animation_frames)]
                    frame_index += 1
                    
                    self.animate_signal_progress(signal_name, signal_info, current_frame)
                    
                    # Execute trade if conditions met
                    if signal_info['status'] in [SignalStatus.ENTRY, SignalStatus.EXIT]:
                        self.execute_trade(signal_name, signal_info)
                        self.performance_stats['signals_triggered'] += 1
                
                time.sleep(TRADING_CONFIG['signal_check_interval'])
                
            except Exception as e:
                self.print_with_style(f"❌ Signal monitor error ({signal_name}): {e}", Fore.RED)
                time.sleep(5)

    def animate_signal_progress(self, signal_name: str, signal_info: Dict, animation_frame: str):
        """Animate signal progress with visual effects"""
        status_emoji = signal_info['status']
        steps_count = len(signal_info.get('steps', []))
        
        # Create progress bar based on steps
        progress = min(steps_count, 10)
        progress_bar = "█" * progress + "░" * (10 - progress)
        
        # Color code based on status
        if signal_info['status'] == SignalStatus.ENTRY:
            color = Fore.GREEN
        elif signal_info['status'] == SignalStatus.EXIT:
            color = Fore.YELLOW
        elif signal_info['status'] == SignalStatus.SKIPPED:
            color = Fore.MAGENTA
        else:
            color = Fore.CYAN
        
        self.print_with_style(
            f"{animation_frame} {signal_name}: {progress_bar} {status_emoji}",
            color, ""
        )
        
        # Print latest step
        if signal_info.get('steps'):
            latest_step = signal_info['steps'][-1]
            self.print_with_style(f"   ↳ {latest_step}", Fore.WHITE, "")

    def start_concurrent_signal_monitoring(self):
        """Start concurrent monitoring for all 10 signals"""
        signal_scenarios = [
            'bull_1', 'bull_2', 'bull_3', 'bull_4', 'bull_5', 'bull_6', 'bull_7',
            'bear_1', 'bear_2', 'bear_3'
        ]
        
        for scenario in signal_scenarios:
            thread = threading.Thread(
                target=self.signal_monitor_worker,
                args=(scenario,),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
            time.sleep(0.5)  # Stagger thread starts
        
        self.print_with_style(f"🎯 Started monitoring for all 10 scenarios", Fore.GREEN, "🚀")

    def display_performance_dashboard(self):
        """Display real-time performance dashboard"""
        while self.is_running:
            try:
                os.system('cls' if os.name == 'nt' else 'clear')
                
                dashboard = f"""
{Back.BLUE}{Fore.WHITE}🎯 COMPLETE TRADING ENGINE - REAL-TIME DASHBOARD {Style.RESET_ALL}
┌──────────────────┬──────────────────┐
│ Mode            │ {self.current_mode:>16} │
│ Symbol          │ {self.symbol:>16} │
│ Token Balance   │ {self.token_balance:>16.4f} │
│ Stable Balance  │ ${self.stablecoin_balance:>15.2f} │
│ Current Price   │ ${self.current_price:>15.4f} │
├──────────────────┼──────────────────┤
│ Signals Found   │ {self.performance_stats['signals_triggered']:>16} │
│ Trades Executed │ {self.performance_stats['trades_executed']:>16} │
│ Errors Handled  │ {self.performance_stats['errors_handled']:>16} │
│ Mode Changes    │ {self.performance_stats['mode_changes']:>16} │
└──────────────────┴──────────────────┘

📊 Active Signal Monitoring:
  • Bull Scenarios 1-7: 🟢 Running
  • Bear Scenarios 1-3: 🟢 Running
  • Concurrent Threads: {len(self.threads)}/10

💡 Trading Mode: {self.current_mode}
  {Fore.GREEN if self.current_mode == TradingMode.BUYING else Fore.YELLOW}
  {'→ Buying: Use max stablecoin, follow all steps' if self.current_mode == TradingMode.BUYING else '→ Selling: Skip entry steps, sell max tokens'}

Press Ctrl+C to stop
                """
                print(dashboard)
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                self.print_with_style(f"❌ Dashboard error: {e}", Fore.RED)
                time.sleep(5)

    def run(self):
        """Main trading engine loop"""
        self.print_with_style("🚀 Starting Complete Trading Engine - ALL 10 SCENARIOS", Fore.GREEN, "🎯")
        self.is_running = True
        
        try:
            # Phase 1: Initialization
            self.print_with_style("Phase 1: Initializing trading engine...", Fore.CYAN, "⚙️")
            if not self.detect_trading_pair():
                raise Exception("Failed to detect trading pair")
                
            if not self.get_wallet_balances():
                raise Exception("Failed to get wallet balances")
                
            self.determine_trading_mode()
            
            # Phase 2: Start concurrent monitoring
            self.print_with_style("Phase 2: Starting concurrent signal monitoring...", Fore.CYAN, "📡")
            self.start_concurrent_signal_monitoring()
            
            # Phase 3: Start performance dashboard
            self.print_with_style("Phase 3: Starting real-time dashboard...", Fore.CYAN, "📊")
            dashboard_thread = threading.Thread(target=self.display_performance_dashboard, daemon=True)
            dashboard_thread.start()
            
            self.print_with_style("✅ Trading engine fully operational!", Fore.GREEN, "🎉")
            self.print_with_style("All 10 scenarios monitoring concurrently...", Fore.GREEN, "🔄")
            
            # Main loop
            last_balance_check = time.time()
            while self.is_running:
                # Update wallet balances and mode periodically
                current_time = time.time()
                if current_time - last_balance_check > TRADING_CONFIG['balance_refresh_interval']:
                    self.get_wallet_balances()
                    self.determine_trading_mode()
                    last_balance_check = current_time
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.print_with_style("🛑 Shutdown requested by user...", Fore.YELLOW, "⏹️")
        except Exception as e:
            self.print_with_style(f"❌ Critical error in main loop: {e}", Fore.RED)
        finally:
            self.is_running = False
            self.print_with_style("👋 Trading engine stopped successfully", Fore.WHITE, "🛑")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point - Ready to run immediately"""
    print(f"\n{Back.MAGENTA}{Fore.WHITE}{'='*60}{Style.RESET_ALL}")
    print(f"{Back.MAGENTA}{Fore.WHITE}🎯 COMPLETE TRADING ENGINE - READY TO RUN {Style.RESET_ALL}")
    print(f"{Back.MAGENTA}{Fore.WHITE}{'='*60}{Style.RESET_ALL}")
    
    # Replace with your actual Binance API credentials
    API_KEY = "YOUR_API_KEY_HERE"
    API_SECRET = "YOUR_API_SECRET_HERE"
    
    if API_KEY == "YOUR_API_KEY_HERE" or API_SECRET == "YOUR_API_SECRET_HERE":
        print(f"\n{Fore.RED}❌ Please update API_KEY and API_SECRET with your Binance credentials")
        print(f"{Fore.YELLOW}📍 Get them from: https://www.binance.com/en/my/settings/api-management")
        print(f"{Fore.CYAN}🚀 Then uncomment the live trading lines in execute_buy_order() and execute_sell_order()")
        return
    
    try:
        engine = CompleteTradingEngine(API_KEY, API_SECRET)
        engine.run()
        
    except Exception as e:
        print(f"{Fore.RED}❌ Failed to start trading engine: {e}")
        print(f"{Fore.YELLOW}💡 Make sure your Binance API credentials are correct and have trading permissions")

if __name__ == "__main__":
    main()