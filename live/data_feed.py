
import pandas as pd
import time
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import sys

# Try standard timezone, fallback to backports
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

# Existing fetcher for LiveFeed
from data.fetcher import BinanceDataFetcher

logger = logging.getLogger(__name__)

class DataFeed(ABC):
    """
    Abstract Base Class for Data Feeds (Historical & Live).
    Provides a unified interface for yielding new bars.
    """
    
    @abstractmethod
    def get_latest_bars(self, n=100) -> pd.DataFrame:
        """Returns the last n closed bars as a DataFrame."""
        pass
    
    @abstractmethod
    def get_current_price(self) -> float:
        """Returns the current market price (Ticker or Next Open)."""
        pass

class HistoricalFeed(DataFeed):
    """
    Simulates a live feed using historical CSV data.
    Used for 'Parity Check' and deterministic testing.
    """
    def __init__(self, csv_path, symbol='BTC/USDT', start_date=None, end_date=None):
        self.symbol = symbol
        self.df_full = pd.read_csv(csv_path)
        self.df_full['datetime'] = pd.to_datetime(self.df_full['datetime'])
        # Ensure UTC aware if not already
        if self.df_full['datetime'].dt.tz is None:
             self.df_full['datetime'] = self.df_full['datetime'].dt.tz_localize('UTC')
             
        self.df_full.set_index('datetime', inplace=True)
        self.df_full.sort_index(inplace=True)
        
        # Date filtering
        if start_date:
            ts_start = pd.Timestamp(start_date).tz_localize('UTC') if isinstance(start_date, str) else start_date
            self.df_full = self.df_full[self.df_full.index >= ts_start]
        if end_date:
            ts_end = pd.Timestamp(end_date).tz_localize('UTC') if isinstance(end_date, str) else end_date
            self.df_full = self.df_full[self.df_full.index <= ts_end]
            
        self.current_idx = 0
        self.total_bars = len(self.df_full)
        self.last_yielded_idx = -1
        
        print(f"[HistoricalFeed] Loaded {self.total_bars} bars from {csv_path}")
        print(f"Range: {self.df_full.index.min()} -> {self.df_full.index.max()}")

    def get_latest_bars(self, n=100) -> pd.DataFrame:
        if self.current_idx == 0:
            return pd.DataFrame()
        
        end_loc = self.current_idx
        start_loc = max(0, end_loc - n)
        return self.df_full.iloc[start_loc:end_loc]

    def get_current_price(self) -> float:
        if self.current_idx < self.total_bars:
            return self.df_full.iloc[self.current_idx]['open']
        else:
            return self.df_full.iloc[-1]['close'] # End of data fallback

    def wait_for_next_bar(self):
        while self.current_idx < self.total_bars - 1: # Leave 1 for 'next open'
            # The 'current' bar to process (Closed Bar)
            bar_to_yield = self.df_full.iloc[self.current_idx:self.current_idx+1]
            
            # Store idx
            self.last_yielded_idx = self.current_idx
            
            self.current_idx += 1
            yield bar_to_yield

class LiveFeed(DataFeed):
    """
    Polls Binance via CCXT for real-time data.
    """
    def __init__(self, symbol='BTC/USDT', timeframe='1h', limit=1000):
        self.symbol = symbol
        self.timeframe = timeframe
        self.fetcher = BinanceDataFetcher(market_type='future')
        self.last_closed_bar_time = None
        self.limit = limit
        
        # Timeframe parsing (simple)
        if timeframe.endswith('h'):
            self.delta = pd.Timedelta(hours=int(timeframe[:-1]))
        elif timeframe.endswith('m'):
            self.delta = pd.Timedelta(minutes=int(timeframe[:-1]))
        elif timeframe.endswith('d'):
            self.delta = pd.Timedelta(days=int(timeframe[:-1]))
        else:
            raise ValueError(f"Unknown timeframe format: {timeframe}")

        print(f"[LiveFeed] Initialized for {symbol} {timeframe}")

    def get_latest_bars(self, n=500) -> pd.DataFrame:
        try:
            # Calculate range for fallback (roughly)
            # Fetcher handles logic.
            # But wait, get_latest_bars usually wants CLOSED bars.
            # Fetcher.fetch_ohlcv includes open bar at end.
            
            now = datetime.now(ZoneInfo("UTC"))
            start = now - (self.delta * (n + 5)) # Fetch a bit more to be safe
            
            df = self.fetcher.fetch_ohlcv(self.symbol, self.timeframe, 
                                          start_date=start, 
                                          end_date=None, 
                                          limit=n+5)
            
            if df.empty:
                return pd.DataFrame()
            
            # Exclude current open bar (last row)
            # Assuming fetcher returns [Closed, Closed, ..., Open]
            # Since fetcher logic is purely time based, if end_date=None it fetches latest.
            # Latest candle is usually open.
            
            closed_df = df.iloc[:-1] # Remove last
            return closed_df.iloc[-n:] # Take last n
            
        except Exception as e:
            logger.error(f"Fetch error: {e}")
            return pd.DataFrame()

    def get_current_price(self) -> float:
        """
        Fetches the current ticker price (Real-time).
        """
        try:
            ticker = self.fetcher.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Ticker Fetch Error: {e}")
            return 0.0

    def wait_for_next_bar(self):
        """
        Infinite generator loops forever.
        Polls every X seconds. If a new closed bar is detected, yields it.
        """
        print("[LiveFeed] Starting poll loop...")
        
        # Initial Seed (Retry Loop)
        retry_count = 0
        max_retries = 5
        while retry_count < max_retries:
            latest_df = self.get_latest_bars(n=200) # Fetch more for warm-up
            if not latest_df.empty:
                self.last_closed_bar_time = latest_df.index[-1]
                print(f"[LiveFeed] Seed Data OK. Baseline: {self.last_closed_bar_time} (Count: {len(latest_df)})")
                break
            
            wait_time = 2 ** retry_count
            print(f"[LiveFeed] Seed Fetch Failed. Retrying in {wait_time}s... ({retry_count+1}/{max_retries})")
            time.sleep(wait_time)
            retry_count += 1
            
        if self.last_closed_bar_time is None:
            logger.error("[LiveFeed] CRITICAL: Could not fetch initial seed data after retries.")
            # Depending on policy, maybe raise or continue waiting
            print("[LiveFeed] Continuing to wait in loop (Degraded Mode)...")
        
        while True:
            # Sleep a bit
            time.sleep(10) # 10s is safe for 1h timeframe
            
            try:
                # Fetch recent candles (fetcher handles start_date=None -> latest)
                df = self.fetcher.fetch_ohlcv(self.symbol, self.timeframe, 
                                              start_date=None, 
                                              end_date=None,
                                              limit=5)
                
                if df is None or df.empty:
                    continue

                if len(df) < 2:
                    current_len = len(df)
                    # print(f"[LiveFeed] Not enough bars ({current_len}). Waiting...")
                    continue
                
                # The latest bar in the array is usually the *current open* bar.
                # The one before it is the *last closed* bar.
                last_closed_candidate = df.index[-2]
                
                if self.last_closed_bar_time is None:
                    # First valid detection
                    self.last_closed_bar_time = last_closed_candidate
                    print(f"[LiveFeed] Baseline established: {self.last_closed_bar_time}")
                    continue
                
                if last_closed_candidate > self.last_closed_bar_time:
                    # Detected a new closure!
                    closed_bar_row = df.iloc[-2:-1] # Keep as DataFrame (1 row)
                    
                    print(f"[LiveFeed] New Bar Detected: {last_closed_candidate}")
                    self.last_closed_bar_time = last_closed_candidate
                    
                    yield closed_bar_row
                    
            except Exception as e:
                logger.error(f"[LiveFeed] Polling Error: {e}")
                time.sleep(5)
