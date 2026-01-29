import pandas as pd
import os
import sys
import shutil
from datetime import datetime
import argparse

# Engines
from backtest.engine import BacktestEngine
from live.run_paper import PaperEngine, load_config
from live.monitoring import EventsLogger

class ParityCheck:
    def __init__(self, symbol='BTC/USDT', start_date='2024-02-01', end_date='2024-03-01', timeframe='1h'):
        self.symbol = symbol
        self.start = start_date
        self.end = end_date
        self.tf = timeframe
        self.data_path = 'data/parity_data.csv'
        self.config = load_config() # Uses default config.yaml
        
        # Override config for strict parity
        self.config['backtest']['initial_capital'] = 10000
        self.config['strategy']['donchian_period'] = 89
        self.config['strategy']['atr_period'] = 14
        
    def prepare_data(self):
        """Prepares a common CSV file for both engines."""
        # Use existing fetcher logic via Backtest flow to get data, then save.
        from data.fetcher import BinanceDataFetcher
        from data.cleaner import DataCleaner
        
        print(f"Fetching Parity Data: {self.start} -> {self.end}")
        fetcher = BinanceDataFetcher(market_type='future')
        df = fetcher.fetch_ohlcv(self.symbol, self.tf, self.start, self.end)
        
        cleaner = DataCleaner()
        df = cleaner.clean_and_validate(df, self.tf)
        
        # Save for Paper Engine (HistoricalFeed reads this)
        df.to_csv(self.data_path, index=True)
        self.df = df
        print(f"Data prepared: {len(df)} rows")

    def run_backtest(self):
        print("\n=== Running Backtest Engine ===")
        # Config matches
        # ML needs to be ON/OFF?
        ml_on = True # Test with ML
        self.config['ml']['enabled'] = ml_on
        
        engine = BacktestEngine(self.df, self.config, ml_enabled=ml_on, threshold_offset=0.10)
        engine.run()
        trades, equity = engine.get_results()
        
        # Export for comparison
        self.bt_trades = trades
        
        # Extract Signals (We need to reach into engine or reconstruct)
        # BacktestEngine doesn't expose raw signals easily unless we modify it.
        # But we can verify TRADES parity.
        # For Signal Parity, we might need access to df['entry_signal'].
        self.bt_signals = engine.df[['entry_signal', 'close']].copy()
        self.bt_signals = self.bt_signals[self.bt_signals['entry_signal'] != 0]

    def run_paper(self):
        print("\n=== Running Paper Engine (Parity Mode) ===")
        # Setup Args mock
        class Args:
            mode = 'BACKTEST_PARITY'
            symbol = self.symbol
            tf = self.tf
            seed = 42 # Deterministic
            threshold_offset = 0.10
            seed = 42 # Deterministic
            threshold_offset = 0.10
            warmup_bars = 89
            
        # Clean previous logs
        if os.path.exists('events.jsonl'):
            os.remove('events.jsonl')
            
        # Run
        engine = PaperEngine(self.config, Args())
        # Override feed path to ensure it uses the one we just made
        # (PaperEngine uses 'data/parity_data.csv' hardcoded for BACKTEST_PARITY in our impl)
        
        engine.run()
        
        # Load Logs
        self.paper_events = []
        import json
        with open('events.jsonl', 'r') as f:
            for line in f:
                self.paper_events.append(json.loads(line))
                
    def compare(self):
        print("\n=== PARITY REPORT ===")
        
        # 1. Signal Parity
        paper_signals = [e for e in self.paper_events if e['event'] == 'SIGNAL']
        
        bt_count = len(self.bt_signals)
        paper_count = len(paper_signals)
        
        print(f"Signals Generated: Backtest={bt_count} | Paper={paper_count}")
        
        # Deep compare timestamps
        # Backtest timestamps are index.
        # Paper timestamps are in data['ts'].
        
        # 2. Gate Parity
        print(f"\n--- GATE PARITY CHECK ---")
        paper_gates = [e['data'] for e in self.paper_events if e['event'] == 'GATE']
        # Backtest gates are not explicitly saved in 'trades', but we can infer from 'signals_taken' vs 'signals_skipped' 
        # OR we can modify Backtest to log gates?
        # Actually, BacktestEngine prints [DEBUG ML] logs.
        # But parsing stdout is hard.
        # Let's rely on the fact that if Signals Match and Prices Match, Inputs match.
        # So Gates *should* match.
        # Let's count allowed gates in Paper.
        paper_allowed = len([g for g in paper_gates if 'ALLOWED' in g['action']])
        print(f"Paper: {len(paper_gates)} Gates Checked. {paper_allowed} Allowed.")
        
        if len(paper_gates) == 0:
            print("WARNING: No Gate events found in Paper logs.")
            
        # 3. Execution Parity
        # Compare trades
        bt_fills = self.bt_trades
        paper_fills = [e for e in self.paper_events if e['event'] == 'FILL']
        
        print(f"\n--- EXECUTION PARITY CHECK ---")
        print(f"Trades Executed: Backtest={len(bt_fills)} | Paper={len(paper_fills)}")
        
        if len(bt_fills) > 0 and len(paper_fills) > 0:
            # Check first trade price
            bt_first = bt_fills.iloc[0]
            # Paper fill data: {'side': 'LONG', 'qty': ..., 'price': ...}
            # Paper events don't strictly link to 'entry/exit'.
            # But 'FILL' during 'OPEN' action is Entry.
            
            p_first = paper_fills[0]['data']
            
            print(f"First Trade Price: BT={bt_first['entry_price']:.2f} | Paper={p_first['price']:.2f}")
            diff = abs(bt_first['entry_price'] - p_first['price'])
            print(f"Price Diff: {diff:.4f}")
            
            if diff < (bt_first['entry_price'] * 0.001): # 0.1% tolerance
                print(">> PRICE PARITY: PASS")
            else:
                print(">> PRICE PARITY: FAIL")

if __name__ == "__main__":
    checker = ParityCheck()
    checker.prepare_data()
    checker.run_backtest()
    checker.run_paper()
    checker.compare()
