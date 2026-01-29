
import pandas as pd
import numpy as np
import yaml
import os
import argparse
import sys
from datetime import datetime
from backtest.engine import BacktestEngine
from src.ml.train_wfo import WFO_ML_Trainer
from data.fetcher import BinanceDataFetcher
from data.cleaner import DataCleaner

def load_config(config_path='config.yaml'):
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Rolling WFO Runner')
    parser.add_argument('--start', type=str, default="2023-01-01")
    parser.add_argument('--end', type=str, default="2024-06-30")
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--download', action='store_true', help='Download fresh data')
    args = parser.parse_args()

    # Load Config
    config = load_config(args.config)
    
    # Update Config dates
    config['data']['start_date'] = args.start
    config['data']['end_date'] = args.end
    config['ml']['enabled'] = True  # Ensure ML is on

    print(f"Starting Rolling WFO from {args.start} to {args.end}...")
    
    # 1. Fetch & Prepare Data
    symbol = config['data']['symbol']
    timeframe = config['data']['timeframe']
    
    fetcher = BinanceDataFetcher()
    cleaner = DataCleaner()
    
    data_filename = f"data/storage/{symbol.replace('/','')}_{timeframe}_{args.start.replace(' ','_').replace(':','')}_{args.end.replace(' ','_').replace(':','')}.csv"
    
    if args.download or not os.path.exists(data_filename):
        print("Downloading data...")
        ccxt_symbol = symbol.replace('USDT', '/USDT') if '/' not in symbol else symbol
        df = fetcher.fetch_ohlcv(ccxt_symbol, timeframe, args.start, args.end)
        df = cleaner.clean_and_validate(df, timeframe)
        os.makedirs("data/storage", exist_ok=True)
        df.to_csv(data_filename)
    else:
        print(f"Using local data: {data_filename}")
        df = pd.read_csv(data_filename)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)

    # 2. Train Models (Rolling)
    print("\n>>> PHASE 1: Rolling Training <<<")
    trainer = WFO_ML_Trainer(config)
    # This will generate models/fold_XX.pkl and meta files
    folds_meta = trainer.train(df)
    
    if not folds_meta:
        print("Training failed or no folds generated.")
        return

    print(f"\nGenerated {len(folds_meta)} folds.")

    # 3. Test Each Fold (OOS)
    print("\n>>> PHASE 2: Rolling Backtest (OOS) <<<")
    
    rollup_metrics = []
    
    for fold in folds_meta:
        fold_id = fold['fold']
        test_start = fold['test_range'][0]
        test_end = fold['test_range'][1]
        
        print(f"\nRunning Fold {fold_id} OOS: {test_start} -> {test_end}")
        
        # Initialize Engine for this slice
        # We must reload engine or re-init to pick up correct models?
        # Actually MLEngine loads all models at init. 
        # Since we trained all models in Phase 1, a single BacktestEngine init *might* work if it loads them all.
        # But to be safe and clean, we execute a run per fold.
        
        # Slice Data for this fold
        df_slice = df[(df.index >= test_start) & (df.index < test_end)].copy()
        
        if df_slice.empty:
            print(f"No data for fold {fold_id} OOS.")
            continue
            
        # Initialize Engine with sliced data
        engine = BacktestEngine(df_slice, config, ml_enabled=True) 
        
        # Run Backtest
        engine.run()
        
        # Collect Metrics
        res = engine.get_results() # Assuming get_results or accessing calculated stats
        
        # Extract specific stats
        final_equity = engine.equity
        initial_equity = engine.initial_capital
        net_return_pct = ((final_equity - initial_equity) / initial_equity) * 100
        
        # Calculate PF & MaxDD from trade list
        trades = engine.trades
        win_rate = 0
        pf = 0
        if trades:
            wins = [t for t in trades if t.pnl > 0]
            loss_abs = [abs(t.pnl) for t in trades if t.pnl <= 0]
            win_rate = (len(wins) / len(trades)) * 100
            total_win = sum(t.pnl for t in wins)
            total_loss = sum(loss_abs)
            pf = total_win / total_loss if total_loss > 0 else 999
            
        # ML Stats
        ml_total = engine.stats.get('signals_total_in_oos', 0)
        
        # Drift check
        # engine.stats['folds'][fold_key]['probs']
        fold_key = f"fold_{fold_id:02d}"
        avg_prob = 0
        probs = []
        if 'folds' in engine.stats and fold_key in engine.stats['folds']:
             # Check if probs key exists, if not usage is limited
             # Note: check backtest/engine.py if it stores 'probs'
             pass
                
        metrics = {
            'Fold': fold_id,
            'Start': test_start,
            'End': test_end,
            'Return (%)': round(net_return_pct, 2),
            'PF': round(pf, 2),
            'Trades': len(trades),
            'Win Rate': round(win_rate, 2),
            'Equity': round(final_equity, 2)
        }
        rollup_metrics.append(metrics)

    # 4. Report
    print("\n>>> ROLLING WFO RESULTS <<<")
    results_df = pd.DataFrame(rollup_metrics)
    print(results_df.to_string(index=False))
    
    # Total Stats
    total_return = results_df['Return (%)'].sum() # Approximation (arithmetic)
    avg_pf = results_df['PF'].mean()
    print(f"\nAverage PF: {avg_pf:.2f}")
    print(f"Total Cumulative Return (Sum): {total_return:.2f}%")
    
    # Save
    results_df.to_csv("wfo_rolling_results.csv", index=False)
    print("\nSaved details to wfo_rolling_results.csv")

if __name__ == '__main__':
    main()
