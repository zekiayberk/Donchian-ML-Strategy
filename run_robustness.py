
import pandas as pd
import numpy as np
import yaml
import os
import argparse
import itertools
import hashlib
from backtest.engine import BacktestEngine
from data.fetcher import BinanceDataFetcher
from data.cleaner import DataCleaner
from src.utils.wfo_utils import generate_fold_ranges, get_data_hash

def load_config(config_path='config.yaml'):
    with open(config_path) as f:
        return yaml.safe_load(f)

def run_backtest_for_fold(df, fold, config):
    test_start = fold['test_range'][0]
    test_end = min(fold['test_range'][1], df.index.max())
    
    df_slice = df[(df.index >= test_start) & (df.index < test_end)].copy()
    
    if df_slice.empty:
        return None
        
    engine = BacktestEngine(df_slice, config, ml_enabled=False)
    engine.run()
    
    # Calculate Metrics
    final_equity = engine.equity
    initial_equity = engine.initial_capital
    net_return_pct = ((final_equity - initial_equity) / initial_equity) * 100
    
    # Max DD
    eq_curve = pd.DataFrame(engine.equity_curve)
    max_dd = 0.0
    if not eq_curve.empty:
        peaks = eq_curve['equity'].cummax()
        dd = (eq_curve['equity'] - peaks) / peaks
        max_dd = dd.min() * 100 # Negative value
        
    # Trade Stats
    trades = engine.trades
    wins = [t for t in trades if t.pnl > 0]
    loss_abs = [abs(t.pnl) for t in trades if t.pnl <= 0]
    total_win = sum(t.pnl for t in wins)
    total_loss = sum(loss_abs)
    
    pf = total_win / total_loss if total_loss > 0 else (999 if total_win > 0 else 0)
    pf_capped = min(pf, 10.0) # Cap for score calc
    
    win_rate = (len(wins) / len(trades) * 100) if trades else 0
    avg_trade = np.mean([t.pnl for t in trades]) if trades else 0
    
    # MAR (CAGR equivalent / MaxDD) -> Here simplified as Return / MaxDD
    # Avoid zero division
    mar = net_return_pct / abs(max_dd) if max_dd < 0 else (net_return_pct if net_return_pct > 0 else 0)
    
    return {
        'Return': net_return_pct,
        'PF': pf,
        'PF_Capped': pf_capped,
        'MaxDD': max_dd,
        'Trades': len(trades),
        'WinRate': win_rate,
        'AvgTrade': avg_trade,
        'MAR': mar
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, default="2023-01-01")
    parser.add_argument('--end', type=str, default="2024-06-30")
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    
    base_config = load_config()
    
    # 1. Data Prep
    start_date = args.start
    end_date = args.end
    
    fetcher = BinanceDataFetcher()
    cleaner = DataCleaner()
    symbol = base_config['data']['symbol']
    timeframe = base_config['data']['timeframe']
    
    data_filename = f"data/storage/{symbol.replace('/','')}_{timeframe}_{start_date.replace(' ','_').replace(':','')}_{end_date.replace(' ','_').replace(':','')}.csv"
    
    if args.download or not os.path.exists(data_filename):
        print("Downloading data...")
        ccxt_symbol = symbol.replace('USDT', '/USDT') if '/' not in symbol else symbol
        df = fetcher.fetch_ohlcv(ccxt_symbol, timeframe, start_date, end_date)
        df = cleaner.clean_and_validate(df, timeframe)
        os.makedirs("data/storage", exist_ok=True)
        df.to_csv(data_filename)
    else:
        print(f"Using local data: {data_filename}")
        df = pd.read_csv(data_filename)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    
    data_hash = get_data_hash(df)
    print(f"Data Hash (SHA256): {data_hash}")
    
    folds = generate_fold_ranges(df, base_config)
    print(f"Generated {len(folds)} folds.")

    # 2. Define Parameter Grid
    grid = {
        'donchian_period': [34, 55, 89],
        'stop_loss_atr_multiplier': [2.0, 2.5, 3.0],
        'stop_cooldown_bars': [6, 12, 18],
        'trailing_stop_active': [True, False]
    }
    
    keys, values = zip(*grid.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Testing {len(permutations)} combinations across {len(folds)} folds...")
    print(f"Total Backtests: {len(permutations) * len(folds)}")
    
    all_fold_results = []
    
    for i, params in enumerate(permutations):
        combo_id = i
        print(f"Running Combo {i+1}/{len(permutations)}: {params}")
        
        # Override Config
        run_config = base_config.copy() # Shallow copy enough here? 
        import copy
        run_config = copy.deepcopy(base_config)
        
        run_config['strategy'].update(params)
        run_config['ml']['enabled'] = False # ML OFF
        
        fold_metrics_list = []
        
        for fold in folds:
            metrics = run_backtest_for_fold(df, fold, run_config)
            if metrics:
                record = metrics.copy()
                record['ComboID'] = combo_id
                record['Fold'] = fold['fold']
                record.update(params)
                all_fold_results.append(record)
                fold_metrics_list.append(metrics)
                
    # 3. Aggregation & Scoring
    print("\nCalculating Robustness Scores...")
    df_res = pd.DataFrame(all_fold_results)
    
    # Save Raw
    df_res.to_csv("robustness_all_results.csv", index=False)
    
    summary_rows = []
    
    for i, params in enumerate(permutations):
        sub = df_res[df_res['ComboID'] == i]
        if sub.empty:
            continue
            
        med_ret = sub['Return'].median()
        med_mar = sub['MAR'].median()
        med_pf = sub['PF_Capped'].median()
        med_trades = sub['Trades'].median()
        
        std_ret = sub['Return'].std()
        std_mar = sub['MAR'].std()
        
        # Penalties
        stability_penalty = std_ret + 0.5 * std_mar
        trade_penalty = max(0, 6 - med_trades) * 0.1
        
        # Score Formula
        score = (0.45 * med_mar) + (0.35 * med_pf) + (0.20 * med_ret) - stability_penalty - trade_penalty
        
        row = {
            'ComboID': i,
            'Score': round(score, 4),
            'Med_Ret': round(med_ret, 2),
            'Med_PF': round(med_pf, 2),
            'Med_MAR': round(med_mar, 2),
            'Med_Trades': round(med_trades, 1),
            'Std_Ret': round(std_ret, 2),
            'Params': str(params)
        }
        summary_rows.append(row)
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary.sort_values('Score', ascending=False, inplace=True)
    
    # Save Summary
    df_summary.to_csv("robustness_summary.csv", index=False)
    
    print("\n>>> TOP 5 ROBUST CONFIGS <<<")
    print(df_summary.head(5).to_string(index=False))
    
    best_combo = df_summary.iloc[0]
    print(f"\nBest Config (ID {best_combo['ComboID']}): {permutations[int(best_combo['ComboID'])]}")

if __name__ == '__main__':
    main()
