
import pandas as pd
import numpy as np
import yaml
import os
import argparse
import hashlib
import glob
from backtest.engine import BacktestEngine
from data.fetcher import BinanceDataFetcher
from data.cleaner import DataCleaner
from src.utils.wfo_utils import generate_fold_ranges, get_data_hash

def load_config(config_path='config.yaml'):
    with open(config_path) as f:
        return yaml.safe_load(f)

def get_model_hash(models_dir="models"):
    """
    Computes a hash of all model/meta files to ensure we are testing the exact same models.
    """
    files = sorted(glob.glob(os.path.join(models_dir, "*")))
    hasher = hashlib.sha256()
    for fpath in files:
        if os.path.isfile(fpath):
            with open(fpath, 'rb') as f:
                content = f.read()
                hasher.update(content)
                hasher.update(os.path.basename(fpath).encode())
    return hasher.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, default="2023-01-01")
    parser.add_argument('--end', type=str, default="2024-06-30")
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    
    config = load_config()
    
    # 1. Data Prep
    start_date = args.start
    end_date = args.end
    symbol = config['data']['symbol']
    timeframe = config['data']['timeframe']
    
    data_filename = f"data/storage/{symbol.replace('/','')}_{timeframe}_{start_date.replace(' ','_').replace(':','')}_{end_date.replace(' ','_').replace(':','')}.csv"
    
    if args.download or not os.path.exists(data_filename):
        print("Downloading data...")
        fetcher = BinanceDataFetcher()
        cleaner = DataCleaner()
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
        
    # LOG HASHES
    d_hash = get_data_hash(df)
    m_hash = get_model_hash("models")
    print(f"Data Hash:  {d_hash}")
    print(f"Model Hash: {m_hash}")
    
    folds = generate_fold_ranges(df, config)
    print(f"Generated {len(folds)} folds.")

    # 2. Sweep Parameters
    threshold_offsets = [0.00, 0.05, 0.10, 0.15, 0.20]
    slippage_scenarios = [2, 5, 10]
    
    results = []
    
    import copy
    
    for offset in threshold_offsets:
        for slip in slippage_scenarios:
            print(f"\n>> Running Sweep: Offset={offset}, Slippage={slip} bps")
            
            # Setup Config
            run_config = copy.deepcopy(config)
            run_config['ml']['enabled'] = True
            run_config['backtest']['slippage_bps'] = slip
            
            fold_results = []
            
            for fold in folds:
                test_start = fold['test_range'][0]
                test_end = min(fold['test_range'][1], df.index.max())
                
                df_slice = df[(df.index >= test_start) & (df.index < test_end)].copy()
                if df_slice.empty: continue
                
                # Engine Run
                engine = BacktestEngine(df_slice, run_config, ml_enabled=True, threshold_offset=offset)
                engine.run()
                
                # Metrics
                final_equity = engine.equity
                net_return = ((final_equity - engine.initial_capital) / engine.initial_capital) * 100
                
                # MaxDD
                eq_curve = pd.DataFrame(engine.equity_curve)
                max_dd = 0.0
                if not eq_curve.empty:
                    peaks = eq_curve['equity'].cummax()
                    if peaks.max() > 0:
                        dd = (eq_curve['equity'] - peaks) / peaks
                        max_dd = dd.min() * 100
                        
                # Stats
                not_exec = engine.ml_allowed - engine.entry_opened
                exec_blocked_reason = "N/A"
                if engine.entry_block_reasons:
                    exec_blocked_reason = str(dict(engine.entry_block_reasons))
                
                # Take Rate
                # engine.stats['signals_total_in_oos'] is total evaluations
                # engine.ml_allowed is accepted
                total_eval = engine.ml_evaluated if hasattr(engine, 'ml_evaluated') else 0
                # Fallback if ml_evaluated not explicitly tracked (engine update needed?)
                # We can assume ml_allowed + ml_skipped (from engine stats) approximately equals total evaluated?
                # Let's rely on engine.stats if possible or add tracking. 
                # In current engine:
                # self.ml_evaluated is NOT explicitly in __init__, but 'ml_allowed' is.
                # 'ml_skipped' is in loop.
                # So total = ml_allowed + ml_skipped
                
                total_signals = engine.ml_allowed + engine.ml_skipped
                take_rate = (engine.ml_allowed / total_signals * 100) if total_signals > 0 else 0
                
                res = {
                    'Offset': offset,
                    'Slippage': slip,
                    'Fold': fold['fold'],
                    'Return': net_return,
                    'MaxDD': max_dd,
                    'Trades': len(engine.trades),
                    'TakeRate': take_rate,
                    'NotExecuted': not_exec,
                    'EndDataBlock': engine.entry_block_reasons.get('END_OF_DATA', 0)
                }
                fold_results.append(res)
                
            # Aggregate for this combo
            df_fold = pd.DataFrame(fold_results)
            mean_ret = df_fold['Return'].mean()
            mean_dd = df_fold['MaxDD'].mean()
            total_trades = df_fold['Trades'].sum()
            avg_take = df_fold['TakeRate'].mean()
            
            results.extend(fold_results)
            print(f"   -> Mean Ret: {mean_ret:.2f}%, Mean DD: {mean_dd:.2f}%, Tot Trades: {total_trades}")

    # Save Results
    df_all = pd.DataFrame(results)
    df_all.to_csv("threshold_sweep_results.csv", index=False)
    print("\nSaved threshold_sweep_results.csv")
    
    # Pivot Summary
    pivot = df_all.pivot_table(index='Offset', columns='Slippage', values='Return', aggfunc='mean')
    print("\n>>> Return Matrix (Offset vs Slippage) <<<")
    print(pivot)

if __name__ == '__main__':
    main()
