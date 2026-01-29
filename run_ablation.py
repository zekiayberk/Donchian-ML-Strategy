
import pandas as pd
import numpy as np
import yaml
import os
import argparse
from backtest.engine import BacktestEngine
from data.fetcher import BinanceDataFetcher
from data.cleaner import DataCleaner

def load_config(config_path='config.yaml'):
    with open(config_path) as f:
        return yaml.safe_load(f)

def generate_fold_ranges(df, config):
    """
    Generates the exact same date ranges as src/ml/train_wfo.py
    without running the heavy training process.
    """
    timestamps = df.index
    start_date = timestamps.min()
    end_date = timestamps.max()
    
    train_days = config['ml'].get('train_days', 180)
    valid_days = config['ml'].get('valid_days', 30)
    test_days = config['ml'].get('test_days', 30)
    
    folds = []
    current_test_start = start_date + pd.Timedelta(days=train_days + valid_days)
    
    fold_idx = 1
    while current_test_start < end_date:
        test_end = current_test_start + pd.Timedelta(days=test_days)
        valid_start = current_test_start - pd.Timedelta(days=valid_days)
        train_start = valid_start - pd.Timedelta(days=train_days)
        
        # Mask check (simplified from train_wfo logic just to ensure we step correctly)
        # In train_wfo there is a check for min samples, we assume data is sufficient here if it worked for WFO
        # But to be exact, we should mimic the check if possible, or just assume the dates are key.
        # Given we want to compare with previous run, using the same date logic is crucial.
        
        # NOTE: train_wfo logic has a continue if not enough data. 
        # We will iterate same way.
        
        train_mask = (timestamps >= train_start) & (timestamps < valid_start)
        valid_mask = (timestamps >= valid_start) & (timestamps < current_test_start)
        
        if train_mask.sum() < 20 or valid_mask.sum() < 5:
             current_test_start += pd.Timedelta(days=test_days)
             continue
             
        folds.append({
            'fold': fold_idx,
            'test_range': (current_test_start, test_end)
        })
        
        fold_idx += 1
        current_test_start += pd.Timedelta(days=test_days)
        
    return folds

def run_condition(df, folds, condition_name, config_overrides, base_config):
    print(f"\n>>> Running Condition: {condition_name} <<<")
    
    # Deep copy config to avoid pollution
    import copy
    run_config = copy.deepcopy(base_config)
    
    # Apply Overrides
    for key, val in config_overrides.items():
        # Handle nested keys if needed (simple implementation for now)
        if '.' in key:
            k1, k2 = key.split('.')
            run_config[k1][k2] = val
        else:
            # Assuming top level or direct mapping not needed for these specific ones
            pass 
            
    # Specific Logic for our known keys
    # ML is always OFF
    run_config['ml']['enabled'] = False
    
    if condition_name == 'A_Base':
        run_config['strategy']['adx_filter_active'] = False
        run_config['strategy']['breakout_strength_threshold'] = 0.0
    elif condition_name == 'B_ADX':
        run_config['strategy']['adx_filter_active'] = True
        run_config['strategy']['adx_threshold'] = 20
        run_config['strategy']['breakout_strength_threshold'] = 0.0
    elif condition_name == 'C_Strength':
        run_config['strategy']['adx_filter_active'] = False
        run_config['strategy']['breakout_strength_threshold'] = 0.2
        
    results = []
    
    for fold in folds:
        fold_id = fold['fold']
        test_start = fold['test_range'][0]
        test_end = min(fold['test_range'][1], df.index.max()) # Clamp to data end
        
        # Slice Data
        df_slice = df[(df.index >= test_start) & (df.index < test_end)].copy()
        
        if df_slice.empty:
            continue
            
        # Run Backtest
        engine = BacktestEngine(df_slice, run_config, ml_enabled=False)
        engine.run()
        
        # Calculate Metrics
        final_equity = engine.equity
        initial_equity = engine.initial_capital
        net_return_pct = ((final_equity - initial_equity) / initial_equity) * 100
        
        # Max DD from engine.equity_curve
        eq_curve = pd.DataFrame(engine.equity_curve)
        max_dd = 0.0
        if not eq_curve.empty:
            peaks = eq_curve['equity'].cummax()
            dd = (eq_curve['equity'] - peaks) / peaks
            max_dd = dd.min() * 100
            
        # Stats
        trades = engine.trades
        wins = [t for t in trades if t.pnl > 0]
        loss_abs = [abs(t.pnl) for t in trades if t.pnl <= 0]
        total_win = sum(t.pnl for t in wins)
        total_loss = sum(loss_abs)
        pf = total_win / total_loss if total_loss > 0 else (999 if total_win > 0 else 0)
        win_rate = (len(wins) / len(trades) * 100) if trades else 0
        
        avg_trade = np.mean([t.pnl for t in trades]) if trades else 0
        
        results.append({
            'Condition': condition_name,
            'Fold': fold_id,
            'Start': test_start,
            'End': test_end,
            'Return': round(net_return_pct, 2),
            'PF': round(pf, 2),
            'MaxDD': round(max_dd, 2),
            'Trades': len(trades),
            'WinRate': round(win_rate, 2),
            'AvgTrade': round(avg_trade, 2)
        })
        
    return results

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
    
    fetcher = BinanceDataFetcher()
    cleaner = DataCleaner()
    symbol = config['data']['symbol']
    timeframe = config['data']['timeframe']
    
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
        
    # 2. Generate Date Ranges (Same as WFO)
    folds = generate_fold_ranges(df, config)
    print(f"Generated {len(folds)} folds (WFO replicator)")
    
    # 3. Run Conditions
    all_results = []
    
    # Condition A: Base
    res_a = run_condition(df, folds, 'A_Base', {}, config)
    all_results.extend(res_a)
    
    # Condition B: +ADX
    res_b = run_condition(df, folds, 'B_ADX', {}, config)
    all_results.extend(res_b)
    
    # Condition C: +Strength
    res_c = run_condition(df, folds, 'C_Strength', {}, config)
    all_results.extend(res_c)
    
    # 4. Save & Report
    res_df = pd.DataFrame(all_results)
    res_df.to_csv("ablation_results.csv", index=False)
    print("\nResults saved to ablation_results.csv")
    
    # Simple Console Summary
    summary = res_df.groupby('Condition').agg({
        'Return': 'median',
        'PF': 'median',
        'MaxDD': 'median',
        'Trades': 'median',
        'WinRate': 'median'
    })
    print("\n>>> MEDIAN SUMMARY <<<")
    print(summary)

if __name__ == '__main__':
    main()
