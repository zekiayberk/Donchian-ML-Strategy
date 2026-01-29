
import argparse
import yaml
import pandas as pd
from datetime import datetime
from data.fetcher import BinanceDataFetcher
from data.cleaner import DataCleaner
from optimization.wfo import WalkForwardOptimizer
import matplotlib.pyplot as plt

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Walk Forward Optimization')
    parser.add_argument('--symbol', type=str, default='BTC/USDT')
    parser.add_argument('--tf', type=str, default='1h')
    parser.add_argument('--start', type=str, default='2023-01-01')
    parser.add_argument('--end', type=str, default='2024-01-01')
    args = parser.parse_args()
    
    config = load_config()
    
    # Grid
    param_grid = {
        'donchian_period': [20, 55],
        'stop_loss_atr_multiplier': [2.0, 2.5, 3.0]
    }
    
    # Data fetch logic (similar to backtest)
    print("Veri hazırlanıyor...")
    # ... (Basitlik için doğrudan fetcher kullanalım)
    fetcher = BinanceDataFetcher(market_type='future')
    df = fetcher.fetch_ohlcv(args.symbol, args.tf, args.start, args.end)
    cleaner = DataCleaner()
    df = cleaner.clean_and_validate(df, args.tf)
    
    optimizer = WalkForwardOptimizer(df, config, param_grid, train_period_days=90, test_period_days=30)
    all_trades, log = optimizer.run()
    
    if not all_trades.empty:
        print("\n--- WFO Sonuçları ---")
        final_equity = all_trades['equity'].iloc[-1]
        print(f"Final Equity (OOS): {final_equity:.2f}")
        print(f"Toplam Trade: {len(all_trades)}")
        
        # Plot
        plt.figure(figsize=(10,6))
        plt.plot(all_trades['exit_time'], all_trades['equity'])
        plt.title('Walk-Forward Optimization Equity Curve')
        plt.savefig('wfo_result.png')
        print("Grafik kaydedildi: wfo_result.png")
        
        # Log kaydet
        pd.DataFrame(log).to_csv('wfo_log.csv', index=False)
        all_trades.to_csv('wfo_trades.csv', index=False)
    else:
        print("Hiç trade oluşmadı.")

if __name__ == "__main__":
    main()
