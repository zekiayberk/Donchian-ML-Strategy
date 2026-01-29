
import argparse
import pandas as pd
import os
import yaml
from data.fetcher import BinanceDataFetcher
from data.cleaner import DataCleaner
from backtest.engine import BacktestEngine
from backtest.metrics import PerformanceMetrics

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_analysis(symbol, start_date, end_date, timeframe, offsets):
    config = load_config()
    
    # 1. Veri Hazırlığı (Tek seferlik)
    print(f"Veri Hazırlanıyor: {symbol} {timeframe} ({start_date} -> {end_date})")
    
    # Tarihleri Timestamp'e çevir
    ts_start = pd.Timestamp(start_date)
    ts_end = pd.Timestamp(end_date)
    
    # Fetcher/Filename uyumu için string formatlar
    str_start = ts_start.isoformat()
    str_end = ts_end.isoformat()
    
    data_dir = config['data']['data_dir']
    os.makedirs(data_dir, exist_ok=True)
    
    # Windows uyumlu dosya ismi
    safe_start = str(ts_start).replace(':', '').replace(' ', '_')
    safe_end = str(ts_end).replace(':', '').replace(' ', '_')
    filename = f"{data_dir}/{symbol.replace('/', '')}_{timeframe}_{safe_start}_{safe_end}.csv"
    
    if os.path.exists(filename):
        print(f"Yerel veri: {filename}")
        df_raw = pd.read_csv(filename)
        df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
    else:
        print("İndiriliyor...")
        fetcher = BinanceDataFetcher(market_type='future')
        ccxt_symbol = symbol.replace('USDT', '/USDT') if '/' not in symbol else symbol
        df_raw = fetcher.fetch_ohlcv(ccxt_symbol, timeframe, str_start, str_end)
        fetcher.save_to_csv(df_raw, filename)
        
    cleaner = DataCleaner()
    df_clean = cleaner.clean_and_validate(df_raw, timeframe)
    
    # Strict Filtering
    df_main = df_clean[(df_clean.index >= ts_start) & (df_clean.index <= ts_end)].copy()
    print(f"Analiz Verisi: {len(df_main)} bar\n")
    
    # 2. Döngü
    results = []
    
    print(f"{'OFFSET':<8} | {'TRADES':<6} | {'RETURN %':<10} | {'PF':<6} | {'MaxDD %':<8} | {'WIN %':<6}")
    print("-" * 65)
    
    for offset in offsets:
        # ML Engine her seferinde yeniden init edilmeli mi? 
        # BacktestEngine içinde ml_engine lazy load ise sorun yok.
        # Ancak threshold_offset init parametresi.
        
        engine = BacktestEngine(df_main.copy(), config, ml_enabled=True, threshold_offset=offset)
        
        # Log karmaşasını önlemek için stdout'u kısabiliriz ama şimdilik gerek yok
        # sessiz modda çalışması için engine koduna dokunmak lazım, şimdilik varsayılan.
        engine.run()
        
        trades, equity = engine.get_results()
        metrics = PerformanceMetrics.calculate(trades, equity, config['backtest']['initial_capital'])
        
        row = {
            'Offset': offset,
            'Trades': metrics.get('Total Trades', 0),
            'Return': metrics.get('Total Return (%)', 0.0),
            'PF': metrics.get('Profit Factor', 0.0),
            'MaxDD': metrics.get('Max Drawdown (%)', 0.0),
            'WinRate': metrics.get('Win Rate (%)', 0.0)
        }
        results.append(row)
        
        print(f"{offset:<8.2f} | {row['Trades']:<6} | {row['Return']:<10.2f} | {row['PF']:<6.2f} | {row['MaxDD']:<8.2f} | {row['WinRate']:<6.2f}")

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description='ML Sensitivity Analysis')
    parser.add_argument('--symbol', type=str, default='BTC/USDT')
    parser.add_argument('--start', type=str, required=True)
    parser.add_argument('--end', type=str, required=True)
    parser.add_argument('--tf', type=str, default='1h')
    parser.add_argument('--ml_on', action='store_true') # Sadece uyumluluk için, script zaten ML zorlar
    
    args = parser.parse_args()
    
    # Test edilecek offsetler (Fine Tuning)
    # Base 0.0.
    offsets = [0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20]
    
    print(">>> SENSITIVITY ANALYSIS STARTED <<<")
    df_res = run_analysis(args.symbol, args.start, args.end, args.tf, offsets)
    
    print("\n>>> ANALYSIS COMPLETE <<<")
    print(df_res)
    
    # En iyi PF'ye göre sırala
    best = df_res.sort_values(by='PF', ascending=False).iloc[0]
    print(f"\nBest Config by PF: Offset {best['Offset']} (PF: {best['PF']:.2f}, Ret: {best['Return']:.2f}%)")

if __name__ == "__main__":
    main()
