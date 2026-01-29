
import argparse
import yaml
import pandas as pd
import os
from datetime import datetime
from data.fetcher import BinanceDataFetcher
from data.cleaner import DataCleaner
from backtest.engine import BacktestEngine
from backtest.metrics import PerformanceMetrics, print_report
from backtest.report import plot_results

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Donchian Breakout Backtest')
    parser.add_argument('--symbol', type=str, help='Trading pair (e.g., BTC/USDT)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--tf', type=str, help='Timeframe (e.g. 1h, 4h, 1d)')
    parser.add_argument('--ml_on', action='store_true', help='Activate ML Filter')
    parser.add_argument('--threshold_offset', type=float, default=None, help='ML Threshold offset (e.g. 0.05 or -0.05)')
    args = parser.parse_args()

    # Config yükle
    config = load_config()
    
    # Argümanlar config'i ezer
    # Argümanlar config'i ezer
    symbol = args.symbol if args.symbol else config['data']['symbol']
    timeframe = args.tf if args.tf else config['data']['timeframe']
    
    # Tarihleri Timestamp'e çevirerek standardize et
    ts_start = pd.Timestamp(args.start if args.start else config['data']['start_date'])
    ts_end = pd.Timestamp(args.end if args.end else config['data']['end_date'])
    
    start_date = ts_start.isoformat()
    end_date = ts_end.isoformat()
    
    if args.ml_on:
        config['ml']['enabled'] = True
        print(">>> ML MODE ENABLED <<<")
        
    # Determine Threshold Offset
    if args.threshold_offset is not None:
        final_threshold_offset = args.threshold_offset
    else:
        # Config'den okumaya çalış, yoksa 0.0
        final_threshold_offset = config.get('ml', {}).get('threshold_offset', 0.0)
    
    print(f"Başlatılıyor: {symbol} [{timeframe}] {start_date} -> {end_date}")

    # 1. Veri Çekme / Yükleme
    data_dir = config['data']['data_dir']
    os.makedirs(data_dir, exist_ok=True)
    
    # Windows uyumlu dosya ismi ( : ve space yok )
    safe_start = str(ts_start).replace(':', '').replace(' ', '_')
    safe_end = str(ts_end).replace(':', '').replace(' ', '_')
    
    filename = f"{data_dir}/{symbol.replace('/', '')}_{timeframe}_{safe_start}_{safe_end}.csv"
    
    if os.path.exists(filename):
        print(f"Yerel veri kullanılıyor: {filename}")
        df = pd.read_csv(filename)
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        print("Veri indiriliyor...")
        fetcher = BinanceDataFetcher(market_type='future') # Varsayılan future
        # ccxt sembol formatı: BTC/USDT
        ccxt_symbol = symbol.replace('USDT', '/USDT') if '/' not in symbol else symbol
        
        # Fetcher'a ISO format gönder
        df = fetcher.fetch_ohlcv(ccxt_symbol, timeframe, start_date, end_date)
        fetcher.save_to_csv(df, filename)

    # 2. Veri Temizleme
    cleaner = DataCleaner()
    df = cleaner.clean_and_validate(df, timeframe)
    
    # 2.5 Tarih Aralığı Filtreleme (Kesin Çözüm)
    # Kullanıcı --start ve --end ile ne verdiyse KESİN olarak o aralığı kullan.
    # CSV'de fazla veri olsa bile kırpılır.
    ts_start = pd.Timestamp(start_date)
    ts_end = pd.Timestamp(end_date)
    
    # Eğer kullanıcı saat vermediyse ve gün sonunu kastediyorsa, pd.Timestamp bunu başta 00:00 alır.
    # Ancak genelde start = 00:00, end = 00:00 (ertesi gün başı) veya end 23:59 beklenir.
    # Şimdilik kullanıcı tam string ne verdiyse o.
    
    df = df[(df.index >= ts_start) & (df.index <= ts_end)].copy()
    print(f"Veri Filtrelendi: {df.index.min()} -> {df.index.max()} ({len(df)} bar)")
    
    # 3. Backtest Çalıştırma
    print(f"Backtest motoru çalışıyor... (ML Offset: {final_threshold_offset:+})")
    engine = BacktestEngine(df, config, ml_enabled=config['ml']['enabled'], 
                          threshold_offset=final_threshold_offset)
    engine.run()
    
    # 4. Sonuçlar
    trades_df, equity_df = engine.get_results()
    
    metrics = PerformanceMetrics.calculate(trades_df, equity_df, config['backtest']['initial_capital'])
    print_report(metrics, trades_df)
    
    # 5. Raporlama
    if not equity_df.empty:
        plot_results(equity_df, trades_df, filename=f"results_{symbol}_{timeframe}.png")
        trades_df.to_csv(f"trades_{symbol}_{timeframe}.csv", index=False)
        
        # Metrics JSON kaydet
        import json
        with open(f"metrics_{symbol}_{timeframe}.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
        print(f"Sonuçlar kaydedildi: trades_{symbol}_{timeframe}.csv, metrics_{symbol}_{timeframe}.json")
        
        # Blocked Signals CSV Export
        if hasattr(engine, 'blocked_events') and engine.blocked_events:
            blocked_df = pd.DataFrame(engine.blocked_events)
            blocked_filename = f"blocked_{symbol}_{timeframe}.csv"
            blocked_df.to_csv(blocked_filename, index=False)
            print(f"Bloke Edilen İşlemler Kaydedildi: {blocked_filename} (Sayısı: {len(blocked_df)})")

if __name__ == "__main__":
    main()
