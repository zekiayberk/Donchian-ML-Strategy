
import time
import schedule
import yaml
import os
import sys
from live.brokers import PaperBroker, BinanceFuturesBroker
from data.fetcher import BinanceDataFetcher
from indicators.donchian import calculate_donchian_channel
from indicators.atr import calculate_atr
from live.state import StateManager
from live.execution import ExecutionEngine
import pandas as pd

# Config Yükle
config_path = 'config.yaml'
if not os.path.exists(config_path):
    print("HATA: config.yaml bulunamadı.")
    sys.exit(1)

with open(config_path) as f:
    config = yaml.safe_load(f)

SYMBOL = config['data']['symbol']
TF = config['data']['timeframe']
IS_PAPER = True # Paper mode varsayılan

# Initialize Modules
if IS_PAPER:
    broker = PaperBroker()
    print("MOD: PAPER TRADING")
else:
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    broker = BinanceFuturesBroker(api_key, api_secret)
    print("MOD: LIVE TRADING")

state_manager = StateManager()
execution_engine = ExecutionEngine(broker, state_manager, config)
# Canlı veri için fetcher (veya broker üzerinden get_klines)
fetcher = BinanceDataFetcher(market_type='future')

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def job():
    try:
        log(f"Tick çalışıyor... {SYMBOL}")
        
        # 1. Market Verisi Al (Son 100 bar)
        # Fetcher'ın fetch_ohlcv metodu tarih aralığı istiyor, canlı bot için son N bar almak daha pratik.
        # Bu yüzden direkt ccxt çağrısı veya klines endpoint daha iyi.
        # Burada fetcher'ın ccxt wrapper'ını kullanacağız.
        
        ohlcv = fetcher.exchange.fetch_ohlcv(SYMBOL, TF, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # 2. İndikatörleri Hesapla
        df = calculate_donchian_channel(df, config['strategy']['donchian_period'])
        df = calculate_atr(df, config['strategy']['atr_period'])
        
        # 3. Son 2 barı al (Tamamlanmış bar [-2] ve güncel bar [-1])
        last_complete_bar = df.iloc[-2]
        current_bar = df.iloc[-1]
        
        close = last_complete_bar['close']
        upper = last_complete_bar['donchian_upper']
        lower = last_complete_bar['donchian_lower']
        atr = last_complete_bar['atr']
        
        current_price = current_bar['close'] # Anlık fiyat
        
        log(f"Fiyat: {current_price} | Donchian: [{lower:.2f}, {upper:.2f}] | ATR: {atr:.2f}")
        
        # 4. Sinyal Kontrolü (Tamamlanmış bar kapanışına göre)
        signal = 0
        if close > upper:
            signal = 1
        elif close < lower:
            signal = -1
            
        # 5. Yürütme (Execution)
        # Önce stop kontrolü (her tick, yani her job döngüsü)
        execution_engine.check_stop_loss(current_price)
        
        # Sinyal varsa işlem yap (Yeni bar açılışında çalışır diye varsayıyoruz bu job'u)
        # Not: Gerçekte bu job her dakika çalışıyorsa, sadece saat başı çalışacak şekilde ayarlanmalı
        # veya "bar yeni kapandı mı?" kontrolü yapılmalı.
        # Basitlik için her döngüde sinyal kontrol edip state ile position yönetiyoruz.
        
        if signal != 0:
            log(f"Sinyal Tespit Edildi: {signal}")
            execution_engine.execute_signal(signal, current_price, atr)
            
    except Exception as e:
        log(f"HATA: {e}")

# Zamanlama
log("Bot döngüsü başlatılıyor. (Çıkış için Ctrl+C)")

# Test için her 10 saniye. Gerçekte timeframe ne ise o olmalı (örn: every().hour.at(":01"))
schedule.every(10).seconds.do(job)

if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(1)
