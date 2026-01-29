
import ccxt
import pandas as pd
import datetime
import os
import time
from pathlib import Path

class BinanceDataFetcher:
    """
    Binance'den geçmiş veri (klines) çeker.
    Spot ve Futures desteği sağlar.
    """
    def __init__(self, exchange_id='binance', market_type='future'):
        """
        :param exchange_id: Borsa ID (varsayılan: binance)
        :param market_type: 'spot' veya 'future' (USD-M futures)
        """
        self.market_type = market_type
        self.exchange = getattr(ccxt, exchange_id)()
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo
            
        self.utc = ZoneInfo("UTC")
        
        if market_type == 'future':
            self.exchange.options['defaultType'] = 'future'
        elif market_type == 'spot':
            self.exchange.options['defaultType'] = 'spot'
            
        self.exchange.load_markets()

    def fetch_ohlcv(self, symbol, timeframe, start_date=None, end_date=None, limit=1000):
        """
        Belirtilen tarih aralığında OHLCV verisini çeker.
        Otomatik olarak parça parça indirir (pagination).
        
        :param symbol: Parite
        :param timeframe: Zaman dilimi
        :param start_date: Başlangıç (None -> Son veriler)
        :param end_date: Bitiş (None -> Şu an)
        :param limit: Limit
        :return: Pandas DataFrame (DatetimeIndex)
        """
        
        # Start Date Handling
        since = None
        if start_date is not None:
            if isinstance(start_date, str):
                since = int(pd.Timestamp(start_date).timestamp() * 1000)
            else:
                since = int(start_date.timestamp() * 1000)
                
        # End Date Handling
        end_ts = int(time.time() * 1000)
        if end_date is not None:
             if isinstance(end_date, str):
                end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
             else:
                end_ts = int(end_date.timestamp() * 1000)

        all_ohlcv = []
        
        # print(f"Veri indiriliyor: {symbol} - {timeframe} | Since: {since}")
        
        if since is None:
            # Fetch Latest (Single Request)
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, None, limit)
                if ohlcv:
                    all_ohlcv.extend(ohlcv)
            except Exception as e:
                print(f"Fetch Error (Latest): {e}")
        else:
            # Pagination Loop
            while since < end_ts:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                    
                    if not ohlcv:
                        break
                    
                    all_ohlcv.extend(ohlcv)
                    
                    # Update pointer
                    last_bar_time = ohlcv[-1][0]
                    since = last_bar_time + 1 
                    
                    if last_bar_time >= end_ts:
                        break
                    
                    time.sleep(self.exchange.rateLimit / 1000)
                    
                except Exception as e:
                    print(f"Fetch Error (Pagination): {e}")
                    time.sleep(5)
                    continue

        if not all_ohlcv:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Datetime Conversion (UTC Aware)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('datetime', inplace=True)
        
        # Precise Filtering
        if start_date is not None:
            # use start_date directly for filtering as 'since' was incremented in the loop
            if isinstance(start_date, str):
                start_ts_val = pd.Timestamp(start_date).tz_localize('UTC') if pd.Timestamp(start_date).tzinfo is None else pd.Timestamp(start_date)
            elif isinstance(start_date, (int, float)):
                 start_ts_val = pd.Timestamp(start_date, unit='ms', tz='UTC')
            else:
                # Assume datetime/Timestamp
                start_ts_val = start_date if hasattr(start_date, 'tzinfo') and start_date.tzinfo else pd.to_datetime(start_date, utc=True)
            
            df = df[df.index >= start_ts_val]
            
        if end_date is not None:
             # end_ts logic can be tricky if exact boundary is needed, usually fetch includes start, excludes end logic
             # but here end_date is usually "until when".
             pass
             
        return df

    def save_to_csv(self, df, filename):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filename, index=True) # Save Index
        print(f"Veri kaydedildi: {filename}")

if __name__ == "__main__":
    fetcher = BinanceDataFetcher(market_type='future')
    df = fetcher.fetch_ohlcv('BTC/USDT', '1h', limit=5)
    print(df.head())
    print("Index Type:", type(df.index))
