
import pandas as pd
import numpy as np

def calculate_atr(df, period=14):
    """
    Wilder's ATR (Average True Range) hesaplar.
    
    :param df: DataFrame (high, low, close)
    :param period: Periyot (varsayılan 14)
    :return: DataFrame (tr, atr eklenmiş)
    """
    df = df.copy()
    
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    
    # True Range Hesaplama
    # 1. High - Low
    # 2. |High - PrevClose|
    # 3. |Low - PrevClose|
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    # Her bar için bu üçünün maksimumunu al
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Wilder's Smoothing (RMA - Running Moving Average)
    # Pandas ewm com = period - 1 formülü alpha = 1/period'a denktir?
    # Wilder's Smoothing: alpha = 1/n
    # Pandas ewm(alpha=1/n, adjust=False) tam olarak Wilder's smoothing'dir.
    
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    # Alternatif (Daha geleneksel döngüsel yaklaşım, ilk değer SMA ile başlar):
    # Ancak Pandas ewm oldukça yakındır ve vektörizedir.
    
    df['tr'] = tr
    df['atr'] = atr
    
    return df
