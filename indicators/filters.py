
import pandas as pd
import numpy as np

def calculate_adx(df, period=14):
    """
    ADX (Average Directional Index) hesaplar.
    Trend gücünü ölçmek için kullanılır.
    
    :param df: DataFrame
    :param period: 14
    :return: DataFrame (plus_di, minus_di, adx)
    """
    df = df.copy()
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # +DM ve -DM hesapla
    # high - prev_high
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Panda series'e çevir
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    
    # True Range (ATR hesaplamasından alabiliriz ama bağımsız olsun diye tekrar TR lazım)
    # Daha önce hesaplanan TR varsa kullan
    if 'tr' not in df.columns:
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    else:
        tr = df['tr']
        
    # Wilder's Smoothing
    atr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_smooth)
    
    # DX
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    
    # ADX (DX'in smoothed versiyonu)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    df['adx'] = adx
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    
    return df

def calculate_volatility_filter(df, threshold=0.01):
    """
    ATR / Close oranı belirli bir eşiğin üstünde mi?
    Yüksek volatilite zamanlarını seçmek için.
    """
    df = df.copy()
    if 'atr' not in df.columns:
        raise ValueError("ATR hesaplanmamış.")
        
    df['volatility_ratio'] = df['atr'] / df['close']
    df['is_volatile'] = df['volatility_ratio'] > threshold
    return df
