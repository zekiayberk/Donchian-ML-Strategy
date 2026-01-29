
import pandas as pd
import numpy as np
from indicators.filters import calculate_adx

def engineering_features(df):
    """
    ML için gerekli feature'ları üretir.
    DİKKAT: Tüm feature'lar T anında bilinen veriler olmalı. Lookahead yok.
    """
    df = df.copy()
    
    # 1. Temel Göstergeler (Eğer yoksa hesapla)
    if 'atr' not in df.columns:
        raise ValueError("ATR hesaplanmamış.")
    if 'donchian_upper' not in df.columns:
        raise ValueError("Donchian hesaplanmamış.")
        
    close = df['close']
    atr = df['atr']
    
    # 2. ADX
    if 'adx' not in df.columns:
        df = calculate_adx(df) # Mevcut filters.py'dan
        
    # 3. EMA'lar
    df['ema50'] = close.ewm(span=50).mean()
    df['ema200'] = close.ewm(span=200).mean()
    
    # 4. Feature Extraction
    
    # -- Breakout Strength --
    # Upper/Lower mesafesi (normalized by ATR)
    # Long sinyali için: (Close - Upper) / ATR
    # Short sinyali için: (Lower - Close) / ATR
    # Genel bir "Strength" feature'ı için Distance from Mid channel kullanılabilir
    # Ama entry-spesifik feature daha iyi. ML modeline direction da vereceğiz.
    
    # Donchian Width Pct
    df['donchian_width'] = (df['donchian_upper'] - df['donchian_lower']) / close
    
    # Normalized ATR
    df['atrp'] = atr / close
    
    # Trend Durumu
    df['trend_ema'] = (df['ema50'] > df['ema200']).astype(int)
    # Slope (Eğim) of EMA50 (10 barlık değişim)
    df['ema50_slope'] = (df['ema50'] - df['ema50'].shift(10)) / df['ema50'].shift(10)
    
    # Returns (Momentum)
    for w in [1, 3, 6, 12, 24, 48]:
        df[f'roc_{w}'] = df['close'].pct_change(w)
        
    # Volatility (Std of Returns)
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()
    
    # Chop Index Proxy: Range to ATR
    # Son 20 barın High-Low range'i / ATR toplamı?
    # Veya basitçe: (MaxHigh20 - MinLow20) / Sum(TR20) -> Klasik Chop Index formülü benzeri
    # Biz basit versiyonunu yapalım: (RollingHigh - RollingLow) / (20 * ATR) (Normalized)
    roll_h = df['high'].rolling(20).max()
    roll_l = df['low'].rolling(20).min()
    df['chop_proxy'] = (roll_h - roll_l) / (20 * df['atr'])
    
    # Sign Flips (Testere ölçer)
    # Son 20 barda close - open işareti kaç kere değişti?
    diff = df['close'] - df['open']
    sign = np.sign(diff)
    # sign shift değişimi
    sign_change = (sign != sign.shift(1)).astype(int)
    df['sign_flips_20'] = sign_change.rolling(20).sum()
    
    # Breakout Strength (Ham, sinyal yönüne göre dataset.py'da seçilecek)
    # Şimdilik "Close'un Upper'a uzaklığı" ve "Lower'a uzaklığı"
    df['dist_upper_atr'] = (df['close'] - df['donchian_upper']) / atr
    df['dist_lower_atr'] = (df['donchian_lower'] - df['close']) / atr # Short için pozitif olması için ters çevirdik
    
    return df

def get_feature_row_at(df, idx, direction):
    """
    Belirli bir indeks (T) ve yön (Long/Short) için ML feature satırını hazırlar.
    Bu fonksiyon, DataFrame önceden engineering_features ile işlendiyse hızlı çalışır.
    """
    row = df.iloc[idx].copy()
    
    # Sinyal Yönüne Özel Hesaplamalar (Breakout Strength)
    if 'breakout_strength' not in row:
        row['breakout_strength'] = _calculate_breakout_strength(df, idx, direction)
        
    # Yön Bilgisi
    row['direction'] = int(direction)
    
    # Feature Engineering'de hesaplanan ancak dinamik ihtiyaç duyulanlar bura eklenebilir.
    return row

def _calculate_breakout_strength(df, idx, direction):
    """
    Sinyal yönüne göre kırılımın gücünü hesaplar.
    """
    # Tekil satır erişimi yavaş olabilir, ancak canlı işlem/backtest için kabul edilebilir.
    # df.iloc[idx] kullanıyoruz.
    c = df['close'].iloc[idx]
    atr = df['atr'].iloc[idx]
    
    if direction == 1: # LONG
        upper = df['donchian_upper'].iloc[idx]
        return (c - upper) / atr if atr > 0 else 0
    elif direction == -1: # SHORT
        lower = df['donchian_lower'].iloc[idx]
        return (lower - c) / atr if atr > 0 else 0
    
    return 0.0
