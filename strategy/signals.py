
import pandas as pd
import numpy as np

from indicators.filters import calculate_adx

class SignalGenerator:
    """
    Ham fiyat verisi ve indikatörler üzerinden potansiyel giriş sinyallerini üretir.
    """
    
    @staticmethod
    def generate_signals(df, config=None):
        """
        Donchian Breakout sinyalleri üretir (1: Long, -1: Short, 0: Yok).
        """
        df = df.copy()
        
        # Filtre Parametreleri
        adx_active = config['strategy']['adx_filter_active'] if config else False
        adx_thresh = config['strategy']['adx_threshold'] if config else 20
        strength_thresh = config['strategy']['breakout_strength_threshold'] if config else 0.0
        
        # ADX Hesapla (Eğer yoksa)
        if adx_active and 'adx' not in df.columns:
            df = calculate_adx(df)
        
        # Signal default 0
        df['entry_signal'] = 0
        
        # -- 1. Temel Breakout --
        # Long: Close > Donchian Upper
        long_cond = df['close'] > df['donchian_upper']
        # Short: Close < Donchian Lower
        short_cond = df['close'] < df['donchian_lower']
        
        # -- 2. Breakout Strength (Güçlü Kırılım) --
        # Long: (Close - Upper) / ATR > Thresh
        if strength_thresh > 0:
            atr = df['atr'] # ATR yoksa hata verir, engine'de hesaplanmış olmalı
            long_strength = (df['close'] - df['donchian_upper']) / atr
            short_strength = (df['donchian_lower'] - df['close']) / atr
            
            long_cond = long_cond & (long_strength > strength_thresh)
            short_cond = short_cond & (short_strength > strength_thresh)
            
        # -- 3. ADX Filtresi --
        if adx_active:
            adx_cond = df['adx'] > adx_thresh
            long_cond = long_cond & adx_cond
            short_cond = short_cond & adx_cond
        
        df.loc[long_cond, 'entry_signal'] = 1
        df.loc[short_cond, 'entry_signal'] = -1
        
        return df
