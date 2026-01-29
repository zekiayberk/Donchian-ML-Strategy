
import pandas as pd
import numpy as np
from .labels import generate_labels
from .features import engineering_features
from strategy.signals import SignalGenerator
from indicators.donchian import calculate_donchian_channel
from indicators.atr import calculate_atr

def build_dataset(raw_df, config):
    """
    Ham veriden ML için X (Features) ve y (Labels) oluşturur.
    Sadece sinyal oluşan barları filtreler.
    """
    df = raw_df.copy()
    
    # 1. Göstergeler (Eğer yoksa)
    # (Genelde backtest öncesi hesaplanır ama garanti olsun)
    if 'donchian_upper' not in df.columns:
        df = calculate_donchian_channel(df, config['strategy']['donchian_period'])
    if 'atr' not in df.columns:
        df = calculate_atr(df, config['strategy']['atr_period'])
    
    # 2. Sinyaller (Mevcut strateji kurallarıyla)
    # SADECE FİLTRELİ SİNYALLERİ ALMAK mantıklı mı? 
    # EVET: Zaten elenen sinyalleri ML'e sokmanın anlamı yok, stratejinin "girmek istediği" ama "ML'in durduracağı" yerleri arıyoruz.
    df = SignalGenerator.generate_signals(df, config)
    
    # 3. Feature Engineering
    df = engineering_features(df)
    
    # 4. Sinyal Noktaları
    # entry_signal != 0 olan yerlerin integer pozisyonlarını (iloc) alıyoruz
    all_signal_ilocs = np.where(df['entry_signal'] != 0)[0]
    
    # Feature'lar NaN olmamalı (ilk 200 barı eliyoruz)
    ilocs = [i for i in all_signal_ilocs if i > 200]
    valid_indices = df.index[ilocs]
    
    if len(ilocs) == 0:
        print("Uyarı: Hiç sinyal yok veya yeterli veri yok.")
        return pd.DataFrame(), pd.Series()
    
    # 5. Label Üretimi
    h_bars = config['ml'].get('horizon_bars', 48)
    tp_r = config['ml'].get('tp_r_multiple', 1.0)
    
    y_series = generate_labels(df, ilocs, df['donchian_upper'], df['donchian_lower'], df['atr'], h_bars, tp_r)
    
    # 6. Feature Seçimi ve Filtreleme
    # Feature sütunlarını belirle
    feature_cols = [
        'adx', 'ema50_slope', 'volatility_20', 'chop_proxy', 'sign_flips_20',
        'atrp', 'donchian_width', 'trend_ema'
    ]
    # ROC sütunları
    feature_cols += [c for c in df.columns if 'roc_' in c]
    
    # Strength (Yöne göre seç)
    # Bunu dataset satırlarında dinamik seçmek lazım.
    # Yöntem: 'breakout_strength' diye tek bir kolon yapalım.
    # Long ise dist_upper, Short ise dist_lower.
    
    X_list = []
    y_list = []
    
    for i, idx in enumerate(valid_indices):
        iloc = ilocs[i]
        
        # Label yoksa atla (dataset sonu)
        if iloc not in y_series.index:
            continue
            
        label = y_series[iloc]
        
        row = df.iloc[iloc]
        signal = row['entry_signal']
        
        # Strength selection
        strength = row['dist_upper_atr'] if signal == 1 else row['dist_lower_atr']
        
        # Özellik vektörü
        features = row[feature_cols].to_dict()
        features['breakout_strength'] = strength
        features['direction'] = int(signal) # 1 or -1
        features['timestamp'] = idx # Satır indeksi (genelde datetime)
        
        X_list.append(features)
        y_list.append(label)
        
    X = pd.DataFrame(X_list)
    y = pd.Series(y_list)
    
    return X, y
