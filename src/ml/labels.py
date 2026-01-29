
import pandas as pd
import numpy as np

def generate_labels(df, entries_iloc, donchian_upper, donchian_lower, atr, h_bars=48, risk_reward_ratio=1.0):
    """
    ML için target etiketlerini (Y) üretir.
    
    Y=1 (Başarılı Trade):
        - Long için: Entry + 1R (Initial Stop Mesafesi) seviyesine, Stop'tan önce ulaşırsa.
        - Short için: Entry - 1R seviyesine, Stop'tan önce ulaşırsa.
        - Süre Sınırı: H bar (Horizon). H bar içinde ikisi de olmazsa Y=0.
    
    :param df: OHLCV DataFrame
    :param entries_iloc: Giriş sinyali olan barların iloc indexleri
    :param h_bars: Horizon (bar sayısı)
    :return: pd.Series (index=entries_iloc, value=0/1)
    """
    labels = {}
    
    # DataFrame Hızlı Erişim (Numpy Array)
    high_arr = df['high'].values
    low_arr = df['low'].values
    open_arr = df['open'].values
    close_arr = df['close'].values # Sinyal barı close'u
    
    # Giriş tipi (Long/Short) ve Stop mesafesi için indikatörler lazım
    # Bu metodun çağrıldığı yerde 'entry_signal', 'donchian_upper', 'atr' tam olmalı
    # entries_iloc, SİNYAL oluşan barlar. İşlem bir sonraki bar OPEN.
    
    signal_arr = df['entry_signal'].values
    atr_arr = atr.values
    
    # Config'den katsayı almalı aslında ama varsayılan 2.5 stop, 1R hedef
    # Not: Initial stop = k * ATR. Target = Initial Stop Dist.
    k_atr = 2.5 
    
    for i in entries_iloc:
        # Sinyal T anında. Giriş T+1 Open.
        entry_idx = i + 1
        if entry_idx >= len(df):
            continue
            
        entry_price = open_arr[entry_idx]
        signal = signal_arr[i]
        curr_atr = atr_arr[i]
        
        # Stop Mesafesi (R)
        r_dist = curr_atr * k_atr
        
        if signal == 1: # LONG
            tp_price = entry_price + (r_dist * risk_reward_ratio)
            sl_price = entry_price - r_dist
            
            # Gelecek H barı tara (Entry barı dahil)
            # Entry barında GAP olabilir mi? Evet.
            # Eğer entry barında open zaten SL altındaysa -> Anında Y=0
            if entry_price <= sl_price: 
                labels[i] = 0
                continue
                
            future_highs = high_arr[entry_idx : entry_idx + h_bars]
            future_lows = low_arr[entry_idx : entry_idx + h_bars]
            
            # İlk hangisi görüldü?
            # Basit vektörize olmayan döngü (H küçük olduğu için hızlı)
            result = 0
            for k in range(len(future_highs)):
                h = future_highs[k]
                l = future_lows[k]
                
                # Önce SL mi TP mi? Bar içi sıralamayı bilmediğimizden muhafazakar olalım:
                # Eğer aynı barda hem SL hem TP varsa -> SL oldu say.
                
                if l <= sl_price:
                    result = 0
                    break
                if h >= tp_price:
                    result = 1
                    break
            labels[i] = result

        elif signal == -1: # SHORT
            tp_price = entry_price - (r_dist * risk_reward_ratio)
            sl_price = entry_price + r_dist

            if entry_price >= sl_price:
                labels[i] = 0
                continue

            future_highs = high_arr[entry_idx : entry_idx + h_bars]
            future_lows = low_arr[entry_idx : entry_idx + h_bars]
            
            result = 0
            for k in range(len(future_lows)):
                h = future_highs[k]
                l = future_lows[k]
                
                if h >= sl_price:
                    result = 0
                    break
                if l <= tp_price:
                    result = 1
                    break
            labels[i] = result
            
    return pd.Series(labels)
