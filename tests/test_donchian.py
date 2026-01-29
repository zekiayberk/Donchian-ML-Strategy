
import pytest
import pandas as pd
import numpy as np
from indicators.donchian import calculate_donchian_channel

def test_donchian_calculation():
    # Basit veri: 10, 20, 30, 40, 50
    data = {
        'high': [10, 20, 30, 40, 50, 60],
        'low':  [ 5, 15, 25, 35, 45, 55],
        'close':[ 8, 18, 28, 38, 48, 58]
    }
    df = pd.DataFrame(data)
    
    # Period = 3
    # Beklenen:
    # Bar 0: NaN
    # Bar 1: NaN (min_periods=3)
    # Bar 2: NaN (Highs: 10, 20, 30 -> Max 30 Backtest N=3 -> Shift 1)
    # ...
    
    # Donchian Upper Mantığı: Shift(1) uygulandığı için
    # T anındaki Upper = Max(High[T-N : T-1])
    
    # Period 2 diyelim kolay olsun
    df = calculate_donchian_channel(df, period=2)
    
    # Index 2 (3. bar, değer 30):
    # Önceki 2 bar: Index 0 (10), Index 1 (20). Max = 20.
    # df.iloc[2]['donchian_upper'] == 20 olmalı.
    
    # Kontrol edelim:
    # Rolling(2).max() -> [NaN, 20, 30, 40, 50, 60] (Kendisi dahil)
    # Shift(1) -> [NaN, NaN, 20, 30, 40, 50]
    
    expected_val_at_index_2 = 20.0
    assert df.iloc[2]['donchian_upper'] == expected_val_at_index_2, f"Beklenen {expected_val_at_index_2}, Alınan {df.iloc[2]['donchian_upper']}"
    
    # Lookahead check:
    # Index 2'deki 'Upper' değeri hesaplanırken Index 2'nin High'ı (30) kullanılmamalı.
    # Kullanılanlar Index 0 ve 1.
    # Eğer Index 2 high 30 ise ve Upper 20 ise -> Kırılım olur (30 > 20).
    # Bu doğrudur.
