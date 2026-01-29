
import pandas as pd
import numpy as np

def calculate_donchian_channel(df, period=20):
    """
    Donchian Kanalı hesaplar.
    
    Kritik: Lookahead bias olmaması için hesaplanan değerler 1 bar öteye (shift) kaydırılır.
    Yani t anındaki Upper band, t-1 dahil son N barın en yükseğidir.
    t anındaki High fiyatı bu Upper bandı kırarsa sinyal oluşur.
    
    :param df: DataFrame (high, low kolonları gerekli)
    :param period: Kanal periyodu (N)
    :return: DataFrame (donchian_upper, donchian_lower, donchian_mid eklenmiş)
    """
    df = df.copy()
    
    # Mevcut bar DAHİL son N barın en yükseği/düşüğü
    # Ancak biz breakout kontrolü yaparken mevcut barın bu kanalı kırıp kırmadığına bakacağız.
    # Bu yüzden, karşılaştırma yapacağımız referans değer "önceki N barın" değeridir.
    
    # Adım 1: Rolling Max/Min hesapla (min_periods=period diyerek ilk N barda NaN olmasını sağla)
    # Bu hesaplama "current bar dahil" pencereye bakar.
    curr_max = df['high'].rolling(window=period, min_periods=period).max()
    curr_min = df['low'].rolling(window=period, min_periods=period).min()
    
    # Adım 2: SHIFT(1) uygula
    # Böylece t anındaki 'donchian_upper' değeri, [t-N, t-1] aralığındaki en yüksek değer olur.
    # t anındaki fiyat buna dahil EDİLMEZ.
    df['donchian_upper'] = curr_max.shift(1)
    df['donchian_lower'] = curr_min.shift(1)
    
    # Orta bant (opsiyonel analiz için)
    df['donchian_mid'] = (df['donchian_upper'] + df['donchian_lower']) / 2
    
    return df
