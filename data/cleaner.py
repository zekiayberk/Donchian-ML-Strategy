
import pandas as pd
import numpy as np

class DataCleaner:
    """
    OHLCV verilerini temizler, eksik barları kontrol eder ve doğrular.
    """
    @staticmethod
    def clean_and_validate(df, timeframe_str):
        """
        Veri setini temizler ve eksik barları raporlar.
        
        :param df: OHLCV DataFrame (datetime kolonu olmalı)
        :param timeframe_str: '1h', '15m' vb.
        :return: Temizlenmiş DataFrame
        """
        df = df.copy()
        
        # Tarih kolonunu index yap (eğer değilse)
        if 'datetime' in df.columns:
            df.set_index('datetime', inplace=True)
            
        # Tekrar eden kayıtları temizle
        duplicate_count = df.index.duplicated().sum()
        if duplicate_count > 0:
            print(f"Uyarı: {duplicate_count} adet mükerrer zaman damgası siliniyor.")
            df = df[~df.index.duplicated(keep='first')]
            
        # Zaman sıralaması
        df.sort_index(inplace=True)
        
        # Eksik bar kontrolü
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=DataCleaner._convert_tf(timeframe_str))
        missing_count = len(full_idx) - len(df)
        
        if missing_count > 0:
            print(f"KRİTİK UYARI: {missing_count} adet eksik bar tespit edildi!")
            # Eksik barları doldurma politikası: Şimdilik sadece raporluyoruz. 
            # Forward fill tehlikeli olabilir, NaN bırakıp analizde handle etmek daha güvenli olabilir 
            # veya ffill yapıp not düşebiliriz.
            # Strateji "No Data Leakage" istediği için, eksik veri varsa o an trade yapmamak en doğrusu.
            # Ancak backtest motorunun sürekli zamana ihtiyacı varsa reindex gerekebilir.
            
            # Reindex ile eksik zamanları NaN olarak ekleyelim
            df = df.reindex(full_idx)
            # Eksik verileri doldurma? Opsiyonel: ffill
            # df.fillna(method='ffill', inplace=True) 
            print("Eksik barlar index'e eklendi (değerler NaN). Backtest motoru bunu handle etmeli.")
            
        # Veri tiplerini garantiye al
        cols = ['open', 'high', 'low', 'close', 'volume']
        for col in cols:
            df[col] = df[col].astype(float)
            
        return df

    @staticmethod
    def _convert_tf(tf_str):
        """
        Pandas frequency string dönüşümü.
        örn: 1h -> 1H, 15m -> 15T, 1d -> 1D
        """
        if tf_str.endswith('m'):
            return f"{tf_str[:-1]}T" # Dakika
        elif tf_str.endswith('h'):
            return f"{tf_str[:-1]}h" # Saat (updated for pandas future version)
        elif tf_str.endswith('d'):
            return f"{tf_str[:-1]}D" # Gün
        return tf_str.upper()

