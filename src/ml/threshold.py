
import numpy as np
import pandas as pd

class ThresholdOptimizer:
    """
    Validasyon setinde en iyi ML skor eşiğini seçer.
    """
    @staticmethod
    def optimize(y_true, y_probs, min_trades=5):
        """
        :param y_true: Gerçek etiketler (0/1)
        :param y_probs: Modelin 1 sınıfı için verdiği olasılıklar
        """
        best_t = 0.5
        best_score = -9999
        
        # 0.30'dan 0.80'e kadar tara
        thresholds = np.linspace(0.30, 0.80, 51)
        
        results = []
        
        for t in thresholds:
            # Filtrele: Olasılık > t olanları al
            # Sadece trade alınan durumları simüle ediyoruz
            
            # y_prob > t olan indeksler -> Trade Alındı
            mask = y_probs >= t
            
            selected_trades_count = np.sum(mask)
            if selected_trades_count < min_trades:
                score = -9999
            else:
                # Seçilenlerin içindeki Başarı Oranı (Precision)
                # Precision = TP / (TP + FP)
                # y_true[mask] = Seçilenlerin gerçek değerleri
                
                wins = np.sum(y_true[mask] == 1)
                losses = np.sum(y_true[mask] == 0)
                
                win_rate = wins / selected_trades_count
                
                # Hedef Fonksiyonu:
                # Sadece Win Rate yetmez, trade sayısı da önemli.
                # Skor = WinRate * log(TradeCount)
                # Veya daha basiti: WinRate, ama trade sayısı çok düşünce ceza.
                
                score = win_rate * 100
                # Trade sayısı cezası (çok az trade istemiyoruz)
                # Amaç PF artırmak. Win Rate artarsa PF artar.
                
            if score > best_score:
                best_score = score
                best_t = t
                
        return best_t, best_score
