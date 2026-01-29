
import pandas as pd
import numpy as np
from itertools import product
from backtest.engine import BacktestEngine
from backtest.metrics import PerformanceMetrics
import copy

class WalkForwardOptimizer:
    def __init__(self, df, config, param_grid, train_period_days=180, test_period_days=30):
        """
        :param df: Tüm veri seti
        :param config: Baz konfigürasyon
        :param param_grid: Test edilecek parametre sözlüğü örn: {'donchian_period': [20, 55], 'stop_loss_atr_multiplier': [2.0, 3.0]}
        :param train_period_days: Eğitim penceresi (gün)
        :param test_period_days: Test penceresi (gün)
        """
        self.df = df
        self.config = config
        self.param_grid = param_grid
        self.train_win = pd.Timedelta(days=train_period_days)
        self.test_win = pd.Timedelta(days=test_period_days)
        
    def generate_folds(self):
        """Data üzerinde kaydırmalı pencereler oluşturur."""
        start_time = self.df.index.min()
        end_time = self.df.index.max()
        
        current_train_start = start_time
        folds = []
        
        while True:
            train_end = current_train_start + self.train_win
            test_end = train_end + self.test_win
            
            if test_end > end_time:
                break
                
            folds.append({
                'train_start': current_train_start,
                'train_end': train_end,
                'test_start': train_end, # Test train'in bittiği yerde başlar
                'test_end': test_end
            })
            
            # Bir sonraki fold için kaydır (Test periyodu kadar kaydırıyoruz - anchored walk forward değil, rolling)
            current_train_start += self.test_win
            
        return folds

    def optimize_fold(self, train_df, param_combinations):
        best_metric = -float('inf')
        best_params = None
        
        for params in param_combinations:
            # Config kopyala ve güncelle
            curr_config = copy.deepcopy(self.config)
            curr_config['strategy'].update(params)
            
            engine = BacktestEngine(train_df, curr_config)
            engine.run()
            trades, equity = engine.get_results()
            metrics = PerformanceMetrics.calculate(trades, equity, curr_config['backtest']['initial_capital'])
            
            # Hedef fonksiyon: Sharpe Ratio veya Net Profit
            # Basit olması için Net Profit * Profit Factor diyelim veya direkt Sharpe benzeri bir skor.
            # Kullanıcı "Sadece net profit değil; Sharpe + MaxDD + trade sayısı gibi çoklu kriter" istedi.
            
            score = 0
            if metrics['Total Trades'] > 5: # En az 5 trade şartı
                # Basit skor: Return / MaxDD (Calmar benzeri)
                ret = metrics['Total Return (%)']
                mdd = metrics['Max Drawdown (%)']
                if mdd == 0: mdd = 0.1
                score = ret / mdd
            else:
                score = -999
                
            if score > best_metric:
                best_metric = score
                best_params = params
                
        return best_params, best_metric

    def run(self):
        folds = self.generate_folds()
        print(f"Toplam {len(folds)} fold üzerinde WFO başlatılıyor...")
        
        # Grid oluştur
        keys, values = zip(*self.param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        oos_trades = []
        oos_equity = []
        
        optimization_log = []
        
        # Başlangıç sermayesi (WFO boyunca kümülatif gidebilir veya resetlenebilir)
        # Burada her test periyodu için sanal birleştirmeyi yapacağız.
        
        for i, fold in enumerate(folds):
            print(f"Fold {i+1}/{len(folds)}: Train {fold['train_start'].date()} Test {fold['test_start'].date()}")
            
            train_mask = (self.df.index >= fold['train_start']) & (self.df.index < fold['train_end'])
            train_df = self.df.loc[train_mask]
            
            # 1. Optimize
            best_params, score = self.optimize_fold(train_df, param_combinations)
            
            if best_params is None:
                # Fallback to default
                best_params = {k: v[0] for k, v in self.param_grid.items()}
                print("  Uygun parametre bulunamadı, varsayılan kullanılıyor.")
            else:
                print(f"  En iyi params: {best_params} (Skor: {score:.2f})")
            
            optimization_log.append({
                'fold': i,
                'test_date': fold['test_start'],
                'best_params': best_params,
                'train_score': score
            })
            
            # 2. Test (Out of Sample)
            test_mask = (self.df.index >= fold['test_start']) & (self.df.index < fold['test_end'])
            test_df = self.df.loc[test_mask]
            
            if not test_df.empty:
                # OOS testi çalıştır
                test_config = copy.deepcopy(self.config)
                test_config['strategy'].update(best_params)
                
                # Test motoru her fold'da "sıfırdan" başlıyor gibi çalışırsa equity curve birleşimi zor olur.
                # Ancak engine'e initial capital verebiliriz.
                # Basit yaklaşım: Sadece trade'leri topla.
                
                engine = BacktestEngine(test_df, test_config)
                engine.run()
                t_df, e_df = engine.get_results()
                
                if not t_df.empty:
                    oos_trades.append(t_df)

        # Sonuçları Birleştir
        if oos_trades:
            all_trades = pd.concat(oos_trades)
            all_trades.sort_values('entry_time', inplace=True)
            
            # Equity curve'ü trade'lerden yeniden oluşturmak en sağlıklısı
            # (Çünkü engine içindeki time-based equity curve parçalı)
            # Basitçe trade sonuçlarını kümülatif topla
            all_trades['cumulative_pnl'] = all_trades['pnl'].cumsum()
            all_trades['equity'] = self.config['backtest']['initial_capital'] + all_trades['cumulative_pnl']
            
            return all_trades, optimization_log
        else:
            return pd.DataFrame(), optimization_log

