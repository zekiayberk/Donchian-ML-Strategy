
import pandas as pd
import numpy as np
import yaml
import json
import pickle
import os
from .dataset import build_dataset
from .models import MLModel
from .calibrate import CalibratedModel
from .threshold import ThresholdOptimizer

class WFO_ML_Trainer:
    def __init__(self, config):
        self.config = config
        self.output_dir = "models"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def train(self, raw_df):
        print("ML Dataset oluşturuluyor...")
        X_full, y_full = build_dataset(raw_df, self.config)
        
        if X_full.empty:
            print("Hata: Yeterli veri yok.")
            return None
            
        # Timestamp kolonunu ayır
        timestamps = X_full['timestamp']
        X = X_full.drop(columns=['timestamp'])
        y = y_full
        
        # WFO Ayarları
        train_days = self.config['ml'].get('train_days', 180)
        valid_days = self.config['ml'].get('valid_days', 30)
        test_days = self.config['ml'].get('test_days', 30) # Her fold'un OOS süresi
        
        # Tarih aralıklarını belirle
        start_date = timestamps.min()
        end_date = timestamps.max()
        
        print(f"Veri Aralığı: {start_date} -> {end_date}")
        
        # Basit Walk-Forward: 
        # Fold 1: [Train] [Valid] [Test1]
        # Fold 2: [Train] [Valid] [Test2] (Test1 süresi kadar kaydırılmış)
        
        folds = []
        current_test_start = start_date + pd.Timedelta(days=train_days + valid_days)
        
        fold_idx = 1
        while current_test_start < end_date:
            test_end = current_test_start + pd.Timedelta(days=test_days)
            valid_start = current_test_start - pd.Timedelta(days=valid_days)
            train_start = valid_start - pd.Timedelta(days=train_days)
            
            # Maskeler
            train_mask = (timestamps >= train_start) & (timestamps < valid_start)
            valid_mask = (timestamps >= valid_start) & (timestamps < current_test_start)
            test_mask = (timestamps >= current_test_start) & (timestamps < test_end)
            
            if train_mask.sum() < 20 or valid_mask.sum() < 5:
                # Yetersiz veri varsa bu fold'u atla veya kaydır
                current_test_start += pd.Timedelta(days=test_days)
                continue

            X_train, y_train = X[train_mask], y[train_mask]
            X_valid, y_valid = X[valid_mask], y[valid_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            
            def get_dist(y_series):
                if y_series.empty: return "N/A"
                vc = y_series.value_counts()
                return f"Total: {len(y_series)} (0: {vc.get(0, 0)}, 1: {vc.get(1, 0)})"

            print(f"\n--- Fold {fold_idx:02d} ---")
            print(f"Train: {train_start.strftime('%Y-%m-%d')} -> {valid_start.strftime('%Y-%m-%d')} | {get_dist(y_train)}")
            print(f"Valid: {valid_start.strftime('%Y-%m-%d')} -> {current_test_start.strftime('%Y-%m-%d')} | {get_dist(y_valid)}")
            print(f"Test:  {current_test_start.strftime('%Y-%m-%d')} -> {min(test_end, end_date).strftime('%Y-%m-%d')} | {get_dist(y_test)}")
            
            # Model Eğitim
            base_model = MLModel(model_type=self.config['ml']['model'])
            clf = CalibratedModel(base_model)
            clf.fit(X_train, y_train, X_valid, y_valid)
            
            # Threshold Optimizasyonu
            valid_probs = clf.predict_proba(X_valid)[:, 1]
            best_t, v_score = ThresholdOptimizer.optimize(y_valid.values, valid_probs)
            
            # Test Performansı
            test_metrics = {}
            if not X_test.empty:
                test_probs = clf.predict_proba(X_test)[:, 1]
                test_preds = (test_probs >= best_t).astype(int)
                acc = np.mean(test_preds == y_test.values)
                test_metrics = {"accuracy": float(acc), "samples": int(len(y_test))}
                print(f"Fold {fold_idx} Test Acc: {acc:.2f}")

            # Kaydet
            model_file = f"fold_{fold_idx:02d}.pkl"
            meta_file = f"fold_{fold_idx:02d}_meta.json"
            
            # Pickle Model
            with open(os.path.join(self.output_dir, model_file), 'wb') as f:
                pickle.dump(clf, f)
                
            # JSON Meta
            meta = {
                "fold": fold_idx,
                "train_range": [str(train_start), str(valid_start)],
                "valid_range": [str(valid_start), str(current_test_start)],
                "test_range": [str(current_test_start), str(test_end)],
                "threshold": float(best_t),
                "feature_cols": X.columns.tolist(),
                "test_metrics": test_metrics
            }
            with open(os.path.join(self.output_dir, meta_file), 'w') as f:
                json.dump(meta, f, indent=4)
                
            folds.append(meta)
            fold_idx += 1
            current_test_start += pd.Timedelta(days=test_days)
            
        print(f"\nTraining tamamlandı. {len(folds)} fold kaydedildi.")
        return folds

if __name__ == "__main__":
    with open('config.yaml') as f:
        conf = yaml.safe_load(f)
        
    # Veriyi yükle
    data_path = f"data/storage/{conf['data']['symbol'].replace('/','')}_{conf['data']['timeframe']}_{conf['data']['start_date']}_{conf['data']['end_date']}.csv"
    if not os.path.exists(data_path):
        print(f"Veri bulunamadı: {data_path}")
    else:
        df = pd.read_csv(data_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        trainer = WFO_ML_Trainer(conf)
        trainer.train(df)
