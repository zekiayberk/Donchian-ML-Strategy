
import pickle
import pandas as pd
import numpy as np
import json
import traceback
from collections import defaultdict, deque
os = __import__('os') # Dosya işlemleri için

class MLEngine:
    def __init__(self, models_dir="models", threshold_offset=0.0, config=None):
        self.folds = [] # Meta verileri tutar
        self.models = {} # Model objelerini tutar (lazy load veya pre-load)
        self.models_dir = models_dir
        self.threshold_offset = threshold_offset
        self.load_all_folds()
        
        # Init Guard (Config varsa)
        if config:
            self._init_guard_state(config)
        else:
            # Dummy init (default True/False?)
            # Config yoksa guard kapalı varsayalım veya default değerlerle açalım
            # Şimdilik minimal default
            self.guard_enabled = False
        
    def load_all_folds(self):
        if not os.path.exists(self.models_dir):
            print(f"Modeller dizini bulunamadı: {self.models_dir}")
            return
            
        files = os.listdir(self.models_dir)
        meta_files = sorted([f for f in files if f.endswith("_meta.json")])
        
        for mf in meta_files:
            with open(os.path.join(self.models_dir, mf), 'r') as f:
                meta = json.load(f)
                # Tarihleri pd.Timestamp'a çevir
                meta['test_start'] = pd.Timestamp(meta['test_range'][0])
                meta['test_end'] = pd.Timestamp(meta['test_range'][1])
                self.folds.append(meta)
                print(f"  > Fold {meta['fold']}: {meta['test_range'][0]} -> {meta['test_range'][1]}")
                
        print(f"{len(self.folds)} ML Fold yüklendi. (Offset: {self.threshold_offset:+})")

    def _get_model_for_time(self, timestamp):
        """Timestamp'a uygun fold'u bulur"""
        # 1. Normal Aralık Kontrolü
        for meta in self.folds:
            if meta['test_start'] <= timestamp < meta['test_end']:
                fold_name = f"fold_{meta['fold']:02d}"
                if fold_name not in self.models:
                    # Lazy load model
                    model_path = os.path.join(self.models_dir, f"{fold_name}.pkl")
                    with open(model_path, 'rb') as f:
                        self.models[fold_name] = pickle.load(f)
                
                # Global offset uygula
                adjusted_threshold = meta['threshold'] + self.threshold_offset
                source = f"fold_{meta['fold']:02d}_meta"
                return self.models[fold_name], adjusted_threshold, meta['feature_cols'], source
        
        # 2. Forward Fallback (Eğer tarih son fold'dan sonraysa)
        # En son fold'u kullanarak tahmin yap (Forward Test)
        if self.folds and timestamp >= self.folds[-1]['test_end']:
            meta = self.folds[-1]
            fold_name = f"fold_{meta['fold']:02d}"
            if fold_name not in self.models:
                model_path = os.path.join(self.models_dir, f"{fold_name}.pkl")
                with open(model_path, 'rb') as f:
                    self.models[fold_name] = pickle.load(f)
            
            adjusted_threshold = meta['threshold'] + self.threshold_offset
            source = f"fold_{meta['fold']:02d}_meta (forward_fallback)"
            return self.models[fold_name], adjusted_threshold, meta['feature_cols'], source

        return None, 0.5 + self.threshold_offset, [], "default_config"

    def predict(self, row_df):
        """
        Tek bir satır (T anı) için tahmin üretir.
        :param row_df: DataFrame (tek satırlık, tüm featurelar hesaplanmış olmalı)
        :returns: dict { 'is_allowed': bool, 'prob': float, 'fold': int, 'status': str }
        """
        res = {'is_allowed': True, 'prob': 1.0, 'fold': -1, 'status': 'no_models'}

        if not self.folds:
            return res
            
        # Timestamp al
        # print(f"DEBUG: PREDICT INCOMING COLS: {list(row_df.columns)}")
        if 'timestamp' in row_df.columns:
            ts = row_df['timestamp'].iloc[0]
        else:
            ts = row_df.index[0]
            
        if not isinstance(ts, pd.Timestamp):
            ts = pd.to_datetime(ts)
            
        model, threshold, feature_cols, source = self._get_model_for_time(ts)
        
        if model is None:
            # Bu tarih için eğitilmiş bir fold TEST aralığı yok
            res['status'] = 'outside_test_range'
            res['used_threshold'] = threshold
            res['source'] = source
            res['feature_cols'] = feature_cols
            return res
            
        try:
            # DEBUG: Input vs Target Columns
            # print(f"DEBUG INFER: DF Cols={list(row_df.columns)}")
            # print(f"DEBUG INFER: Feat Cols={feature_cols}")
            
            # Check overlap
            # missing = [c for c in feature_cols if c not in row_df.columns]
            # if missing:
                # print(f"DEBUG INFER: MISSING COLS -> {missing}")

            # UYARI FİLTRESİ
            # LGBMClassifier ve Sklearn validasyonları arasındaki feature name uyuşmazlığını sustur
            # Biz feature_cols ile sırayı garanti ediyoruz, bu yüzden uyarı irrelevant.
            import warnings
            warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

            # --- EKSİK/FAZLA KOLON KONTROLÜ (User Request) ---
            missing = [c for c in feature_cols if c not in row_df.columns]
            extra = [c for c in row_df.columns if c not in feature_cols and c != 'timestamp'] # timestamp hariç
            
            if missing:
                print(f"[ML WARNING] Missing features (filled with 0): {missing}")
            # if extra:
            #     print(f"[ML DEBUG] Extra features ignored: {extra[:5]} ...")

            # Modelin beklediği kolon sırasını (metadata'dan) garanti et
            # reindex, eğer kolon eksikse NaN atar (fillna ile 0 yaparız)
            # Bu işlem hem feature name uyarısını çözer hem de sıralamayı sabitler
            X = row_df.reindex(columns=feature_cols).fillna(0)
            
            # Index reset (tek satır problemi için)
            X = X.reset_index(drop=True)
            
            # Predict (CalibratedModel predict_proba döner)
            prob = model.predict_proba(X)[0, 1]
            
            res['prob'] = float(prob)
            
            # Status ve Fold Belirle
            # Timestamp hangi fold aralığında?
            matching = [f for f in self.folds if f['test_start'] <= ts < f['test_end']]
            
            if matching:
                res['fold'] = int(matching[0]['fold'])
                res['status'] = 'oos_test'
            else:
                # Fallback durumu (Forward Test)
                if self.folds:
                    res['fold'] = int(self.folds[-1]['fold'])
                res['status'] = 'outside_test_range'
            
            res['used_threshold'] = threshold
            res['source'] = source
            res['feature_cols'] = feature_cols
            
            if prob >= threshold:
                res['is_allowed'] = True
            else:
                res['is_allowed'] = False
                
            return res
        except KeyError as e:
             # Spesifik olarak eksik kolon hatasını yakala
            print(f"ML Missing Columns Error ({ts}): {e}")
            if 'row_df' in locals():
                print(f"  Input DF Cols: {list(row_df.columns)}")
                print(f"  Target Cols: {feature_cols}")
            res['status'] = f'key_error: {str(e)}'
            return res
        except Exception as e:
            print(f"ML Prediction Hatası ({ts}) TYPE={type(e)}: {e}")
            traceback.print_exc()
            res['status'] = f'error: {str(e)}'
            return res

    # --- GUARD LOGIC ---
    def _init_guard_state(self, config):
        ml_cfg = config.get("ml", {})

        self.guard_enabled = bool(ml_cfg.get("guard_enabled", True))
        self.guard_min_signals_before_eval = int(ml_cfg.get("guard_min_signals_before_eval", 5))
        self.guard_min_take_rate = float(ml_cfg.get("guard_min_take_rate", 0.10))
        self.guard_top_k = int(ml_cfg.get("guard_top_k", 3))

        self.guard_quantile = float(ml_cfg.get("guard_quantile", 0.70))
        self.guard_floor = float(ml_cfg.get("guard_floor", 0.30))
        self.guard_hist_maxlen = int(ml_cfg.get("guard_hist_maxlen", 500))

        self.guard_debug = bool(ml_cfg.get("guard_debug", False))

        # Fold state
        self.fold_total_signals = defaultdict(int)
        self.fold_taken_base = defaultdict(int)
        self.fold_taken_guard = defaultdict(int)

        self.fold_prob_hist = defaultdict(lambda: deque(maxlen=self.guard_hist_maxlen))

    def _fold_take_rate_base(self, fold: str) -> float:
        total = self.fold_total_signals[fold]
        if total <= 0:
            return 0.0
        return self.fold_taken_base[fold] / total

    def _guard_threshold_for_fold(self, fold: str) -> float:
        hist = list(self.fold_prob_hist[fold])
        # az veri varsa floor'a dön
        if len(hist) < 10:
            return self.guard_floor
        q = float(np.quantile(hist, self.guard_quantile))
        return max(q, self.guard_floor)

    def _should_activate_guard(self, fold: str):
        """
        Returns (activate: bool, reason: str)
        """
        if not self.guard_enabled:
            return False, "guard_disabled"

        total = self.fold_total_signals[fold]
        if total < self.guard_min_signals_before_eval:
            return False, f"not_enough_signals(total={total})"

        base_rate = self._fold_take_rate_base(fold)
        if base_rate >= self.guard_min_take_rate:
            return False, f"base_rate_ok(rate={base_rate:.2f})"

        if self.fold_taken_guard[fold] >= self.guard_top_k:
            return False, f"topk_full(guard_taken={self.fold_taken_guard[fold]})"

        return True, f"low_base_rate(rate={base_rate:.2f})"

    def decide_with_guard(self, prob: float, fold: str, base_threshold: float, offset: float):
        """
        Returns:
            allowed: bool
            tag: "BASE" | "GUARD" | "SKIP"
            used_threshold: float
            meta: dict (debug)
        """
        if not self.guard_enabled:
             # Fallback to simple logic if guard not initialized
             target_t = float(base_threshold + offset)
             if prob >= target_t:
                 return True, "BASE", target_t, {}
             else:
                 return False, "SKIP", target_t, {}

        # update fold stats
        self.fold_total_signals[fold] += 1
        self.fold_prob_hist[fold].append(prob)

        target_t = float(base_threshold + offset)

        # BASE Check
        if prob >= target_t:
            self.fold_taken_base[fold] += 1
            return True, "BASE", target_t, {"reason": "prob>=base", "base_t": target_t}

        # GUARD Check (conditional)
        activate, reason = self._should_activate_guard(fold)
        if activate:
            guard_t = self._guard_threshold_for_fold(fold)
            if prob >= guard_t:
                self.fold_taken_guard[fold] += 1
                return True, "GUARD", guard_t, {
                    "reason": reason,
                    "base_t": target_t,
                    "guard_t": guard_t,
                    "guard_taken": self.fold_taken_guard[fold],
                    "base_rate": self._fold_take_rate_base(fold),
                }
            else:
                return False, "SKIP", target_t, {
                    "reason": f"{reason}|prob<guard_t",
                    "base_t": target_t,
                    "guard_t": guard_t,
                    "prob": prob
                }

        # SKIP
        return False, "SKIP", target_t, {"reason": reason, "base_t": target_t}
