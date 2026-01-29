
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

# Opsiyonel: LightGBM
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except (ImportError, OSError):
    HAS_LGBM = False

class MLModel(BaseEstimator, ClassifierMixin):
    """
    Model Wrapper. LightGBM varsa kullanır, yoksa RandomForest.
    Class imbalance yönetimi içerir.
    """
    def __init__(self, model_type='lightgbm', **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = None
        
    def fit(self, X, y):
        # Class weight hesapla
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        if self.model_type == 'lightgbm' and HAS_LGBM:
            # LightGBM params
            params = self.kwargs.copy()
            if 'n_estimators' not in params: params['n_estimators'] = 100
            if 'learning_rate' not in params: params['learning_rate'] = 0.05
            if 'max_depth' not in params: params['max_depth'] = 5
            
            self.model = LGBMClassifier(
                scale_pos_weight=scale_pos_weight,
                verbosity=-1,
                **params
            )
        else:
            # Fallback: Random Forest
            if self.model_type == 'lightgbm':
                print("Uyarı: LightGBM bulunamadı, RandomForest kullanılıyor.")
            
            params = self.kwargs.copy()
            if 'n_estimators' not in params: params['n_estimators'] = 100
            if 'max_depth' not in params: params['max_depth'] = 7
            
            self.model = RandomForestClassifier(
                class_weight='balanced',
                n_jobs=-1,
                **params
            )
            
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
