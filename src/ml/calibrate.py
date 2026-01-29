
from sklearn.isotonic import IsotonicRegression
import numpy as np

class CalibratedModel:
    """
    Model olasılıklarını kalibre eder (Manuel Isotonic Regression).
    Scikit-learn sürüm uyumsuzluklarını önlemek için CalibratedClassifierCV yerine doğrudan implementasyon.
    """
    def __init__(self, base_model, method='isotonic'):
        self.base_model = base_model
        self.method = method
        self.calibrator = None
        
    def fit(self, X_train, y_train, X_valid, y_valid):
        """
        Train setinde modeli eğitir, Valid setinde kalibrasyon yapar.
        """
        # 1. Base model eğit
        print("Base Model eğitiliyor...")
        self.base_model.fit(X_train, y_train)
        
        # 2. Valid olasılıklarını al
        valid_probs = self.base_model.predict_proba(X_valid)[:, 1]
        
        # 3. Kalibratörü eğit
        print("Isotonic Calibration uygulanıyor...")
        if self.method == 'isotonic':
            # out_of_bounds='clip': 0-1 aralığında tut
            self.calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            self.calibrator.fit(valid_probs, y_valid)
            
        return self

    def predict_proba(self, X):
        # Base model tahminleri
        probs = self.base_model.predict_proba(X)[:, 1]
        
        if self.calibrator:
            # Kalibre edilmiş olasılıklar (Tek boyutlu array döner)
            calibrated_probs = self.calibrator.predict(probs)
            # Sklearn formatına uydurmak için [n_samples, 2] matrisi (Class 0, Class 1)
            return np.vstack([1 - calibrated_probs, calibrated_probs]).T
        
        return np.vstack([1 - probs, probs]).T
