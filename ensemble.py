import os, joblib
import numpy as np

MODEL_DIR = "./model"

def ensemble_catboost_with_calibrators(prefix="A", n_folds=5):
    models, calibrators = [], []
    for fold in range(n_folds):
        m_path = os.path.join(MODEL_DIR, f"catboost_{prefix}_fold{fold}.pkl")
        c_path = os.path.join(MODEL_DIR, f"calibrator_{prefix}_fold{fold}.pkl")
        if os.path.exists(m_path) and os.path.exists(c_path):
            models.append(joblib.load(m_path))
            calibrators.append(joblib.load(c_path))
    print(f"{prefix}: loaded {len(models)} models and calibrators.")

    class CalibratedEnsemble:
        def __init__(self, models, calibrators):
            self.models = models
            self.calibrators = calibrators
        def predict_proba(self, X):
            preds = []
            for m, c in zip(self.models, self.calibrators):
                p = m.predict_proba(X)[:,1]
                p_cal = c.predict(p)
                preds.append(p_cal)
            mean_pred = np.mean(preds, axis=0)
            return np.vstack([1 - mean_pred, mean_pred]).T

    return CalibratedEnsemble(models, calibrators)

# 두 모델 앙상블 생성
ens_A = ensemble_catboost_with_calibrators("A")
ens_B = ensemble_catboost_with_calibrators("B")

# 제출용 이름으로 저장
joblib.dump(ens_A, os.path.join(MODEL_DIR, "lgbm_A.pkl"))
joblib.dump(ens_B, os.path.join(MODEL_DIR, "lgbm_B.pkl"))
print("✅ Saved lgbm_A.pkl, lgbm_B.pkl")