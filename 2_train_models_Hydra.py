# ============================================================
# 2_Train_Models(Hydra).py
# 목적: 전처리된 Feather 불러와 K-Fold + CatBoost + Isotonic 보정
# Hydra + Optuna로 자동 하이퍼파라미터 탐색 및 결과 로깅
# ============================================================

import os
import numpy as np
import pandas as pd
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
import joblib
import warnings
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from datetime import datetime

warnings.filterwarnings("ignore")


# ============================================================
# Metric Functions
# ============================================================

def expected_calibration_error(y_true, y_prob, n_bins=10):
    if len(y_true) == 0 or len(y_prob) == 0:
        return 0.0
    y_prob = np.nan_to_num(y_prob, nan=0.0)

    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_edges[0], bin_edges[-1] = -0.001, 1.001

    df["y_prob"] = np.clip(df["y_prob"], 0, 1)
    df["bin"] = pd.cut(df["y_prob"], bins=bin_edges, right=True)

    bin_stats = df.groupby("bin", observed=True).agg(
        bin_total=("y_prob", "count"),
        prob_true=("y_true", "mean"),
        prob_pred=("y_prob", "mean"),
    )

    non_empty = bin_stats[bin_stats["bin_total"] > 0]
    if len(non_empty) == 0:
        return 0.0

    weights = non_empty["bin_total"] / len(y_prob)
    ece = np.sum(weights * np.abs(non_empty["prob_true"] - non_empty["prob_pred"]))
    return ece


def combined_score(y_true, y_prob, n_bins_ece=10):
    if (
        len(y_true) == 0
        or len(y_prob) == 0
        or np.sum(y_true) == 0
        or np.sum(y_true) == len(y_true)
    ):
        print("  [주의] 단일 클래스 데이터 → Combined Score = 1.0")
        return 1.0

    y_prob = np.nan_to_num(y_prob, nan=0.0)
    auc = roc_auc_score(y_true, y_prob)
    brier = mean_squared_error(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins_ece)
    score = 0.5 * (1 - auc) + 0.25 * brier + 0.25 * ece

    print(f"  AUC={auc:.4f}, Brier={brier:.4f}, ECE={ece:.4f}, Combined={score:.5f}")
    return score


# ============================================================
# Model Training
# ============================================================

def train_model(cfg, X_train, y_train, X_val, y_val, pk_stats_fold=None, group_label="A"):
    cat_features = list(cfg.cat_features)
    drop_cols = set(cfg.drop_cols_train)

    # leakage-free pk_stats_fold (B모델에만 사용)
    if pk_stats_fold is not None:
        X_train = X_train.merge(pk_stats_fold, on="PrimaryKey", how="left")
        X_val = X_val.merge(pk_stats_fold, on="PrimaryKey", how="left")

    # numeric feature 선택
    numeric_cols = [
        c for c in X_train.columns
        if c not in cat_features and c not in drop_cols
    ]
    numeric_cols = list(set(numeric_cols) & set(X_val.columns))

    cb_X_train = X_train[numeric_cols + cat_features].copy()
    cb_X_val = X_val[numeric_cols + cat_features].copy()

    for col in cat_features:
        cb_X_train[col] = cb_X_train[col].fillna("nan").astype(str)
        cb_X_val[col] = cb_X_val[col].fillna("nan").astype(str)

    cat_idx = [
        cb_X_train.columns.get_loc(c)
        for c in cat_features
        if c in cb_X_train.columns
    ]

    # 어떤 설정 쓸지 결정 (A or B)
    mcfg = cfg.modelA if group_label.startswith("A") else cfg.modelB

    print(f"\n[{group_label}] CatBoost 학습 시작 ({len(cb_X_train.columns)} features)")
    model = cb.CatBoostClassifier(
        iterations=int(mcfg.iterations),
        learning_rate=float(mcfg.learning_rate),
        depth=int(mcfg.depth),
        l2_leaf_reg=float(mcfg.l2_leaf_reg),
        loss_function=str(mcfg.loss_function),
        eval_metric=str(mcfg.eval_metric),
        random_seed=42,
        thread_count=int(mcfg.thread_count),
        early_stopping_rounds=int(mcfg.early_stopping_rounds),
        verbose=1000,
        task_type=str(mcfg.task_type),
    )

    model.fit(
        cb_X_train,
        y_train,
        eval_set=[(cb_X_val, y_val)],
        cat_features=cat_idx,
    )

    # Calibration with Isotonic Regression
    pred_uncal = model.predict_proba(cb_X_val)[:, 1]
    print(f"[{group_label}] 비보정 점수:")
    _ = combined_score(y_val, pred_uncal, cfg.metric.n_bins_ece)

    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    calibrator.fit(pred_uncal, y_val)

    pred_cal = calibrator.predict(pred_uncal)
    print(f"[{group_label}] ★보정 후 점수★")
    score = combined_score(y_val, pred_cal, cfg.metric.n_bins_ece)

    return model, calibrator, score


# ============================================================
# Main
# ============================================================

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 🔑 Optuna가 modelA.*, modelB.*를 override 할 수 있도록 struct 잠금 해제
    OmegaConf.set_struct(cfg, False)
    if "modelA" in cfg:
        OmegaConf.set_struct(cfg.modelA, False)
    if "modelB" in cfg:
        OmegaConf.set_struct(cfg.modelB, False)

    print("Big-Tech ML Engineer: Hydra 기반 K-Fold 학습/보정 + 결과 로깅 시작")

    base_dir = to_absolute_path(cfg.general.base_dir)
    model_dir = to_absolute_path(cfg.general.model_save_dir)
    os.makedirs(model_dir, exist_ok=True)

    # 로그 디렉토리
    log_dir = os.path.join(model_dir, "hydra_logs")
    os.makedirs(log_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(log_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    # 데이터 로드
    feature_path = os.path.join(base_dir, cfg.general.feature_file)
    all_train_df = pd.read_feather(feature_path)
    all_train_df["Label"] = all_train_df["Label"].fillna(0)
    print("데이터 로드 완료:", all_train_df.shape)

    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=cfg.cv.n_splits,
        shuffle=cfg.cv.shuffle,
        random_state=cfg.cv.random_state,
    )

    fold_results = []
    all_pk_stats_folds = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(all_train_df, all_train_df["Label"])
    ):
        print(f"\n========== Fold {fold + 1}/{cfg.cv.n_splits} ==========")

        train_df = all_train_df.iloc[train_idx].copy()
        val_df = all_train_df.iloc[val_idx].copy()

        # ----- PK Stats (B모델 용) -----
        agg_funcs = {
            "Age_num": ["mean", "min", "max"],
            "RiskScore": ["mean", "std", "max"],
            "Test_id": ["count"],
        }
        valid_agg = {c: fs for c, fs in agg_funcs.items() if c in train_df.columns}

        pk_stats = train_df.groupby("PrimaryKey").agg(valid_agg)
        pk_stats.columns = ["_".join(col).strip() for col in pk_stats.columns.values]
        if "Test_id_count" in pk_stats.columns:
            pk_stats.rename(
                columns={"Test_id_count": "pk_test_total_count"}, inplace=True
            )
        pk_stats.reset_index(inplace=True)
        all_pk_stats_folds.append(pk_stats)

        # ----- A/B 분리 -----
        X_train_A = train_df[train_df["Test_x"] == "A"]
        y_train_A = X_train_A["Label"].values
        X_val_A = val_df[val_df["Test_x"] == "A"]
        y_val_A = X_val_A["Label"].values

        X_train_B = train_df[train_df["Test_x"] == "B"]
        y_train_B = X_train_B["Label"].values
        X_val_B = val_df[val_df["Test_x"] == "B"]
        y_val_B = X_val_B["Label"].values

        # ----- Model A -----
        model_A, calib_A, score_A = train_model(
            cfg,
            X_train_A,
            y_train_A,
            X_val_A,
            y_val_A,
            pk_stats_fold=None,
            group_label=f"A_fold{fold+1}",
        )
        joblib.dump(model_A, os.path.join(model_dir, f"catboost_A_fold{fold}.pkl"))
        joblib.dump(calib_A, os.path.join(model_dir, f"calibrator_A_fold{fold}.pkl"))

        # ----- Model B -----
        if (
            len(X_train_B) > 0
            and len(X_val_B) > 0
            and len(np.unique(y_train_B)) > 1
        ):
            model_B, calib_B, score_B = train_model(
                cfg,
                X_train_B,
                y_train_B,
                X_val_B,
                y_val_B,
                pk_stats_fold=pk_stats,
                group_label=f"B_fold{fold+1}",
            )
            joblib.dump(model_B, os.path.join(model_dir, f"catboost_B_fold{fold}.pkl"))
            joblib.dump(
                calib_B, os.path.join(model_dir, f"calibrator_B_fold{fold}.pkl")
            )
        else:
            score_B = np.nan
            print(f"[Fold {fold+1}] B 모델 스킵 (데이터 부족 or 단일 클래스)")

        fold_mean = np.nanmean([score_A, score_B])
        fold_results.append(
            {
                "fold": fold + 1,
                "score_A": score_A,
                "score_B": score_B,
                "combined_mean": fold_mean,
            }
        )

    # ----- Fold 결과 정리 -----
    results_df = pd.DataFrame(fold_results)
    mean_score = np.nanmean(results_df["combined_mean"])
    results_df["overall_mean"] = mean_score

    # 설정 + 결과 요약
    param_summary = {
        "run_id": run_id,
        "mean_score": mean_score,
    }
    if "modelA" in cfg:
        for k, v in cfg.modelA.items():
            param_summary[f"modelA.{k}"] = v
    if "modelB" in cfg:
        for k, v in cfg.modelB.items():
            param_summary[f"modelB.{k}"] = v

    # CSV 저장
    results_df.to_csv(
        os.path.join(run_dir, "training_results.csv"), index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([param_summary]).to_csv(
        os.path.join(run_dir, "config_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print(f"\n[INFO] Fold별 점수 및 설정 저장 완료 → {run_dir}")

    # best_result.csv 갱신
    best_path = os.path.join(log_dir, "best_result.csv")
    if os.path.exists(best_path):
        prev = pd.read_csv(best_path)
        best_prev = float(prev.iloc[0]["mean_score"])
    else:
        best_prev = np.inf

    if mean_score < best_prev:
        pd.DataFrame([param_summary]).to_csv(
            best_path, index=False, encoding="utf-8-sig"
        )
        print(f"🏆 새 최고 점수 갱신! ({mean_score:.5f}) → best_result.csv 업데이트")
    else:
        print(f"현재 점수 {mean_score:.5f} (기존 최고 {best_prev:.5f})")

    print("\nBig-Tech ML Engineer: Hydra run 종료")
    return mean_score


if __name__ == "__main__":
    main()
