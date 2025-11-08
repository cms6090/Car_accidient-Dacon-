# script.py (v13: GroupKFold CatBoost + Global Isotonic + Safe Submission)

import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import catboost as cb  # for unpickling
from sklearn.isotonic import IsotonicRegression  # for unpickling
import warnings

warnings.filterwarnings('ignore')
tqdm.pandas()


# =======================
# 1. 전처리 유틸
# =======================

def convert_age(val):
    if pd.isna(val):
        return np.nan
    try:
        s = str(val)
        base = int(s[:-1])
        return base if s[-1] == "a" else base + 5
    except Exception:
        return np.nan


def split_testdate(val):
    try:
        v = int(val)
        return v // 100, v % 100
    except Exception:
        return np.nan, np.nan


def seq_mean(series: pd.Series) -> pd.Series:
    return series.fillna("").progress_apply(
        lambda x: np.fromstring(x, sep=",").mean() if x else np.nan
    )


def seq_std(series: pd.Series) -> pd.Series:
    return series.fillna("").progress_apply(
        lambda x: np.fromstring(x, sep=",").std() if x else np.nan
    )


def masked_operation(cond_series, val_series, target_conds, operation='mean'):
    """
    cond_series: '1,2,1,...' 조건 시퀀스
    val_series : 같은 길이 값 시퀀스 (RT 또는 정답 코드)
    target_conds: 선택 조건 값(들)
    operation: 'mean' / 'std' / 'rate' / 'rate_yn'
    """
    cond_df = cond_series.fillna("").str.split(",", expand=True).replace("", np.nan).to_numpy(dtype=float)
    val_df = val_series.fillna("").str.split(",", expand=True).replace("", np.nan).to_numpy(dtype=float)

    if isinstance(target_conds, (list, set, tuple)):
        mask = np.isin(cond_df, list(target_conds))
    else:
        mask = (cond_df == target_conds)

    masked_vals = np.where(mask, val_df, np.nan)

    with np.errstate(invalid="ignore"):
        if operation == 'mean':
            sums = np.nansum(masked_vals, axis=1)
            counts = np.sum(mask, axis=1)
            out = sums / np.where(counts == 0, np.nan, counts)
        elif operation == 'std':
            out = np.nanstd(masked_vals, axis=1)
        elif operation in ('rate', 'rate_yn'):
            corrects = np.nansum(np.where(masked_vals == 1, 1, 0), axis=1)
            total = np.sum(mask, axis=1)
            out = corrects / np.where(total == 0, np.nan, total)
        else:
            sums = np.nansum(masked_vals, axis=1)
            counts = np.sum(mask, axis=1)
            out = sums / np.where(counts == 0, np.nan, counts)

    return pd.Series(out, index=cond_series.index)


def seq_rate_A3(series, target_codes):
    """A3-5: valid/invalid + correct/incorrect 조합코드 비율"""
    def calc(x):
        if not x:
            return np.nan
        s = x.split(',')
        correct = sum(s.count(code) for code in target_codes if code in ['1', '3'])
        incorrect = sum(s.count(code) for code in target_codes if code in ['2', '4'])
        total = correct + incorrect
        return correct / total if total > 0 else np.nan

    return series.fillna("").progress_apply(calc)


def seq_rate_B1_B2(series, target_codes):
    """B1-3, B2-3: change / non-change 조건 정확도"""
    def calc(x):
        if not x:
            return np.nan
        s = x.split(',')
        correct = sum(s.count(code) for code in target_codes if code in ['1', '3'])
        incorrect = sum(s.count(code) for code in target_codes if code in ['2', '4'])
        total = correct + incorrect
        return correct / total if total > 0 else np.nan

    return series.fillna("").progress_apply(calc)


def seq_rate_B4(series, target_codes):
    """B4-1: Flanker congruent/incongruent 정확도"""
    def calc(x):
        if not x:
            return np.nan
        s = x.split(',')
        correct = sum(s.count(code) for code in target_codes if code in ['1', '3', '5'])
        incorrect = sum(s.count(code) for code in target_codes if code in ['2', '4', '6'])
        total = correct + incorrect
        return correct / total if total > 0 else np.nan

    return series.fillna("").progress_apply(calc)


def seq_rate_simple(series):
    """B3, B5, B6, B7, B8 등 1=정답, 2=오답"""
    def calc(x):
        if not x:
            return np.nan
        s = x.split(',')
        correct = s.count('1')
        incorrect = s.count('2')
        total = correct + incorrect
        return correct / total if total > 0 else np.nan

    return series.fillna("").progress_apply(calc)


# =======================
# 1차 Feature Engineering (A/B)
# =======================

def preprocess_A(df: pd.DataFrame) -> pd.DataFrame:
    print("Step 1 (A): Age, TestDate 파생...")

    # 방어: 필수 컬럼 없으면 더미 생성
    if "Age" not in df.columns:
        df["Age"] = np.nan
    if "TestDate" not in df.columns:
        df["TestDate"] = np.nan

    df["Age_num"] = df["Age"].map(convert_age)
    ym = df["TestDate"].map(split_testdate)
    df["Year"] = [y for y, m in ym]
    df["Month"] = [m for y, m in ym]

    feats = pd.DataFrame(index=df.index)
    print("Step 2 (A): A1~A5 features (도메인 피처)...")

    # A1 (속도 예측)
    feats["A1_rt_mean"] = seq_mean(df["A1-4"])
    feats["A1_rt_std"] = seq_std(df["A1-4"])
    feats["A1_rt_left"] = masked_operation(df["A1-1"], df["A1-4"], 1, 'mean')
    feats["A1_rt_right"] = masked_operation(df["A1-1"], df["A1-4"], 2, 'mean')
    feats["A1_rt_slow"] = masked_operation(df["A1-2"], df["A1-4"], 1, 'mean')
    feats["A1_rt_norm"] = masked_operation(df["A1-2"], df["A1-4"], 2, 'mean')
    feats["A1_rt_fast"] = masked_operation(df["A1-2"], df["A1-4"], 3, 'mean')
    feats["A1_acc_slow"] = masked_operation(df["A1-2"], df["A1-3"], 1, 'rate')
    feats["A1_acc_norm"] = masked_operation(df["A1-2"], df["A1-3"], 2, 'rate')
    feats["A1_acc_fast"] = masked_operation(df["A1-2"], df["A1-3"], 3, 'rate')

    # A2 (정지 예측)
    feats["A2_rt_mean"] = seq_mean(df["A2-4"])
    feats["A2_rt_std"] = seq_std(df["A2-4"])
    feats["A2_rt_slow_c1"] = masked_operation(df["A2-1"], df["A2-4"], 1, 'mean')
    feats["A2_rt_norm_c1"] = masked_operation(df["A2-1"], df["A2-4"], 2, 'mean')
    feats["A2_rt_fast_c1"] = masked_operation(df["A2-1"], df["A2-4"], 3, 'mean')
    feats["A2_rt_slow_c2"] = masked_operation(df["A2-2"], df["A2-4"], 1, 'mean')
    feats["A2_rt_norm_c2"] = masked_operation(df["A2-2"], df["A2-4"], 2, 'mean')
    feats["A2_rt_fast_c2"] = masked_operation(df["A2-2"], df["A2-4"], 3, 'mean')
    feats["A2_acc_slow"] = masked_operation(df["A2-1"], df["A2-3"], 1, 'rate')
    feats["A2_acc_norm"] = masked_operation(df["A2-1"], df["A2-3"], 2, 'rate')
    feats["A2_acc_fast"] = masked_operation(df["A2-1"], df["A2-3"], 3, 'rate')

    # A3 (주의 전환)
    feats["A3_valid_acc"] = seq_rate_A3(df["A3-5"], ['1', '2'])
    feats["A3_invalid_acc"] = seq_rate_A3(df["A3-5"], ['3', '4'])
    feats["A3_rt_mean"] = seq_mean(df["A3-7"])
    feats["A3_rt_std"] = seq_std(df["A3-7"])
    feats["A3_rt_small"] = masked_operation(df["A3-1"], df["A3-7"], 1, 'mean')
    feats["A3_rt_big"] = masked_operation(df["A3-1"], df["A3-7"], 2, 'mean')
    feats["A3_rt_left"] = masked_operation(df["A3-3"], df["A3-7"], 1, 'mean')
    feats["A3_rt_right"] = masked_operation(df["A3-3"], df["A3-7"], 2, 'mean')

    # A4 (Stroop)
    feats["A4_rt_mean"] = seq_mean(df["A4-5"])
    feats["A4_rt_std"] = seq_std(df["A4-5"])
    feats["A4_rt_congruent"] = masked_operation(df["A4-1"], df["A4-5"], 1, 'mean')
    feats["A4_rt_incongruent"] = masked_operation(df["A4-1"], df["A4-5"], 2, 'mean')
    feats["A4_acc_congruent"] = masked_operation(df["A4-1"], df["A4-3"], 1, 'rate')
    feats["A4_acc_incongruent"] = masked_operation(df["A4-1"], df["A4-3"], 2, 'rate')

    # A5 (변화 탐지)
    feats["A5_acc_nonchange"] = masked_operation(df["A5-1"], df["A5-2"], 1, 'rate')
    feats["A5_acc_pos_change"] = masked_operation(df["A5-1"], df["A5-2"], 2, 'rate')
    feats["A5_acc_color_change"] = masked_operation(df["A5-1"], df["A5-2"], 3, 'rate')
    feats["A5_acc_shape_change"] = masked_operation(df["A5-1"], df["A5-2"], 4, 'rate')

    # A6, A7
    feats["A6_correct_count"] = df["A6-1"]
    feats["A7_correct_count"] = df["A7-1"]

    # A8, A9 (원래 컬럼명 유지)
    feats["A8-1"] = df["A8-1"]
    feats["A8-2"] = df["A8-2"]
    feats["A9-1"] = df["A9-1"]
    feats["A9-2"] = df["A9-2"]
    feats["A9-3"] = df["A9-3"]
    feats["A9-4"] = df["A9-4"]
    feats["A9-5"] = df["A9-5"]

    seq_cols = [
        "A1-1","A1-2","A1-3","A1-4",
        "A2-1","A2-2","A2-3","A2-4",
        "A3-1","A3-2","A3-3","A3-4","A3-5","A3-6","A3-7",
        "A4-1","A4-2","A4-3","A4-4","A4-5",
        "A5-1","A5-2","A5-3",
        "A6-1","A7-1","A8-1","A8-2",
        "A9-1","A9-2","A9-3","A9-4","A9-5"
    ]
    print("A 검사 1차 전처리 완료")
    return pd.concat([df, feats], axis=1).drop(columns=seq_cols, errors="ignore")


def preprocess_B(df: pd.DataFrame) -> pd.DataFrame:
    print("Step 1 (B): Age, TestDate 파생...")

    if "Age" not in df.columns:
        df["Age"] = np.nan
    if "TestDate" not in df.columns:
        df["TestDate"] = np.nan

    df["Age_num"] = df["Age"].map(convert_age)
    ym = df["TestDate"].map(split_testdate)
    df["Year"] = [y for y, m in ym]
    df["Month"] = [m for y, m in ym]

    feats = pd.DataFrame(index=df.index)
    print("Step 2 (B): B1~B10 features (도메인 피처)...")

    # B1, B2
    feats["B1_task1_acc"] = seq_rate_simple(df["B1-1"])
    feats["B1_rt_mean"] = seq_mean(df["B1-2"])
    feats["B1_rt_std"] = seq_std(df["B1-2"])
    feats["B1_change_acc"] = seq_rate_B1_B2(df["B1-3"], ['1', '2'])
    feats["B1_nonchange_acc"] = seq_rate_B1_B2(df["B1-3"], ['3', '4'])

    feats["B2_task1_acc"] = seq_rate_simple(df["B2-1"])
    feats["B2_rt_mean"] = seq_mean(df["B2-2"])
    feats["B2_rt_std"] = seq_std(df["B2-2"])
    feats["B2_change_acc"] = seq_rate_B1_B2(df["B2-3"], ['1', '2'])
    feats["B2_nonchange_acc"] = seq_rate_B1_B2(df["B2-3"], ['3', '4'])

    # B3
    feats["B3_acc_rate"] = seq_rate_simple(df["B3-1"])
    feats["B3_rt_mean"] = seq_mean(df["B3-2"])
    feats["B3_rt_std"] = seq_std(df["B3-2"])

    # B4
    feats["B4_congruent_acc"] = seq_rate_B4(df["B4-1"], ['1', '2'])
    feats["B4_incongruent_acc"] = seq_rate_B4(df["B4-1"], ['3', '4', '5', '6'])
    feats["B4_rt_mean"] = seq_mean(df["B4-2"])
    feats["B4_rt_std"] = seq_std(df["B4-2"])

    # B5, B6, B7, B8
    feats["B5_acc_rate"] = seq_rate_simple(df["B5-1"])
    feats["B5_rt_mean"] = seq_mean(df["B5-2"])
    feats["B5_rt_std"] = seq_std(df["B5-2"])
    feats["B6_acc_rate"] = seq_rate_simple(df["B6"])
    feats["B7_acc_rate"] = seq_rate_simple(df["B7"])
    feats["B8_acc_rate"] = seq_rate_simple(df["B8"])

    # B9, B10 (원래 컬럼명 유지)
    feats["B9-1"] = df["B9-1"]
    feats["B9-2"] = df["B9-2"]
    feats["B9-3"] = df["B9-3"]
    feats["B9-4"] = df["B9-4"]
    feats["B9-5"] = df["B9-5"]

    feats["B10-1"] = df["B10-1"]
    feats["B10-2"] = df["B10-2"]
    feats["B10-3"] = df["B10-3"]
    feats["B10-4"] = df["B10-4"]
    feats["B10-5"] = df["B10-5"]
    feats["B10-6"] = df["B10-6"]

    seq_cols = [
        "B1-1","B1-2","B1-3",
        "B2-1","B2-2","B2-3",
        "B3-1","B3-2",
        "B4-1","B4-2",
        "B5-1","B5-2",
        "B6","B7","B8",
        "B9-1","B9-2","B9-3","B9-4","B9-5",
        "B10-1","B10-2","B10-3","B10-4","B10-5","B10-6"
    ]
    print("B 검사 1차 전처리 완료")
    return pd.concat([df, feats], axis=1).drop(columns=seq_cols, errors="ignore")


# =======================
# 2차 Feature Engineering
# =======================

def _has(df, cols):
    return all(c in df.columns for c in cols)


def _safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def add_features_A(df: pd.DataFrame) -> pd.DataFrame:
    feats = df.copy()
    eps = 1e-6

    if _has(feats, ["Year", "Month"]):
        feats["YearMonthIndex"] = feats["Year"] * 12 + feats["Month"]

    if _has(feats, ["A1_rt_mean", "A1_acc_norm"]):
        feats["A1_speed_acc_tradeoff"] = _safe_div(feats["A1_rt_mean"], feats["A1_acc_norm"], eps)
    if _has(feats, ["A2_rt_mean", "A2_acc_norm"]):
        feats["A2_speed_acc_tradeoff"] = _safe_div(feats["A2_rt_mean"], feats["A2_acc_norm"], eps)
    if _has(feats, ["A4_rt_mean", "A4_acc_congruent"]):
        feats["A4_speed_acc_tradeoff"] = _safe_div(feats["A4_rt_mean"], feats["A4_acc_congruent"], eps)

    for k in ["A1", "A2", "A3", "A4"]:
        m, s = f"{k}_rt_mean", f"{k}_rt_std"
        if _has(feats, [m, s]):
            feats[f"{k}_rt_cv"] = _safe_div(feats[s], feats[m], eps)

    if _has(feats, ["A1_rt_fast", "A1_rt_slow"]):
        feats["A1_rt_speed_cost"] = feats["A1_rt_fast"] - feats["A1_rt_slow"]
    if _has(feats, ["A1_acc_fast", "A1_acc_slow"]):
        feats["A1_acc_speed_cost"] = feats["A1_acc_fast"] - feats["A1_acc_slow"]
    if _has(feats, ["A2_rt_fast_c1", "A2_rt_slow_c1"]):
        feats["A2_rt_speed_cost_c1"] = feats["A2_rt_fast_c1"] - feats["A2_rt_slow_c1"]
    if _has(feats, ["A2_acc_fast", "A2_acc_slow"]):
        feats["A2_acc_speed_cost"] = feats["A2_acc_fast"] - feats["A2_acc_slow"]

    if _has(feats, ["A3_rt_big", "A3_rt_small"]):
        feats["A3_rt_size_cost"] = feats["A3_rt_big"] - feats["A3_rt_small"]
    if _has(feats, ["A3_valid_acc", "A3_invalid_acc"]):
        feats["A3_acc_attention_cost"] = feats["A3_valid_acc"] - feats["A3_invalid_acc"]

    if _has(feats, ["A4_rt_incongruent", "A4_rt_congruent"]):
        feats["A4_stroop_rt_cost"] = feats["A4_rt_incongruent"] - feats["A4_rt_congruent"]
    if _has(feats, ["A4_acc_congruent", "A4_acc_incongruent"]):
        feats["A4_stroop_acc_cost"] = feats["A4_acc_congruent"] - feats["A4_acc_incongruent"]

    if _has(feats, ["A5_acc_nonchange", "A5_acc_pos_change"]):
        feats["A5_acc_cost_pos"] = feats["A5_acc_nonchange"] - feats["A5_acc_pos_change"]
    if _has(feats, ["A5_acc_nonchange", "A5_acc_color_change"]):
        feats["A5_acc_cost_color"] = feats["A5_acc_nonchange"] - feats["A5_acc_color_change"]
    if _has(feats, ["A5_acc_nonchange", "A5_acc_shape_change"]):
        feats["A5_acc_cost_shape"] = feats["A5_acc_nonchange"] - feats["A5_acc_shape_change"]

    if _has(feats, ["A3_valid_acc", "A3_invalid_acc", "A4_acc_congruent", "A4_acc_incongruent"]):
        feats["A_selective_attention_index"] = (
            (feats["A3_valid_acc"] - feats["A3_invalid_acc"]).fillna(0) +
            (feats["A4_acc_congruent"] - feats["A4_acc_incongruent"]).fillna(0)
        )

    wm_cols = [c for c in [
        "A5_acc_nonchange", "A5_acc_pos_change",
        "A5_acc_color_change", "A5_acc_shape_change"
    ] if c in feats.columns]
    if wm_cols:
        wm_mat = feats[wm_cols].apply(pd.to_numeric, errors="coerce")
        feats["A_working_memory_index"] = wm_mat.mean(axis=1)

    cog_cols_A = [c for c in [
        "A6_correct_count", "A7_correct_count",
        "A8-1", "A8-2",
        "A9-1", "A9-2", "A9-3", "A9-4", "A9-5"
    ] if c in feats.columns]
    if cog_cols_A:
        cog_mat = feats[cog_cols_A].apply(pd.to_numeric, errors="coerce")
        feats["A_cog_sum"] = cog_mat.sum(axis=1)
        feats["A_cog_mean"] = cog_mat.mean(axis=1)

    parts = []
    if "A4_stroop_rt_cost" in feats:
        parts.append(0.30 * feats["A4_stroop_rt_cost"].fillna(0))
    if "A4_stroop_acc_cost" in feats:
        parts.append(0.20 * (1 - feats["A4_stroop_acc_cost"].fillna(1)))
    if "A3_acc_attention_cost" in feats:
        parts.append(0.20 * feats["A3_acc_attention_cost"].fillna(0).abs())
    if "A1_rt_cv" in feats:
        parts.append(0.20 * feats["A1_rt_cv"].fillna(0))
    if "A2_rt_cv" in feats:
        parts.append(0.10 * feats["A2_rt_cv"].fillna(0))
    if "A5_acc_cost_pos" in feats:
        parts.append(0.10 * feats["A5_acc_cost_pos"].fillna(0).clip(lower=0))
    if "A5_acc_cost_color" in feats:
        parts.append(0.10 * feats["A5_acc_cost_color"].fillna(0).clip(lower=0))
    if "A5_acc_cost_shape" in feats:
        parts.append(0.10 * feats["A5_acc_cost_shape"].fillna(0).clip(lower=0))

    if parts:
        feats["RiskScore"] = sum(parts)

    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats


def add_features_B(df: pd.DataFrame) -> pd.DataFrame:
    feats = df.copy()
    eps = 1e-6

    if _has(feats, ["Year", "Month"]):
        feats["YearMonthIndex"] = feats["Year"] * 12 + feats["Month"]

    for k, acc_col, rt_col in [
        ("B1", "B1_task1_acc", "B1_rt_mean"),
        ("B2", "B2_task1_acc", "B2_rt_mean"),
        ("B3", "B3_acc_rate", "B3_rt_mean"),
        ("B5", "B5_acc_rate", "B5_rt_mean"),
    ]:
        if _has(feats, [rt_col, acc_col]):
            feats[f"{k}_speed_acc_tradeoff"] = _safe_div(feats[rt_col], feats[acc_col], eps)

    if _has(feats, ["B4_rt_mean", "B4_congruent_acc"]):
        feats["B4_speed_acc_tradeoff"] = _safe_div(feats["B4_rt_mean"], feats["B4_congruent_acc"], eps)

    for k in ["B1", "B2", "B3", "B4", "B5"]:
        m, s = f"{k}_rt_mean", f"{k}_rt_std"
        if _has(feats, [m, s]):
            feats[f"{k}_rt_cv"] = _safe_div(feats[s], feats[m], eps)

    if _has(feats, ["B1_change_acc", "B1_nonchange_acc"]):
        feats["B1_acc_cost"] = feats["B1_nonchange_acc"] - feats["B1_change_acc"]
    if _has(feats, ["B2_change_acc", "B2_nonchange_acc"]):
        feats["B2_acc_cost"] = feats["B2_nonchange_acc"] - feats["B2_change_acc"]
    if _has(feats, ["B4_congruent_acc", "B4_incongruent_acc"]):
        feats["B4_flanker_acc_cost"] = feats["B4_congruent_acc"] - feats["B4_incongruent_acc"]

    rt_cv_cols = [c for c in [
        "B1_rt_cv", "B2_rt_cv", "B3_rt_cv", "B4_rt_cv", "B5_rt_cv"
    ] if c in feats.columns]
    if rt_cv_cols:
        cv_mat = feats[rt_cv_cols].apply(pd.to_numeric, errors="coerce")
        feats["B_visuomotor_variability"] = cv_mat.mean(axis=1)

    acc_simple_cols = [c for c in [
        "B3_acc_rate", "B5_acc_rate",
        "B6_acc_rate", "B7_acc_rate", "B8_acc_rate"
    ] if c in feats.columns]
    if acc_simple_cols:
        acc_mat = feats[acc_simple_cols].apply(pd.to_numeric, errors="coerce")
        feats["B_reaction_overall"] = acc_mat.mean(axis=1)

    exec_parts = []
    if "B4_flanker_acc_cost" in feats:
        exec_parts.append((1 - feats["B4_flanker_acc_cost"]).fillna(0))
    for c in ["B1_acc_cost", "B2_acc_cost"]:
        if c in feats:
            exec_parts.append((1 - feats[c]).fillna(0))
    if exec_parts:
        exec_mat = pd.concat(exec_parts, axis=1)
        feats["B_executive_control_index"] = exec_mat.mean(axis=1)

    b9_cols = [c for c in ["B9-1", "B9-2", "B9-3", "B9-4", "B9-5"] if c in feats.columns]
    if b9_cols:
        b9_mat = feats[b9_cols].apply(pd.to_numeric, errors="coerce")
        feats["B9_sum"] = b9_mat.sum(axis=1)
        feats["B9_mean"] = b9_mat.mean(axis=1)

    b10_cols = [c for c in ["B10-1", "B10-2", "B10-3", "B10-4", "B10-5", "B10-6"] if c in feats.columns]
    if b10_cols:
        b10_mat = feats[b10_cols].apply(pd.to_numeric, errors="coerce")
        feats["B10_sum"] = b10_mat.sum(axis=1)
        feats["B10_mean"] = b10_mat.mean(axis=1)

    cog_cols_B = list(set(b9_cols + b10_cols))
    if cog_cols_B:
        cog_mat_B = feats[cog_cols_B].apply(pd.to_numeric, errors="coerce")
        feats["B_cog_sum"] = cog_mat_B.sum(axis=1)
        feats["B_cog_mean"] = cog_mat_B.mean(axis=1)

    parts = []
    for k in ["B4", "B5"]:
        cv_col = f"{k}_rt_cv"
        if cv_col in feats:
            parts.append(0.20 * feats[cv_col].fillna(0))
    for k in ["B3", "B5"]:
        acc = f"{k}_acc_rate"
        if acc in feats:
            parts.append(0.20 * (1 - feats[acc].fillna(1)))
    if "B4_flanker_acc_cost" in feats:
        parts.append(0.20 * (1 - feats["B4_flanker_acc_cost"].fillna(1)))
    for k in ["B1", "B2"]:
        acc = f"{k}_task1_acc"
        if acc in feats:
            parts.append(0.10 * (1 - feats[acc].fillna(1)))
        tcol = f"{k}_speed_acc_tradeoff"
        if tcol in feats:
            parts.append(0.10 * feats[tcol].fillna(0))
    if parts:
        feats["RiskScore_B"] = sum(parts)

    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats


# =======================
# 3. 피처 정렬 (CatBoost용)
# =======================

CAT_FEATURES = ['Age', 'PrimaryKey']
DROP_COLS = ["Test_id", "Test", "TestDate", "Year", "Month"]


def align_features_A(X_df: pd.DataFrame, cat_model) -> pd.DataFrame:
    """A모델용 피처 정렬 (PK Stats 미포함)"""
    base_cols = [c for c in X_df.columns if not c.startswith('pk_')]
    feat_names = list(getattr(cat_model, "feature_names_", []))

    if not feat_names:
        print("경고: A(CatBoost) 피처 이름을 찾을 수 없습니다.")
        return pd.DataFrame(index=X_df.index)

    drop_cols_align = [c for c in DROP_COLS if c in X_df.columns] + ['Test_x', 'Test_y']
    cat_X = X_df[base_cols].drop(columns=drop_cols_align, errors="ignore").copy()

    for c in feat_names:
        if c not in cat_X.columns:
            if c in CAT_FEATURES:
                cat_X[c] = 'nan'
            else:
                cat_X[c] = 0.0

    for col in CAT_FEATURES:
        if col in cat_X.columns:
            cat_X[col] = cat_X[col].fillna('nan').astype(str)

    cat_X = cat_X[feat_names]

    num_cols = [c for c in feat_names if c not in CAT_FEATURES]
    if num_cols:
        cat_X[num_cols] = cat_X[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return cat_X


def align_features_B(X_df: pd.DataFrame, cat_model) -> pd.DataFrame:
    """B모델용 피처 정렬 (PK Stats 포함)"""
    feat_names = list(getattr(cat_model, "feature_names_", []))
    drop_cols_align = [c for c in DROP_COLS if c in X_df.columns] + ['Test_x', 'Test_y']

    if not feat_names:
        print("경고: B(CatBoost) 피처 이름을 찾을 수 없습니다.")
        return pd.DataFrame(index=X_df.index)

    cat_X = X_df.drop(columns=drop_cols_align, errors="ignore").copy()

    missing = set(feat_names) - set(cat_X.columns)
    if missing:
        for c in missing:
            if c in CAT_FEATURES:
                cat_X[c] = 'nan'
            else:
                cat_X[c] = 0.0

    for col in CAT_FEATURES:
        if col in cat_X.columns:
            cat_X[col] = cat_X[col].fillna('nan').astype(str)

    cat_X = cat_X[feat_names]

    num_cols = [c for c in feat_names if c not in CAT_FEATURES]
    if num_cols:
        cat_X[num_cols] = cat_X[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return cat_X


# =======================
# 4. main (추론)
# =======================

def main():
    N_SPLITS = 5

    BASE_DIR = os.environ.get("OPEN_DATA_PATH", "./data")
    TEST_META_PATH = os.path.join(BASE_DIR, "test.csv")
    TEST_A_PATH = os.path.join(BASE_DIR, "test", "A.csv")
    TEST_B_PATH = os.path.join(BASE_DIR, "test", "B.csv")
    SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "sample_submission.csv")

    MODEL_DIR = "./model"
    OUT_DIR = os.environ.get("OUTPUT_DATA_PATH", "./output")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    # 운전자 통계 피처
    PK_STATS_PATH = os.path.join(MODEL_DIR, "pk_stats_final.csv")
    try:
        pk_stats = pd.read_csv(PK_STATS_PATH)
        print(f"운전자 통계 피처 로드 완료: {pk_stats.shape}")
    except FileNotFoundError:
        print(f"경고: {PK_STATS_PATH} 파일이 없어, PK Stats 없이 진행합니다.")
        pk_stats = pd.DataFrame()

    # Fold별 CatBoost 모델 로드
    print(f"{N_SPLITS}-Fold CatBoost 모델 로드...")
    models_A = []
    models_B = []
    for fold in range(N_SPLITS):
        try:
            cat_A = joblib.load(os.path.join(MODEL_DIR, f"catboost_A_fold{fold}.pkl"))
        except FileNotFoundError:
            cat_A = None
        models_A.append(cat_A)

        try:
            cat_B = joblib.load(os.path.join(MODEL_DIR, f"catboost_B_fold{fold}.pkl"))
        except FileNotFoundError:
            cat_B = None
        models_B.append(cat_B)

    # Global Isotonic 보정기
    cal_A_global = None
    cal_B_global = None
    path_A_global = os.path.join(MODEL_DIR, "calibrator_A_global.pkl")
    path_B_global = os.path.join(MODEL_DIR, "calibrator_B_global.pkl")

    if os.path.exists(path_A_global):
        cal_A_global = joblib.load(path_A_global)
        print("Global A Isotonic 보정기 로드 완료.")
    else:
        print("Global A Isotonic 보정기 없음 → 비보정 확률 사용.")

    if os.path.exists(path_B_global):
        cal_B_global = joblib.load(path_B_global)
        print("Global B Isotonic 보정기 로드 완료.")
    else:
        print("Global B Isotonic 보정기 없음 → 비보정 확률 사용.")

    # 테스트 데이터 로드
    print("테스트 데이터 로드...")
    meta = pd.read_csv(TEST_META_PATH)
    Araw = pd.read_csv(TEST_A_PATH)
    Braw = pd.read_csv(TEST_B_PATH)
    print(f" meta={len(meta)}, Araw={len(Araw)}, Braw={len(Braw)}")

    if "Test" not in meta.columns:
        raise KeyError("meta(test.csv)에 'Test' 컬럼이 없습니다.")

    # A/B 메타 분리
    if "Test_id" in meta.columns and "Test" in meta.columns:
        A_meta = meta[meta["Test"] == "A"][["Test_id", "Test"]].copy()
        B_meta = meta[meta["Test"] == "B"][["Test_id", "Test"]].copy()
    else:
        A_meta = meta[meta["Test"] == "A"].copy()
        B_meta = meta[meta["Test"] == "B"].copy()

    # A/B 매핑
    A_df = A_meta.merge(Araw, on="Test_id", how="left")
    B_df = B_meta.merge(Braw, on="Test_id", how="left")
    print(f" mapped: A={len(A_df)}, B={len(B_df)}")

    # 핵심 컬럼 보정
    def ensure_core_cols(df, name):
        if "PrimaryKey" not in df.columns:
            if "PrimaryKey_x" in df.columns:
                df["PrimaryKey"] = df["PrimaryKey_x"]
            elif "PrimaryKey_y" in df.columns:
                df["PrimaryKey"] = df["PrimaryKey_y"]
            else:
                df["PrimaryKey"] = -1
        if "Age" not in df.columns:
            if "Age_x" in df.columns:
                df["Age"] = df["Age_x"]
            elif "Age_y" in df.columns:
                df["Age"] = df["Age_y"]
            else:
                df["Age"] = np.nan
        if "TestDate" not in df.columns:
            if "TestDate_x" in df.columns:
                df["TestDate"] = df["TestDate_x"]
            elif "TestDate_y" in df.columns:
                df["TestDate"] = df["TestDate_y"]
            else:
                df["TestDate"] = np.nan
        return df

    A_df = ensure_core_cols(A_df, "A")
    B_df = ensure_core_cols(B_df, "B")

    # 전처리 (1차 + 2차)
    print("\n[INFO] Preprocessing A (1차, 2차)...")
    A_feat = add_features_A(preprocess_A(A_df)) if len(A_df) > 0 else pd.DataFrame()
    print("\n[INFO] Preprocessing B (1차, 2차)...")
    B_feat = add_features_B(preprocess_B(B_df)) if len(B_df) > 0 else pd.DataFrame()

    # PK Stats 병합 (B만)
    print("\n[INFO] 운전자 통계 피처 병합...")
    if len(B_feat) > 0 and not pk_stats.empty:
        if "PrimaryKey" not in B_feat.columns:
            B_feat["PrimaryKey"] = -1
        B_feat = B_feat.merge(pk_stats, on='PrimaryKey', how='left')
        print("B 모델에 PK Stats 병합 완료.")
    else:
        print("PK Stats 병합 생략 (데이터 없음 또는 파일 없음).")

    # K-Fold 예측
    print("\n[INFO] Inference K-Fold Ensemble...")
    preds_A_raw_folds = []
    preds_B_raw_folds = []

    for fold in range(N_SPLITS):
        print(f"--- Fold {fold+1}/{N_SPLITS} 예측 ---")

        # A
        cat_A = models_A[fold]
        if len(A_feat) == 0 or cat_A is None:
            raw_A = np.zeros(len(A_feat))
        else:
            XA = align_features_A(A_feat, cat_A)
            if XA.shape[0] == 0 or XA.shape[1] == 0:
                raw_A = np.zeros(len(A_feat))
            else:
                raw_A = cat_A.predict_proba(XA)[:, 1]
        preds_A_raw_folds.append(raw_A)

        # B
        cat_B = models_B[fold]
        if len(B_feat) == 0 or cat_B is None:
            raw_B = np.zeros(len(B_feat))
        else:
            XB = align_features_B(B_feat, cat_B)
            if XB.shape[0] == 0 or XB.shape[1] == 0:
                raw_B = np.zeros(len(B_feat))
            else:
                raw_B = cat_B.predict_proba(XB)[:, 1]
        preds_B_raw_folds.append(raw_B)

    # Fold 평균 + Global Isotonic 적용
    if len(A_feat) > 0:
        mean_A_raw = np.mean(preds_A_raw_folds, axis=0)
        if cal_A_global is not None:
            final_predA = cal_A_global.predict(mean_A_raw)
        else:
            final_predA = mean_A_raw
    else:
        final_predA = np.array([])

    if len(B_feat) > 0:
        mean_B_raw = np.mean(preds_B_raw_folds, axis=0)
        if cal_B_global is not None:
            final_predB = cal_B_global.predict(mean_B_raw)
        else:
            final_predB = mean_B_raw
    else:
        final_predB = np.array([])

    # 안전장치: NaN/inf 제거 + [0,1] 클리핑
    final_predA = np.nan_to_num(final_predA.astype(float), nan=0.0, posinf=1.0, neginf=0.0)
    final_predB = np.nan_to_num(final_predB.astype(float), nan=0.0, posinf=1.0, neginf=0.0)
    final_predA = np.clip(final_predA, 0.0, 1.0)
    final_predB = np.clip(final_predB, 0.0, 1.0)

    # Test_id 매핑
    subA = pd.DataFrame({"Test_id": A_df["Test_id"].values, "prob": final_predA})
    subB = pd.DataFrame({"Test_id": B_df["Test_id"].values, "prob": final_predB})
    probs = pd.concat([subA, subB], axis=0, ignore_index=True)

    # sample_submission 포맷 맞추기
    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        sample = pd.read_csv(SAMPLE_SUB_PATH)
    except FileNotFoundError:
        print("경고: sample_submission.csv가 없어 비상 생성합니다.")
        sample = pd.DataFrame({"Test_id": meta["Test_id"].values, "Label": 0.0})

    out = sample.drop(columns=["Label"], errors="ignore").merge(probs, on="Test_id", how="left")
    out["Label"] = out["prob"].astype(float).fillna(0.0)
    out = out.drop(columns=["prob"])

    # Label 값 방어
    out["Label"] = np.nan_to_num(out["Label"].astype(float), nan=0.0, posinf=1.0, neginf=0.0)
    out["Label"] = np.clip(out["Label"], 0.0, 1.0)

    # 컬럼 순서 sample과 동일
    out = out[sample.columns]

    out.to_csv(OUT_PATH, index=False)
    print(f"✅ Saved: {OUT_PATH} (rows={len(out)})")


if __name__ == "__main__":
    main()
