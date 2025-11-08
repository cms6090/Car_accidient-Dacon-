## ----------------------------------------------------------------
## [Jupyter용] 1_Preprocess.ipynb (PDF 도메인 지식 총동원)
##
## 목적: PDF 명세서 기반으로 '진짜' 피처를 생성하고 Feather 파일로 저장
## ----------------------------------------------------------------

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import joblib 

tqdm.pandas()
warnings.filterwarnings('ignore')

print("데이터 전처리 파이프라인 시작 (시간이 오래 걸립니다)...")

## 1. 로컬 경로 설정 및 데이터 로드
BASE_DIR = "./data"
MODEL_SAVE_DIR = "./model" 
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

try:
    train_meta = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
    train_A_raw = pd.read_csv(os.path.join(BASE_DIR, "train", "A.csv"))
    train_B_raw = pd.read_csv(os.path.join(BASE_DIR, "train", "B.csv"))
except FileNotFoundError:
    print(f"경고: '{BASE_DIR}' 경로에 파일이 없습니다. data 폴더에 원본 데이터를 넣어주세요.")
    raise
print("데이터 로드 완료:", train_meta.shape, train_A_raw.shape, train_B_raw.shape)


## 2. 전처리 유틸 (안전한 progress_apply 버전)
def convert_age(val):
    if pd.isna(val): return np.nan
    try:
        base = int(str(val)[:-1])
        return base if str(val)[-1] == "a" else base + 5
    except:
        return np.nan
def split_testdate(val):
    try:
        v = int(val)
        return v // 100, v % 100
    except:
        return np.nan, np.nan
def seq_mean(series):
    return series.fillna("").progress_apply(
        lambda x: np.fromstring(x, sep=",").mean() if x else np.nan
    )
def seq_std(series):
    return series.fillna("").progress_apply(
        lambda x: np.fromstring(x, sep=",").std() if x else np.nan
    )

# [!!!!! 신규 도메인 피처 함수 !!!!!]

def masked_operation(cond_series, val_series, target_conds, operation='mean'):
    """
    조건 시리즈(cond_series)에서 target_conds에 해당하는 값들의
    값 시리즈(val_series)에서의 통계치(mean, std, or rate)를 계산
    """
    # 데이터프레임으로 변환
    cond_df = cond_series.fillna("").str.split(",", expand=True).replace("", np.nan).to_numpy(dtype=float)
    val_df = val_series.fillna("").str.split(",", expand=True).replace("", np.nan).to_numpy(dtype=float)
    
    # 조건에 맞는 마스크 생성
    if isinstance(target_conds, (list, set)):
        mask = np.isin(cond_df, list(target_conds))
    else:
        mask = (cond_df == target_conds)
        
    # 마스크된 값만 추출
    masked_vals = np.where(mask, val_df, np.nan)
    
    with np.errstate(invalid="ignore"):
        if operation == 'mean':
            sums = np.nansum(masked_vals, axis=1)
            counts = np.sum(mask, axis=1)
            out = sums / np.where(counts == 0, np.nan, counts)
        elif operation == 'std':
            out = np.nanstd(masked_vals, axis=1)
        elif operation == 'rate': # val_series가 응답(1=correct, 2=incorrect)일 때 사용
            corrects = np.nansum(np.where(masked_vals == 1, 1, 0), axis=1)
            total = np.sum(mask, axis=1)
            out = corrects / np.where(total == 0, np.nan, total)
            
    return pd.Series(out, index=cond_series.index)


# [!!!!! PDF 명세 기반 신규 함수 !!!!!]
def seq_rate_A3(series, target_codes):
    def calc(x):
        if not x: return np.nan
        s = x.split(',')
        correct = sum(s.count(code) for code in target_codes if code in ['1', '3'])
        incorrect = sum(s.count(code) for code in target_codes if code in ['2', '4'])
        total = correct + incorrect
        return correct / total if total > 0 else np.nan
    return series.fillna("").progress_apply(calc)

def seq_rate_B1_B2(series, target_codes):
    def calc(x):
        if not x: return np.nan
        s = x.split(',')
        correct = sum(s.count(code) for code in target_codes if code in ['1', '3'])
        incorrect = sum(s.count(code) for code in target_codes if code in ['2', '4'])
        total = correct + incorrect
        return correct / total if total > 0 else np.nan
    return series.fillna("").progress_apply(calc)

def seq_rate_B4(series, target_codes):
    def calc(x):
        if not x: return np.nan
        s = x.split(',')
        correct = sum(s.count(code) for code in target_codes if code in ['1', '3', '5'])
        incorrect = sum(s.count(code) for code in target_codes if code in ['2', '4', '6'])
        total = correct + incorrect
        return correct / total if total > 0 else np.nan
    return series.fillna("").progress_apply(calc)

def seq_rate_simple(series): # B3, B5, B6, B7, B8
    def calc(x):
        if not x: return np.nan
        s = x.split(',')
        correct = s.count('1')
        incorrect = s.count('2')
        total = correct + incorrect
        return correct / total if total > 0 else np.nan
    return series.fillna("").progress_apply(calc)
# [!!!!!!!!!!!!!!!!!!!!!!!!!!!]


## 3. 1차 Feature Engineering (도메인 적용)
def preprocess_A(df):
    print("Step 1 (A): Age, TestDate 파생...")
    df["Age_num"] = df["Age"].map(convert_age)
    ym = df["TestDate"].map(split_testdate); df["Year"] = [y for y, m in ym]; df["Month"] = [m for y, m in ym]
    feats = pd.DataFrame(index=df.index)
    print("Step 2 (A): A1~A5 features (신규 도메인 피처)...")
    
    # A1 (속도 예측) [cite: 128, 129]
    feats["A1_rt_mean"] = seq_mean(df["A1-4"]); feats["A1_rt_std"] = seq_std(df["A1-4"])
    feats["A1_rt_left"] = masked_operation(df["A1-1"], df["A1-4"], 1, 'mean')
    feats["A1_rt_right"] = masked_operation(df["A1-1"], df["A1-4"], 2, 'mean')
    feats["A1_rt_slow"] = masked_operation(df["A1-2"], df["A1-4"], 1, 'mean')
    feats["A1_rt_norm"] = masked_operation(df["A1-2"], df["A1-4"], 2, 'mean')
    feats["A1_rt_fast"] = masked_operation(df["A1-2"], df["A1-4"], 3, 'mean')
    feats["A1_acc_slow"] = masked_operation(df["A1-2"], df["A1-3"], 1, 'rate') # [신규] A1-3(응답)은 1=Y(정답), 0=N(오답)
    feats["A1_acc_norm"] = masked_operation(df["A1-2"], df["A1-3"], 2, 'rate') # [신규]
    feats["A1_acc_fast"] = masked_operation(df["A1-2"], df["A1-3"], 3, 'rate') # [신규]

    # A2 (정지 예측) [cite: 162-164]
    feats["A2_rt_mean"] = seq_mean(df["A2-4"]); feats["A2_rt_std"] = seq_std(df["A2-4"])
    feats["A2_rt_slow_c1"] = masked_operation(df["A2-1"], df["A2-4"], 1, 'mean') # C1=속도
    feats["A2_rt_norm_c1"] = masked_operation(df["A2-1"], df["A2-4"], 2, 'mean')
    feats["A2_rt_fast_c1"] = masked_operation(df["A2-1"], df["A2-4"], 3, 'mean')
    feats["A2_rt_slow_c2"] = masked_operation(df["A2-2"], df["A2-4"], 1, 'mean') # C2=가속도
    feats["A2_rt_norm_c2"] = masked_operation(df["A2-2"], df["A2-4"], 2, 'mean')
    feats["A2_rt_fast_c2"] = masked_operation(df["A2-2"], df["A2-4"], 3, 'mean')
    feats["A2_acc_slow"] = masked_operation(df["A2-1"], df["A2-3"], 1, 'rate') # [신규]
    feats["A2_acc_norm"] = masked_operation(df["A2-1"], df["A2-3"], 2, 'rate') # [신규]
    feats["A2_acc_fast"] = masked_operation(df["A2-1"], df["A2-3"], 3, 'rate') # [신규]

    # A3 (주의 전환) [cite: 190, 195]
    feats["A3_valid_acc"] = seq_rate_A3(df["A3-5"], ['1', '2']) # 1=valid correct, 2=valid incorrect
    feats["A3_invalid_acc"] = seq_rate_A3(df["A3-5"], ['3', '4']) # 3=invalid correct, 4=invalid incorrect
    feats["A3_rt_mean"] = seq_mean(df["A3-7"]); feats["A3_rt_std"] = seq_std(df["A3-7"])
    feats["A3_rt_small"] = masked_operation(df["A3-1"], df["A3-7"], 1, 'mean') # A3-1 (크기)
    feats["A3_rt_big"] = masked_operation(df["A3-1"], df["A3-7"], 2, 'mean')
    feats["A3_rt_left"] = masked_operation(df["A3-3"], df["A3-7"], 1, 'mean') # A3-3 (방향)
    feats["A3_rt_right"] = masked_operation(df["A3-3"], df["A3-7"], 2, 'mean')
    
    # A4 (Stroop) [cite: 223, 225, 227]
    feats["A4_rt_mean"] = seq_mean(df["A4-5"]); feats["A4_rt_std"] = seq_std(df["A4-5"])
    feats["A4_rt_congruent"] = masked_operation(df["A4-1"], df["A4-5"], 1, 'mean') # A4-1 (1=con, 2=incon)
    feats["A4_rt_incongruent"] = masked_operation(df["A4-1"], df["A4-5"], 2, 'mean')
    feats["A4_acc_congruent"] = masked_operation(df["A4-1"], df["A4-3"], 1, 'rate') # [신규] A4-3 (1=correct, 2=incorrect)
    feats["A4_acc_incongruent"] = masked_operation(df["A4-1"], df["A4-3"], 2, 'rate') # [신규]
    
    # A5 (변화 탐지) 
    feats["A5_acc_nonchange"] = masked_operation(df["A5-1"], df["A5-2"], 1, 'rate') # A5-1 (1=non-change)
    feats["A5_acc_pos_change"] = masked_operation(df["A5-1"], df["A5-2"], 2, 'rate') # [신규] (2=pos)
    feats["A5_acc_color_change"] = masked_operation(df["A5-1"], df["A5-2"], 3, 'rate') # [신규] (3=color)
    feats["A5_acc_shape_change"] = masked_operation(df["A5-1"], df["A5-2"], 4, 'rate') # [신규] (4=shape)

    # A6, A7 (문제풀이) - 개수만 있으므로 그대로 사용
    feats["A6_correct_count"] = df["A6-1"]
    feats["A7_correct_count"] = df["A7-1"]
    
    # A8, A9 (질문지) - 개수만 있으므로 그대로 사용
    feats["A8-1"] = df["A8-1"]; feats["A8-2"] = df["A8-2"]
    feats["A9-1"] = df["A9-1"]; feats["A9-2"] = df["A9-2"]; feats["A9-3"] = df["A9-3"]; feats["A9-4"] = df["A9-4"]; feats["A9-5"] = df["A9-5"]

    # 원본 CSV의 컬럼 목록 (전처리 후 드롭)
    seq_cols = [
        "A1-1","A1-2","A1-3","A1-4",
        "A2-1","A2-2","A2-3","A2-4",
        "A3-1","A3-2","A3-3","A3-4","A3-5","A3-6","A3-7",
        "A4-1","A4-2","A4-3","A4-4","A4-5",
        "A5-1","A5-2","A5-3",
        "A6-1", "A7-1", "A8-1", "A8-2",
        "A9-1", "A9-2", "A9-3", "A9-4", "A9-5"
    ]
    print("A 검사 1차 전처리 완료")
    return pd.concat([df, feats], axis=1).drop(columns=seq_cols, errors="ignore")

def preprocess_B(df):
    print("Step 1 (B): Age, TestDate 파생...")
    df["Age_num"] = df["Age"].map(convert_age)
    ym = df["TestDate"].map(split_testdate); df["Year"] = [y for y, m in ym]; df["Month"] = [m for y, m in ym]
    feats = pd.DataFrame(index=df.index)
    print("Step 2 (B): B1~B10 features (도메인 수정)...")
    
    # B1, B2 (시야각) [cite: 4, 6, 8, 10, 12, 13]
    feats["B1_task1_acc"] = seq_rate_simple(df["B1-1"])
    feats["B1_rt_mean"] = seq_mean(df["B1-2"]); feats["B1_rt_std"] = seq_std(df["B1-2"])
    feats["B1_change_acc"] = seq_rate_B1_B2(df["B1-3"], ['1', '2'])
    feats["B1_nonchange_acc"] = seq_rate_B1_B2(df["B1-3"], ['3', '4'])

    feats["B2_task1_acc"] = seq_rate_simple(df["B2-1"])
    feats["B2_rt_mean"] = seq_mean(df["B2-2"]); feats["B2_rt_std"] = seq_std(df["B2-2"])
    feats["B2_change_acc"] = seq_rate_B1_B2(df["B2-3"], ['1', '2'])
    feats["B2_nonchange_acc"] = seq_rate_B1_B2(df["B2-3"], ['3', '4'])

    # B3 (신호등) [cite: 25]
    feats["B3_acc_rate"] = seq_rate_simple(df["B3-1"]); feats["B3_rt_mean"] = seq_mean(df["B3-2"]); feats["B3_rt_std"] = seq_std(df["B3-2"])
    
    # B4 (Flanker) [cite: 31, 32]
    feats["B4_congruent_acc"] = seq_rate_B4(df["B4-1"], ['1', '2'])
    feats["B4_incongruent_acc"] = seq_rate_B4(df["B4-1"], ['3', '4', '5', '6'])
    feats["B4_rt_mean"] = seq_mean(df["B4-2"]); feats["B4_rt_std"] = seq_std(df["B4-2"])

    # B5 (공간판단), B6, B7 (시각기억), B8 (주의지속) [cite: 44, 55, 58, 79]
    feats["B5_acc_rate"] = seq_rate_simple(df["B5-1"]); feats["B5_rt_mean"] = seq_mean(df["B5-2"]); feats["B5_rt_std"] = seq_std(df["B5-2"])
    feats["B6_acc_rate"] = seq_rate_simple(df["B6"])
    feats["B7_acc_rate"] = seq_rate_simple(df["B7"])
    feats["B8_acc_rate"] = seq_rate_simple(df["B8"])
    
    # B9, B10 (다중과제) - 개수만 있으므로 그대로 사용
    feats["B9-1"] = df["B9-1"]; feats["B9-2"] = df["B9-2"]; feats["B9-3"] = df["B9-3"]; feats["B9-4"] = df["B9-4"]; feats["B9-5"] = df["B9-5"]
    feats["B10-1"] = df["B10-1"]; feats["B10-2"] = df["B10-2"]; feats["B10-3"] = df["B10-3"]; feats["B10-4"] = df["B10-4"]; feats["B10-5"] = df["B10-5"]; feats["B10-6"] = df["B10-6"]

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


print("\n[INFO] 1차 피처 엔지니어링 (도메인 적용) 시작...")
train_A_features = preprocess_A(train_A_raw)
train_B_features = preprocess_B(train_B_raw)

## 4. 2차 Feature Engineering (신규 도메인 피처 기반)
def _has(df, cols): return all(c in df.columns for c in cols)
def _safe_div(a, b, eps=1e-6): return a / (b + eps)

def add_features_A(df: pd.DataFrame) -> pd.DataFrame:
    feats = df.copy(); eps = 1e-6
    if _has(feats, ["Year","Month"]): feats["YearMonthIndex"] = feats["Year"] * 12 + feats["Month"]
    
    # Speed-Accuracy Tradeoffs
    if _has(feats, ["A1_rt_mean","A1_acc_norm"]): feats["A1_speed_acc_tradeoff"] = _safe_div(feats["A1_rt_mean"], feats["A1_acc_norm"], eps)
    if _has(feats, ["A2_rt_mean","A2_acc_norm"]): feats["A2_speed_acc_tradeoff"] = _safe_div(feats["A2_rt_mean"], feats["A2_acc_norm"], eps)
    if _has(feats, ["A4_rt_mean","A4_acc_congruent"]): feats["A4_speed_acc_tradeoff"] = _safe_div(feats["A4_rt_mean"], feats["A4_acc_congruent"], eps)
    
    # CV
    for k in ["A1","A2","A3","A4", "A5"]: # A5 RT가 없음
        m, s = f"{k}_rt_mean", f"{k}_rt_std"
        if _has(feats, [m, s]): feats[f"{k}_rt_cv"] = _safe_div(feats[s], feats[m], eps)
        
    # A1/A2 Diffs
    if _has(feats, ["A1_rt_fast", "A1_rt_slow"]): feats["A1_rt_speed_cost"] = feats["A1_rt_fast"] - feats["A1_rt_slow"]
    if _has(feats, ["A1_acc_fast", "A1_acc_slow"]): feats["A1_acc_speed_cost"] = feats["A1_acc_fast"] - feats["A1_acc_slow"]
    if _has(feats, ["A2_rt_fast_c1", "A2_rt_slow_c1"]): feats["A2_rt_speed_cost_c1"] = feats["A2_rt_fast_c1"] - feats["A2_rt_slow_c1"]
    if _has(feats, ["A2_acc_fast", "A2_acc_slow"]): feats["A2_acc_speed_cost"] = feats["A2_acc_fast"] - feats["A2_acc_slow"]
    
    # A3 Diffs
    if _has(feats, ["A3_rt_big", "A3_rt_small"]): feats["A3_rt_size_cost"] = feats["A3_rt_big"] - feats["A3_rt_small"]
    if _has(feats, ["A3_valid_acc", "A3_invalid_acc"]): feats["A3_acc_attention_cost"] = feats["A3_valid_acc"] - feats["A3_invalid_acc"]

    # A4 Diffs (Stroop)
    if _has(feats, ["A4_rt_incongruent", "A4_rt_congruent"]): feats["A4_stroop_rt_cost"] = feats["A4_rt_incongruent"] - feats["A4_rt_congruent"]
    if _has(feats, ["A4_acc_congruent", "A4_acc_incongruent"]): feats["A4_stroop_acc_cost"] = feats["A4_acc_congruent"] - feats["A4_acc_incongruent"]

    # A5 Diffs
    if _has(feats, ["A5_acc_nonchange", "A5_acc_pos_change"]): feats["A5_acc_cost_pos"] = feats["A5_acc_nonchange"] - feats["A5_acc_pos_change"]
    if _has(feats, ["A5_acc_nonchange", "A5_acc_color_change"]): feats["A5_acc_cost_color"] = feats["A5_acc_nonchange"] - feats["A5_acc_color_change"]
    
    feats.replace([np.inf, -np.inf], np.nan, inplace=True); return feats

def add_features_B(df: pd.DataFrame) -> pd.DataFrame:
    feats = df.copy(); eps = 1e-6
    if _has(feats, ["Year","Month"]): feats["YearMonthIndex"] = feats["Year"] * 12 + feats["Month"]
    
    # Speed-Accuracy Tradeoffs
    if _has(feats, ["B1_rt_mean", "B1_task1_acc"]): feats["B1_speed_acc_tradeoff"] = _safe_div(feats["B1_rt_mean"], feats["B1_task1_acc"], eps)
    if _has(feats, ["B2_rt_mean", "B2_task1_acc"]): feats["B2_speed_acc_tradeoff"] = _safe_div(feats["B2_rt_mean"], feats["B2_task1_acc"], eps)
    if _has(feats, ["B3_rt_mean", "B3_acc_rate"]): feats["B3_speed_acc_tradeoff"] = _safe_div(feats["B3_rt_mean"], feats["B3_acc_rate"], eps)
    if _has(feats, ["B4_rt_mean", "B4_congruent_acc"]): feats["B4_speed_acc_tradeoff"] = _safe_div(feats["B4_rt_mean"], feats["B4_congruent_acc"], eps)
    if _has(feats, ["B5_rt_mean", "B5_acc_rate"]): feats["B5_speed_acc_tradeoff"] = _safe_div(feats["B5_rt_mean"], feats["B5_acc_rate"], eps)
        
    # CV
    for k in ["B1","B2","B3","B4","B5"]:
        m, s = f"{k}_rt_mean", f"{k}_rt_std"
        if _has(feats, [m, s]): feats[f"{k}_rt_cv"] = _safe_div(feats[s], feats[m], eps)

    # B1/B2 Diffs
    if _has(feats, ["B1_change_acc", "B1_nonchange_acc"]): feats["B1_acc_cost"] = feats["B1_nonchange_acc"] - feats["B1_change_acc"]
    if _has(feats, ["B2_change_acc", "B2_nonchange_acc"]): feats["B2_acc_cost"] = feats["B2_nonchange_acc"] - feats["B2_change_acc"]
        
    # B4 Diffs
    if _has(feats, ["B4_congruent_acc", "B4_incongruent_acc"]): feats["B4_flanker_acc_cost"] = feats["B4_congruent_acc"] - feats["B4_incongruent_acc"]

    feats.replace([np.inf, -np.inf], np.nan, inplace=True); return feats

print("\n[INFO] 2차 피처 엔지니어링 (도메인 적용) 시작...")
print("[INFO] 2차 (A) 시작...")
train_A_features = add_features_A(train_A_features)
print("[INFO] 2차 (A) 완료.")
print("[INFO] 2차 (B) 시작...")
train_B_features = add_features_B(train_B_features)
print("[INFO] 2차 (B) 완료.")

## 5. 최종 데이터 병합 및 저장
print("\n[INFO] 최종 피처 병합 및 Feather 파일로 저장...")

meta_A = train_meta[train_meta["Test"]=="A"].reset_index(drop=True)
meta_B = train_meta[train_meta["Test"]=="B"].reset_index(drop=True)

meta_A_features = meta_A.merge(train_A_features, on='Test_id', how='left')
meta_B_features = meta_B.merge(train_B_features, on='Test_id', how='left')

print("[INFO] A, B 데이터프레임 병합 중...")
all_train_df = pd.concat([meta_A_features, meta_B_features], sort=False).reset_index(drop=True)
print("[INFO] 병합 완료.")

FEATURE_SAVE_PATH = os.path.join(BASE_DIR, "all_train_data.feather")
print(f"[INFO] Feather 파일 저장 중... ({FEATURE_SAVE_PATH})")
all_train_df.to_feather(FEATURE_SAVE_PATH)
print("[INFO] Feather 파일 저장 완료.")

print("\n--- [ 1_Preprocess.ipynb ] 작업 완료 ---")
print(f"다음 파일이 생성되었습니다:")
print(f"1. {FEATURE_SAVE_PATH} (모든 피처가 포함된 학습 데이터)")
print("이제 2_Train_Models.ipynb 노트북을 실행하세요.")
