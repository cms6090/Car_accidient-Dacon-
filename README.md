# Base 모델 정리 (LB 0.1536969035)

> 운전 적성 검사 예측 문제에서 현재 기준이 되는 **Base 모델** 정리입니다.  
> 이 문서는 팀원들이 전체 파이프라인과 모델 구조를 한 번에 이해하고,  
> 앞으로의 개선 방향(Feature / Model / Hyperparameter)을 논의하기 위한 자료입니다.

---

## 1. 전체 파이프라인 개요

파이프라인은 크게 두 단계로 나뉩니다.

1.  **전처리 노트북**: `1_Preprocess.ipynb`
    * PDF 명세 + 도메인 지식 기반 Feature Engineering
    * A/B 검사 raw CSV → 도메인 피처 / 요약 인덱스 생성
    * `all_train_data.feather` 저장

2.  **모델 학습 노트북**: `2_Train_Models.ipynb`
    * `all_train_data.feather` 로드
    * Hydra + hydra-optuna-sweeper를 이용한 하이퍼 파라미터 자동 탐색, 저장
    * StratifiedKFold(5-fold) 기반 A/B 분리 학습
    * CatBoost + Isotonic Regression (fold별 calibration)
    * PK Stats(B 전용 group-level 피처) 생성 및 사용
    * 최종 모델/보정기/PK Stats 저장 → `submit.zip` 구성

---

## 2. 입력 데이터 설명

### 2.1 메타 데이터

-   파일: `./data/train.csv`
-   주요 컬럼:
    -   `Test_id` : 검사 단위 ID (A/B 한 번 시행이 하나의 row)
    -   `PrimaryKey` : 사람 ID (같은 사람이 여러 번 검사)
    -   `Test` : `'A'` (신규 자격) / `'B'` (자격 유지)
    -   `Label` : 타깃 (0/1)

### 2.2 A/B 원본 데이터

-   A 검사: `./data/train/A.csv`
-   B 검사: `./data/train/B.csv`
-   특징:
    -   `"1,2,1,3,..."` 형식의 **trial-level 시퀀스**가 많음
    -   예시 (A 쪽):
        -   `A1-1, A1-2, A1-3, A1-4` (조건/정답/RT 등)
        -   …
        -   `A9-1` ~ `A9-5`
    -   예시 (B 쪽):
        -   `B1-1` ~ `B1-3`
        -   …
        -   `B10-1` ~ `B10-6`

전처리에서 이 시퀀스들을 **평균, 표준편차, 조건별 accuracy, cost 지표**로 요약해서 사용합니다.

---

## 3. 1_Preprocess.ipynb: Feature Engineering

### 3.1 공통 유틸

```python
def convert_age(val):
    # "25a" → 25, "25b" → 30
    ...

def split_testdate(val):
    # 202401 → (2024, 1)
    ...

def seq_mean(series):
    # "1,2,3" → np.mean
    ...

def seq_std(series):
    # "1,2,3" → np.std
    ...

def masked_operation(cond_series, val_series, target_conds, operation='mean'/'std'/'rate'):
    # 조건(cond)이 특정 값일 때 val의 mean / std / correct rate 계산
    ...
```

### 3.2 A 검사: 1차 도메인 피처 (preprocess_A)

1.  **기본 파생**
    * `Age` → `Age_num`
    * `TestDate` → `Year`, `Month`

2.  **A1 (속도 예측)**
    * RT/정확도 요약:
    * 전체 RT:
        * `A1_rt_mean`, `A1_rt_std`
    * 방향/속도 조건별 RT:
        * `A1_rt_left`, `A1_rt_right`
        * `A1_rt_slow`, `A1_rt_norm`, `A1_rt_fast`
    * 속도 조건별 accuracy:
        * `A1_acc_slow`, `A1_acc_norm`, `A1_acc_fast`

3.  **A2 (정지 예측)**
    * RT:
        * `A2_rt_mean`, `A2_rt_std`
        * `A2_rt_slow_c1`, `A2_rt_norm_c1`, `A2_rt_fast_c1` (속도 조건)
        * `A2_rt_slow_c2`, `A2_rt_norm_c2`, `A2_rt_fast_c2` (가속도 조건)
    * accuracy:
        * `A2_acc_slow`, `A2_acc_norm`, `A2_acc_fast`

4.  **A3 (주의 전환)**
    * valid/invalid accuracy:
        * `A3_valid_acc`, `A3_invalid_acc`
    * RT:
        * `A3_rt_mean`, `A3_rt_std`
        * `A3_rt_small`, `A3_rt_big`
        * `A3_rt_left`, `A3_rt_right`

5.  **A4 (Stroop)**
    * RT:
        * `A4_rt_mean`, `A4_rt_std`
        * `A4_rt_congruent`, `A4_rt_incongruent`
    * accuracy:
        * `A4_acc_congruent`, `A4_acc_incongruent`

6.  **A5 (변화 탐지)**
    * 조건별 accuracy:
        * `A5_acc_nonchange`
        * `A5_acc_pos_change`
        * `A5_acc_color_change`
        * `A5_acc_shape_change`

7.  **A6, A7 (문제풀이)**
    * `A6_correct_count`, `A7_correct_count`

8.  **A8, A9 (질문지)**
    * `A8-1`, `A8-2`
    * `A9-1` ~ `A9-5`

원본 시퀀스 컬럼(`A1-1` ~ `A9-5`)은 요약 피처 생성 후 **drop** 합니다.

### 3.3 B 검사: 1차 도메인 피처 (preprocess_B)

1.  **기본 파생**
    * 동일하게 `Age_num`, `Year`, `Month`

2.  **B1, B2 (시야각)**
    * task accuracy:
        * `B1_task1_acc`, `B2_task1_acc`
    * RT:
        * `B1_rt_mean`, `B1_rt_std`
        * `B2_rt_mean`, `B2_rt_std`
    * change vs non-change accuracy:
        * `B1_change_acc`, `B1_nonchange_acc`
        * `B2_change_acc`, `B2_nonchange_acc`

3.  **B3 (신호등)**
    * `B3_acc_rate`, `B3_rt_mean`, `B3_rt_std`

4.  **B4 (Flanker)**
    * `B4_congruent_acc`, `B4_incongruent_acc`
    * `B4_rt_mean`, `B4_rt_std`

5.  **B5, B6, B7, B8 (표지판/도로찾기/추적/주의지속)**
    * `B5_acc_rate`, `B5_rt_mean`, `B5_rt_std`
    * `B6_acc_rate`, `B7_acc_rate`, `B8_acc_rate`

6.  **B9, B10 (점수형)**
    * `B9-1` ~ `B9-5`
    * `B10-1` ~ `B10-6`

원본 시퀀스 컬럼(`B1-1` ~ `B10-6`)은 **drop**.

### 3.4 A/B 2차 피처 (add_features_A, add_features_B)

**A 쪽 2차 피처**

* 검사 시점 인덱스:
    * `YearMonthIndex = Year * 12 + Month`
* Speed–Accuracy Tradeoff:
    * `A1_speed_acc_tradeoff = A1_rt_mean / A1_acc_norm`
    * `A2_speed_acc_tradeoff = A2_rt_mean / A2_acc_norm`
    * `A4_speed_acc_tradeoff = A4_rt_mean / A4_acc_congruent`
* RT 변동성 (CV):
    * `A1_rt_cv`, `A2_rt_cv`, `A3_rt_cv`, `A4_rt_cv`
* 속도/조건 간 cost:
    * `A1_rt_speed_cost = A1_rt_fast - A1_rt_slow`
    * `A1_acc_speed_cost = A1_acc_fast - A1_acc_slow`
    * `A2_rt_speed_cost_c1`, `A2_acc_speed_cost`
    * `A3_rt_size_cost = A3_rt_big - A3_rt_small`
    * `A3_acc_attention_cost = A3_valid_acc - A3_invalid_acc`
    * `A4_stroop_rt_cost = A4_rt_incongruent - A4_rt_congruent`
    * `A4_stroop_acc_cost = A4_acc_congruent - A4_acc_incongruent`
    * `A5_acc_cost_pos`, `A5_acc_cost_color`, `A5_acc_cost_shape`
* 작업기억/인지 요약:
    * `A_working_memory_index`
    * `A_cog_sum`, `A_cog_mean`
* 종합 위험 점수:
    * `RiskScore`
    * (Stroop cost, attention cost, RT CV, 변화탐지 cost 등을 가중합)

**B 쪽 2차 피처**

* 검사 시점 인덱스:
    * `YearMonthIndex`
* Speed–Accuracy Tradeoff:
    * `B1_speed_acc_tradeoff`, `B2_speed_acc_tradeoff`,
        `B3_speed_acc_tradeoff`, `B4_speed_acc_tradeoff`, `B5_speed_acc_tradeoff`
* RT 변동성 (CV):
    * `B1_rt_cv` ~ `B5_rt_cv`
* change vs non-change / Flanker cost:
    * `B1_acc_cost = B1_nonchange_acc - B1_change_acc`
    * `B2_acc_cost = B2_nonchange_acc - B2_change_acc`
    * `B4_flanker_acc_cost = B4_congruent_acc - B4_incongruent_acc`
* 시지각/운동 일관성:
    * `B_visuomotor_variability` (RT CV 평균)
* 순수 반응/신호탐지 능력:
    * `B_reaction_overall` (B3, B5, B6, B7, B8 accuracy 평균)
* 집행기능/억제 컨트롤:
    * `B_executive_control_index`
* B9/B10/인지 요약:
    * `B9_sum`, `B9_mean`
    * `B10_sum`, `B10_mean`
    * `B_cog_sum`, `B_cog_mean`
* 종합 위험 점수:
    * `RiskScore_B`

최종적으로 A/B 모두 메타(`train_meta`)와 merge 후  
`./data/all_train_data.feather`로 저장합니다.

---

## 4. 2_Train_Models.ipynb: 모델 학습

### 4.1 데이터 로드

```python
FEATURE_SAVE_PATH = "./data/all_train_data.feather"
all_train_df = pd.read_feather(FEATURE_SAVE_PATH)
```

`all_train_df`는 A/B 통합된 학습 테이블

`Label`이 없는 경우 0으로 채움:

```python
all_train_df['Label'] = all_train_df['Label'].fillna(0)
```

---

## 5. 평가 지표 & Combined Score

### 5.1 ECE (Expected Calibration Error)

```python
def expected_calibration_error(y_true, y_prob, n_bins=10):
    if len(y_true) == 0 or len(y_prob) == 0:
        return 0.0
    y_prob = np.nan_to_num(y_prob, nan=0.0)

    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_edges[0] = -0.001
    bin_edges[-1] = 1.001

    df['y_prob'] = np.clip(df['y_prob'], 0, 1)
    df['bin'] = pd.cut(df['y_prob'], bins=bin_edges, right=True)

    bin_stats = df.groupby('bin', observed=True).agg(
        bin_total=('y_prob', 'count'),
        prob_true=('y_true', 'mean'),
        prob_pred=('y_prob', 'mean'),
    )

    non_empty_bins = bin_stats[bin_stats['bin_total'] > 0]
    if len(non_empty_bins) == 0:
        return 0.0

    bin_weights = non_empty_bins['bin_total'] / len(y_prob)
    prob_true = non_empty_bins['prob_true']
    prob_pred = non_empty_bins['prob_pred']

    ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))
    return ece
```

### 5.2 Combined Score

```python
def combined_score(y_true, y_prob):
    if len(y_true) == 0 or len(y_prob) == 0 or \
       np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
        print("  AUC: N/A (단일 클래스), Brier: N/A, ECE: N/A (No data)")
        return 1.0

    y_prob = np.nan_to_num(y_prob, nan=0.0)

    mean_auc = roc_auc_score(y_true, y_prob)
    mean_brier = mean_squared_error(y_true, y_prob)
    mean_ece = expected_calibration_error(y_true, y_prob)

    score = 0.5 * (1 - mean_auc) + 0.25 * mean_brier + 0.25 * mean_ece

    print(f"  AUC: {mean_auc:.4f}, Brier: {mean_brier:.4f}, ECE: {mean_ece:.4f}")
    print(f"  Combined Score: {score:.5f}")
    return score
```

* 목적: **Combined Score를 최소화** (낮을수록 좋음)
* `AUC`는 높을수록 좋으므로 `(1 - AUC)`로 뒤집어서 사용
* `Brier`, `ECE`는 원래 낮을수록 좋은 값
* 리더보드 스코어(대회 평가)는 이 Combined Score와 연동되어 있다고 가정하고,
    A-only / B-only / 전체에 대해 각각 관찰합니다.

---

## 6. K-Fold 구성

```python
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

train_indices_list = []
val_indices_list = []

for train_idx, val_idx in skf.split(all_train_df, all_train_df['Label']):
    train_indices_list.append(train_idx)
    val_indices_list.append(val_idx)
```

* 전체 row 단위로 `StratifiedKFold(5-fold)` 적용
* `Label` 비율 균형 유지
* 현재 Base 모델은 `PrimaryKey` 기준 `GroupKFold`는 **사용하지 않음**
* 즉, 같은 사람이 train/val에 동시에 들어갈 수 있는 구조
* 이 때문에 검증 점수가 다소 **optimistic**할 수 있음

---

## 7. 피처 세팅

```python
CAT_FEATURES = ['Age', 'PrimaryKey']
DROP_COLS_TRAIN = [
    'Test_id', 'Test_x', 'Test_y', 'Label',
    'TestDate', 'Year', 'Month', 'base_index',
]
```

* **범주형 피처:**
    * `Age` : 문자열로 캐스팅해서 CatBoost category로 사용
    * `PrimaryKey` : 사람 ID, 개인 특성을 캡처하는 카테고리 피처
* **드롭하는 컬럼** (학습에서 제외):
    * ID/메타: `Test_id`, `Test_x`, `Test_y`, `TestDate`, `Year`, `Month`, `base_index`
    * 타깃: `Label`

---

## 8. PK Stats (PrimaryKey level 집계 피처)

Fold 루프 안에서, 각 Fold의 **Train 데이터만** 사용하여 사람 단위 통계를 만듭니다.

```python
agg_funcs = {
    'Age_num': ['mean', 'min', 'max'],
    'YearMonthIndex': ['mean', 'std', 'min', 'max'],
    'A1_rt_mean': ['mean', 'std'],
    'A4_acc_congruent': ['mean', 'std'],
    'A4_acc_incongruent': ['mean', 'std'],
    'A4_stroop_rt_cost': ['mean', 'std'],
    'RiskScore': ['mean', 'std', 'max'],
    'B1_change_acc': ['mean', 'std'],
    'B1_nonchange_acc': ['mean', 'std'],
    'B3_rt_mean': ['mean', 'std'],
    'B4_flanker_acc_cost': ['mean', 'std'],
    'B4_rt_mean': ['mean', 'std'],
    'RiskScore_B': ['mean', 'std', 'max'],
    'Test_id': ['count'],
}

pk_stats_fold = train_df_fold.groupby('PrimaryKey').agg(valid_agg_funcs)
pk_stats_fold.columns = ['_'.join(col).strip() for col in pk_stats_fold.columns.values]
pk_stats_fold.rename(columns={'Test_id_count': 'pk_test_total_count'}, inplace=True)
```

* `PrimaryKey` 단위로:
    * 나이/검사 시점 분포
    * A/B 핵심 도메인 피처의 mean/std/max
    * `RiskScore`/`RiskScore_B` 요약
    * `pk_test_total_count`: 검사 횟수

검사 타입별 횟수도 추가:

```python
pk_test_type_count_fold = train_df_fold.groupby('PrimaryKey')['Test_x'] \
    .value_counts().unstack(fill_value=0)

if 'A' not in pk_test_type_count_fold.columns:
    pk_test_type_count_fold['A'] = 0
if 'B' not in pk_test_type_count_fold.columns:
    pk_test_type_count_fold['B'] = 0

pk_test_type_count_fold = pk_test_type_count_fold[['A', 'B']]
pk_test_type_count_fold.columns = ['pk_test_A_count', 'pk_test_B_count']

pk_stats_fold = pk_stats_fold.join(pk_test_type_count_fold, how='left').reset_index()
```

Fold별로 생성된 `pk_stats_fold`를 리스트에 저장

모든 Fold 종료 후:

```python
all_pk_stats_df = pd.concat(all_pk_stats_folds)
final_pk_stats = all_pk_stats_df.groupby('PrimaryKey').mean().reset_index()
final_pk_stats.to_csv("./model/pk_stats_final.csv", index=False)
```

→ 추론 시 test 데이터와 merge해 사용

---

## 9. 모델 구조 및 하이퍼파라미터

### 9.1 공통 CatBoost 설정 (A/B 동일)

```python
cat_base_model = cb.CatBoostClassifier(
    iterations=3000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    thread_count=-1,
    early_stopping_rounds=100,
    verbose=1000,
)
```

* `iterations=3000` + `early_stopping_rounds=100`
* `learning_rate=0.05`
* `depth=6`
* `eval_metric='AUC'` (validation은 AUC 기준)
* **카테고리:**
    * `Age`, `PrimaryKey` → string으로 변환 후 CatBoost에 인덱스로 전달

**카테고리 변환:**

```python
for col in CAT_FEATURES:
    cb_X_train[col] = cb_X_train[col].fillna('nan').astype(str)
    cb_X_val[col] = cb_X_val[col].fillna('nan').astype(str)

cat_features_indices = [cb_X_train.columns.get_loc(c) for c in CAT_FEATURES if c in cb_X_train]
```

### 9.2 A 모델 (신규 자격)

Fold마다:

```python
X_train_A = train_df_fold[train_df_fold['Test_x'] == 'A'].copy()
y_train_A = X_train_A['Label'].values

X_val_A = val_df_fold[val_df_fold['Test_x'] == 'A'].copy()
y_val_A = X_val_A['Label'].values
```

**피처 선택:**

```python
base_cols = [col for col in X_train.columns if not col.startswith('pk_')]
numeric_cols = list(set(base_cols) - set(CAT_FEATURES) - set(DROP_COLS_TRAIN))

cb_X_train = X_train[numeric_cols + CAT_FEATURES]
cb_X_val = X_val[numeric_cols + CAT_FEATURES]
```

* A 모델은 **PK Stats(`pk_`)를 사용하지 않음**
* 나머지 모든 수치형 도메인 피처 + 카테고리(`Age`, `PrimaryKey`) 사용

**학습 & 확률 보정:**

```python
cat_A = CatBoostClassifier(...).fit(cb_X_train, y_train_A, eval_set=[(cb_X_val, y_val_A)], ...)

pred_cat_uncal = cat_A.predict_proba(cb_X_val)[:, 1]
combined_score(y_val_A, pred_cat_uncal)  # 비보정 점수

calibrator_A = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
calibrator_A.fit(pred_cat_uncal, y_val_A)

pred_cat_calibrated = calibrator_A.predict(pred_cat_uncal)
combined_score(y_val_A, pred_cat_calibrated)  # 보정 후 점수
```

**fold별 산출물:**

* `./model/catboost_A_fold{0~4}.pkl`
* `./model/calibrator_A_fold{0~4}.pkl`

### 9.3 B 모델 (자격 유지)

Fold마다:

```python
X_train_B = train_df_fold[train_df_fold['Test_x'] == 'B'].copy()
y_train_B = X_train_B['Label'].values

X_val_B = val_df_fold[val_df_fold['Test_x'] == 'B'].copy()
y_val_B = X_val_B['Label'].values
```

**PK Stats 병합:**

```python
X_train_B = X_train_B.merge(pk_stats_fold, on='PrimaryKey', how='left')
X_val_B = X_val_B.merge(pk_stats_fold, on='PrimaryKey', how='left')
```

**피처 선택:**

```python
numeric_cols = list(set(X_train_B.columns) - set(CAT_FEATURES) - set(DROP_COLS_TRAIN))
common_numeric_cols = list(set(X_train_B[numeric_cols].columns) & set(X_val_B[numeric_cols].columns))

cb_X_train = X_train_B[common_numeric_cols + CAT_FEATURES]
cb_X_val = X_val_B[common_numeric_cols + CAT_FEATURES]
```

* B 모델은 **PK Stats(`pk_`) 포함**해서 수치형 피처 전체 사용
* train/val 모두에 있는 **공통 컬럼**만 선택

**학습 & 확률 보정:**

```python
cat_B = CatBoostClassifier(...).fit(cb_X_train, y_train_B, eval_set=[(cb_X_val, y_val_B)], ...)

pred_cat_uncal = cat_B.predict_proba(cb_X_val)[:, 1]
combined_score(y_val_B, pred_cat_uncal)

calibrator_B = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
calibrator_B.fit(pred_cat_uncal, y_val_B)

pred_cat_calibrated = calibrator_B.predict(pred_cat_uncal)
combined_score(y_val_B, pred_cat_calibrated)
```

**fold별 산출물:**

* `./model/catboost_B_fold{0~4}.pkl`
* `./model/calibrator_B_fold{0~4}.pkl`

**B 데이터가 너무 적거나 단일 클래스일 경우:**

* 해당 Fold B 모델은 `None`으로 저장

---

## 10. 최종 산출물 (submit.zip 구성 요소)

* **모델:**
    * `catboost_A_fold0.pkl` ~ `catboost_A_fold4.pkl`
    * `catboost_B_fold0.pkl` ~ `catboost_B_fold4.pkl`
* **캘리브레이터:**
    * `calibrator_A_fold0.pkl` ~ `calibrator_A_fold4.pkl`
    * `calibrator_B_fold0.pkl` ~ `calibrator_B_fold4.pkl`
* **PK Stats:**
    * `pk_stats_final.csv`

이 11개 파일이 현재 LB 0.1536969035 (Base 모델) 제출 구성입니다.

---

## 11. 한 줄 요약

Base 모델은 A/B 각각에 대해 도메인 기반 RT/정확도/Cost/인덱스 피처를 구성하고,  
B 쪽에는 `PrimaryKey` 단위 PK Stats까지 얹은 뒤,  
`StratifiedKFold(5-fold)`로 A/B 별 CatBoost(3000 iters, depth 6, lr 0.05)를 학습하고  
fold별 `Isotonic Regression`으로 확률 보정을 수행하여  
Combined Score(AUC + Brier + ECE)를 최소화하는 구조입니다.

---

## 12. 앞으로의 계획 (요구사항)

Base 모델을 기준으로, 이후 실험들은 아래 목표를 갖습니다.

* **검증 단계에서 항상 세 가지 점수 트래킹**
    * A-only Combined Score
    * B-only Combined Score
    * 전체(A+B) Combined Score

* **개선 방향**
    * **Feature 변경:**
        * A/B 도메인 피처 추가/수정 (A/B 영상 텍스트 변환 후 피터 생성 시도해보았음)
        * cross-task 피처, longitudinal(시점) 피처 
    * **Model 구조 변경:**
        * `GroupKFold` / `StratifiedGroupKFold` 도입 여부
        * 다른 모델 구조 (LightGBM/ensemble) 실험
    * **Hyperparameter 튜닝:**
        * `depth`, `learning_rate`, `L2`, `iterations`, `class_weight` 등

* **규칙**
    * 어떤 변경이든 Base 대비 A/B/전체 Combined Score를 **낮추는지를 기준**으로 검증
    * 리더보드 점수와의 괴리도 같이 체크해서 overfitting/underfitting 판단
