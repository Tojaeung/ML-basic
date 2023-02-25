# %% 라이브러리, 모듈 호출
from kiwisolver import Solver
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.linear_model import LogisticRegression

# %% 데이터셋 불러오기, 결과값 확인
"""
pregnancies: 임신 횟수
Glucose: 포도당 수치
BloodPressure: 혈압
SkinThickness: 팔 삼두근 뒤쪽의 피하지방 측정값
InSulin: 인슐린
BMI: 체질량
DiabetesPedigreeFunction: 당뇨 내력 가중치 값
Age: 나이
Outecome: 결정값 (0, 1)
"""

diabetes_data = pd.read_csv("C:\\Users\\tojaeung\\datasets\\diabetes.csv")
print(diabetes_data["Outcome"].value_counts())
diabetes_data.head()
diabetes_data.info()
diabetes_data.shape


# %% 편의 함수 선언
def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)

    print("오차행렬:\n", confusion)
    print(
        "정확도: {0:.4f},정밀도: {1:.4f},재현율: {2:.4f},F1: {3:.4f},AUC: {4:.4f}".format(
            accuracy, precision, recall, f1, roc_auc
        )
    )


def precision_recall_curve_plot(y_test=None, pred_proba_c1=None):
    # threshold ndarray와 이 threshold에 따른 정밀도, 재현율 추출
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)

    # X축을 threshold값, Y축을 정밀도, 재현율로 설정 (정밀도는 점선으로 표시)
    plt.figure(figsize=(8, 6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(
        thresholds, precisions[0:threshold_boundary], linestyle="--", label="precision"
    )
    plt.plot(thresholds, recalls[0:threshold_boundary], label="recall")

    # threshold값 X축의 scale을 0.1단위로 변경
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))

    # x축, y축 label과 legend 그리고 grid 설정
    plt.xlabel("Threshold value")
    plt.ylabel("Precision and Recall value")
    plt.legend()  # 우측상단에 그래프 라벨 표시
    plt.grid()  # 그래프 격자선 표시
    plt.show()


def get_eval_by_threshold(y_test, pred_proba_c1, thresholds):
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1)
        pred = binarizer.transform(pred_proba_c1)
        print("임계값:", custom_threshold)
        get_clf_eval(y_test, pred, pred_proba_c1)


# %% 로지스틱 회귀를 이용해 학습 및 예측 수행
X = diabetes_data.iloc[:, :-1]  # 마지막 칼럼 결과값이 outcome 분리
y = diabetes_data.iloc[:, -1]  # 마지막 칼럼 제외 나머지 데이터셋 분리

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=156, stratify=y
)

# 로지스틱 회귀로 학습, 예측 및 평가 수행
lr_clf = LogisticRegression(solver="liblinear")
lr_clf.fit(X_train, y_train)

pred = lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:, 1]  # 0, 1일떄 확률을 구하는데, 1일때 확률을 분리해서 구한다.

get_clf_eval(y_test, pred, pred_proba)

# %% precision recall 곡선
pred_proba_c1 = lr_clf.predict_proba(X_test)[:, 1]
precision_recall_curve_plot(y_test, pred_proba_c1)

# %% 각 피쳐들의 4분위 분포 확인
diabetes_data.describe()
"""
데이터 분포도를 보고 이상함을 감지해야한다.
min 최소값을 확인하면, 포도당, 혈압등 수치가 0인 데이터가 있다.
죽은 사람인가?? 데이터가 이상하기 떄문에 이상한 데이터는 정제해준다.
"""

# %% 포도당 피처의 분포도
plt.hist(diabetes_data["Glucose"], bins=100)  # bins는 몇개로 쪼갤것인가
plt.show()

# %% 0값이 있는 피처들에서 0값의 데이터 건수와 퍼센트 계산
zero_features = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]  # 0 데이터가 있는 칼럼들

total_count = diabetes_data["Glucose"].count()  # 전체 데이터 건수

for feature in zero_features:
    zero_count = diabetes_data[diabetes_data[feature] == 0][feature].count()
    print(
        "{0}의 건수: {1}, 비율: {2:.2f}%".format(
            feature, zero_count, 100 * (zero_count / total_count)
        )
    )

# %% 0값을 평균값으로 대체
diabetes_data[zero_features] = diabetes_data[zero_features].replace(
    0, diabetes_data[zero_features].mean()  # type: ignore
)
# %% 스탠다드 스케일러 적용 (로지스틱 회귀는 스탠다드 스케일러가 좋음)
X = diabetes_data.iloc[:, :-1]
y = diabetes_data.iloc[:, -1]

# 스탠다드 클래스를 이용해 피처 데이터셋에 일괄적으로 스케일링 적용
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=156, stratify=y
)

lr_clf = LogisticRegression(solver="liblinear")
lr_clf.fit(X_train, y_train)
pred = lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:, 1]

get_clf_eval(y_test, pred, pred_proba)

# %% 분류결정 임계값을 변경하면서 성능 측정

# 앞에 지표에서 재현율이 낮아... 재현율 올리기 위해 임계값 0.5 아래에서 판별
thresholds = [0.3, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.5]

pred_proba = lr_clf.predict_proba(X_test)
get_eval_by_threshold(y_test, pred_proba[:, 1].reshape(-1, 1), thresholds)

# %% 임계값 0.48로 테스트
binarizer = Binarizer(threshold=0.48)
pred_th_048 = binarizer.fit_transform(pred_proba[:, 1].reshape(-1, 1))
get_clf_eval(y_test, pred_th_048, pred_proba[:, 1])
