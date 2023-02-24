"""
재현율이 중요한 경우, 실제 positive를 negative로 잘못예측 (암 진단)
정밀도가 중요한 경우, 실제 negative를 positive로 잘못예측 (스팸메일)

업무 특성상 정밀도, 재현율을 조정해야할 경우 분류 결정 임계값(Threshold)를 사용하면 된다.

정밀도 = TP / (FP + TP)
재현율 = TP / (FN + TP)

분류 결정 임계값이 낮아질수록 positive 예측 확률이 높아진다. (재현율 증가)
"""
# %% 라이브러리 호출
from concurrent.futures import thread
from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.linear_model import LogisticRegression


# %% 모델 가공을 위한 함수 선언
# Null값 처리
def fillna(df: pd.DataFrame) -> pd.DataFrame:
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    df["Cabin"].fillna("N", inplace=True)
    df["Embarked"].fillna("N", inplace=True)
    return df


# 불필요한 칼럼 제거
def drop_featrues(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)
    return df


# 문자열 칼럼 숫자롤 바꿔주기(인코딩)
def format_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Cabin"] = df["Cabin"].str[:1]
    features = ["Cabin", "Sex", "Embarked"]

    for feature in features:
        label = LabelEncoder()
        label = label.fit(df[feature])
        df[feature] = label.transform(df[feature])

    return df


# 위의 3함수 묶어서 실행
def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    df = fillna(df)
    df = drop_featrues(df)
    df = format_features(df)
    return df


# %% 오차행렬, 정밀도, 재현율을 모두 확인할 수 있는 함수 선언
def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    print("오차행렬:\n", confusion)
    print("정확도: {0:.4f}, 정밀도:{1:.4f}, 재현율:{2:.4f}".format(accuracy, precision, recall))


# %% 로지스틱 회귀 분류를 이용한 정밀도, 재현율 분석
lr_clf = LogisticRegression(solver="liblinear")

# 타이타닉 데이터셋 이용
titanic_df = pd.read_csv("../data/titanic_train.csv")

y_titanic_df = titanic_df["Survived"]
X_titanic_df = titanic_df.drop("Survived", axis=1)
X_titanic_df = transform_features(X_titanic_df)
X_train, X_test, y_train, y_test = train_test_split(
    X_titanic_df, y_titanic_df, test_size=0.2, random_state=11
)

lr_clf.fit(X_train, y_train)
pred = lr_clf.predict(X_test)
get_clf_eval(y_test, pred)

# %% predict_proba 메서드 확인
# predict_proba은 negative, positive 확률을 각각 나타낸다.
# 첫번째 칼럼은 negative 확률, 두번째 칼럼은 positive 칼럼을 나타낸다.
pred_proba = lr_clf.predict_proba(X_test)
print("predict_proba 메서드 확인:\n", pred_proba[:3])

# concatenate 메서드로 pred_proba와 그 결과값(0 또는 1)을 묶는다.
pred_proba_result = np.concatenate([pred_proba, pred.reshape(-1, 1)], axis=1)
print("두개의 클래스 중에서 더 큰 확률을 클래스 값으로 예측:\n", pred_proba_result[:3])

# %% Threshold 조정을 위한 Binarizer 활용
from sklearn.preprocessing import Binarizer

X = [[1, -1, 2], [2, 0, 0], [0, 1.1, 1.2]]

# threshold 기준값보다 같거나 작으면 0, 크면 1을 반환한다.
binarizer = Binarizer(threshold=1.1)
print(binarizer.fit_transform(X))

# %% 여러개의 분류 결정 임계값을 변경해서 정밀도, 재현율의 변화를 확인해보자
thresholds = [0.4, 0.45, 0.5, 0.55, 0.6]


def get_eval_by_threshold(y_test, pred_proba_c1, thresholds):
    for custom_threshold in thresholds:
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1)
        custom_pred = binarizer.transform(pred_proba_c1)
        print("임겟값:", custom_threshold)
        get_clf_eval(y_test, custom_pred)


get_eval_by_threshold(y_test, pred_proba[:, 1].reshape(-1, 1), thresholds)
"""
임계값이 낮아질수록 재현율이 높아진다.
임계값이 높아질수록 정밀도가 높아진다.
업무 특성을 고려하여 설정하면 된다.
"""
# %% precision_recall_curve(): 임곗값에 따른 정밀도 재현율 값 추출
from sklearn.metrics import precision_recall_curve

# 첫번째 칼럼 추출 (negative 칼럼)
pred_proba_c1 = lr_clf.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)

# 15씩 건너 뛰어서 thresholds 추출
thr_index = np.arange(0, thresholds.shape[0], 15)

print("샘플 임계값별 정밀도:", np.round(precisions[thr_index], 3))
print("샘플 임계값별 재현율:", np.round(recalls[thr_index], 3))
