# %% 라이브러리 호출
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits


# %% 정확도의 모순을 확인하기 위한 커스텀 분류 클래스 생성
class MyDummyClassifier(BaseEstimator):
    # 아무것도 학습하지 않음
    def fit(self, X, y=None):
        pass

    # 여자면 생존, 남자면 죽음
    def predict(self, X):
        pred = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            if X["Sex"].iloc[i] == 1:
                pred[i] = 0
            else:
                pred[i] = 1

        return pred


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


# %% train_test_split를 이용한 정확도 출력
titanic_df = pd.read_csv("../data/titanic_train.csv")

y_titanic_df = titanic_df["Survived"]
X_titanic_df = titanic_df.drop("Survived", axis=1)
X_titanic_df = transform_features(X_titanic_df)
X_train, X_test, y_train, y_test = train_test_split(
    X_titanic_df, y_titanic_df, test_size=0.2, random_state=0
)

myclf = MyDummyClassifier()
myclf.fit(X_train, y_train)

pred = myclf.predict(X_test)
print("더미클래스의 정확도: {0:.4f}".format(accuracy_score(y_test, pred)))

"""
단순히 여자면 생존, 남자는 죽음으로 알고리즘을 해도
정확도가 80프로에 육박한다.
불균형한 레이블, 이진분류 상황시, 정확도의 지표는 왜곡을 가져올 수 있다..
"""

# 정확도 모순 확인을 위한 MNIST 예제


# %% 정확도 무순 확인하기 위한 커스텀 분류 클래스 생성
class MyFakeClassifier(BaseEstimator):
    def fit(self, X, y):
        pass

    # 모든 예측을 0으로 한다. (찾으려는 7이 존재하지 않는다.)
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


# %% 사이킷런 내장 데이터셋 MNIST 로딩
digits = load_digits()

print(digits.data)  # type: ignore
print(digits.data.shape)  # type: ignore
print(digits.target)  # type: ignore

# %% 7번이면 true, 아니면 false
y = (digits.target == 7).astype(int)  # type: ignore
X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=11)  # type: ignore

# %% 불균형한 레이블 분포도 확인
print(y_test.shape)  # type: ignore
print("테스트셋 0,1의 분포도", pd.Series(y_test).value_counts())

# %% 커스텀 더미 클래스로 정확도 확인
fakeClf = MyFakeClassifier()
fakeClf.fit(X_train, y_train)
pred = fakeClf.predict(X_test)
print("더미클래스의 정확도: {0:.4f}".format(accuracy_score(y_test, pred)))
"""
모든 예측을 0으로 해도 정확도가 90프로에 육박한다.
불균형한 데이터셋, 이진분류인 경우에는 정확도를 사용하지 않는다.
앞으로 이진분류에서 사용되는 다른 평가지표를 공부하자 !!
"""

# %% 오차행렬
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, pred))
"""
[[ TN(405),  FP(0) ],
[ FN(45),   TP(0) ]]

0으로 학습했기떄문에(Negative) positive 지표가 수행되지 않아 0으로 표시되었다.
이 오차행렬을 보고 모델의 이상징후를 감지해야한다.
"""

# %% 오차행렬을 수치적으로 표현할 수 있는 정밀도, 재현율
from sklearn.metrics import precision_score, recall_score

print("정밀도:", precision_score(y_test, pred))  # 0
print("재현율:", recall_score(y_test, pred))  # 0
"""
정밀도 = TP / FP + TP (TP는 높이고 FP 낮춰야한다.)
재현율 = TP / FN + TP (TP는 높이고 FN 낮춰야한다.)

0으로만 학습해서 positive 지표가 없음
그래서, 정밀도, 재현율이 0이다.

여기서 정확도의 모순을 잡아낼 수 있다 !!
"""


# %% 오차행렬, 정밀도, 재현율을 모두 확인할 수 있는 함수 선언
def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    print("오차행렬:\n", confusion)
    print("정확도: {0:.4f}, 정밀도:{1:.4f}, 재현율:{2:.4f}".format(accuracy, precision, recall))


# %% 로지스틱 회귀 분류를 이용한 정밀도, 재현율 분석
from sklearn.linear_model import LogisticRegression

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
