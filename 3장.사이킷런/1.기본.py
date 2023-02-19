import sklearn
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris_data = load_iris()

# 타입지정이 안되있어서 경고
iris_data.data  # type: ignore
iris_data.target  # type: ignore
iris_data.target_names  # type: ignore

dt_clf = DecisionTreeClassifier()
train_data = iris_data.data  # type: ignore
train_label = iris_data.target  # type: ignore

dt_clf.fit(train_data, train_label)
pred = dt_clf.predict(train_data)  # 학습용 데이터를 테스트 데이터롤 사용했다.
print("예측 정확도:", accuracy_score(train_label, pred))  # 그래서, 정확도가 100%


# train_test_split를 이용하면 자동으로 학습용, 테스트용 데이터를 분리해줘서 편리하다
X_train, X_test, y_train, y_test = train_test_split(
    iris_data.data, iris_data.target, test_size=0.3, random_state=121
)  # X는 학습용 데이터,테스트용 데이터 , y는 학습용 레이블,테스트용 레이블
dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(X_test)
print("예측 정확도: {0:.4f}".format(accuracy_score(y_test, pred)))

# 사이킷런 기본은 넘파이다. 그러나, 판다스로 가능
iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris_df["target"] = iris_data.target

ftr_df = iris_df.iloc[:, :-1]  # 맨끝 칼럼 제외 (피쳐데이터만 분리)
tgt_df = iris_df.iloc[:, -1]  # 맨끝 칼럼만 (타겟 분리)
X_train, X_test, y_train, y_test = train_test_split(
    ftr_df, tgt_df, test_size=0.3, random_state=121
)

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)  # 학습
pred = dt_clf.predict(X_test)
print("예측 정확도: {0:.4f}".format(accuracy_score(y_test, pred)))
