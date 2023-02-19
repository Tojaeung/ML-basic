import imp
from weakref import ref
import sklearn
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

iris = load_iris()

features = iris.data  # type: ignore
label = iris.target  # type: ignore
dt_clf = DecisionTreeClassifier(random_state=156)

kfold = KFold(n_splits=5)  # k-fold로 데이터세트가 5개로 분할된다.
cv_accuracy = []

print("봋꽃 데이터 세트 크기:", features.shape[0])

# k-fold: 모의고사 여러번 본다.
# 120 ~ 149 검증데이터
# 90 ~ 119 검증데이터
# 60 ~ 89 검증데이터
# 30 ~ 59 검증데이터
# 0 ~ 29 검증데이터
for train_index, test_index in kfold.split(features):  # 5번 루프를 돈다
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]

    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)

    accuracy = np.round(accuracy_score(y_test, pred), 4)
    print("루프당 검증 정확도:", accuracy)
    cv_accuracy.append(accuracy)

print("평균 검증 정확도:", np.mean(cv_accuracy))

# k-fold 사용할때 문제점 !!
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)  # type: ignore
iris_df["label"] = iris.target  # type: ignore
iris_df["label"].value_counts()

kfold = KFold(n_splits=3)

for train_index, test_index in kfold.split(iris_df):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]

    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)

    accuracy = np.round(accuracy_score(y_test, pred), 4)
    print("루프당 검증 정확도:", accuracy)
    cv_accuracy.append(accuracy)

print("평균 검증 정확도:", np.mean(cv_accuracy))  # 0.56

"""
왜 검증 정확도가 떨어지는가??
iris_df["label"].value_counts()를 찍어보면 알다시피,
0 ~ 49 = 0
50 ~ 99  = 1
100 ~ 149 = 2

위의 데이터셋에서 kfold = KFold(n_splits=3)를 사용해서 나눈다면...
0 , 1를 학습하고 2로 검증하게 된다.. 또는...
1, 2를 학습하고 0으로 검증하게된다. (잘못된 교차)
이것은 마치... 국어, 수학 공부하고 영어시험 치르는 꼴이다 !! 성적이 잘나올리 없다 !!

이 문제를 해결하기 위해 StratifiedKFold를 사용한다.
"""

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3)

for train_index, test_index in skf.split(iris_df, iris_df["label"]):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]

    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)

    accuracy = np.round(accuracy_score(y_test, pred), 4)
    print("루프당 검증 정확도:", accuracy)
    cv_accuracy.append(accuracy)

print("평균 검증 정확도:", np.mean(cv_accuracy))  # 0.96

"""
StratifiedKFold를 사용해서 정확도가 0.96나왔다 !!
split메서드 인자에 iris_df["label"]를 넣어서 label 값의 분포도에 따라...
적절하게 학습, 검증 데이터를 나누어주었다. (매우 고른 분포도를 보여준다..)

1. 검증을 하려면 검증데이터에 대한 학습도 이루어져야한다.
2. 타겟값에 대한 분포가 학습과 검증 동일하게 이루어져야한다.
"""

# kfold보다 간편한 cross_val_score
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    dt_clf, features, label, scoring="accuracy", cv=3
)  # StratifiedKFold이 적용된다.
print("교차 검증별 정확도", np.round(scores, 4))
print("평균 검증 정확도", np.round(np.mean(scores), 4))  # 0.96

"""
코드 길이만 봐도 k-fold보다 간단하다 즉, 편리하다
"""

# gridSearchCV: 최적의 하이퍼 파라미터를 찾아서 최고의 결과를 !!
from sklearn.model_selection import GridSearchCV, train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=121)  # type: ignore
parameters = {"max_depth": [1, 2, 3], "min_samples_split": [2, 3]}

grid_dtree = GridSearchCV(
    dt_clf, param_grid=parameters, cv=3, refit=True, return_train_score=True
)

grid_dtree.fit(X_train, y_train)  # 하이퍼 파라미터들을 순차적으로 학습/평가

scores_df = pd.DataFrame(grid_dtree.cv_results_)  # 결과값이 딕셔너리로 저장되서 데이터프레임으로 변경(가독성)
scores_df[
    [
        "params",
        "mean_test_score",
        "rank_test_score",
        "split0_test_score",
        "split1_test_score",
        "split2_test_score",
    ]
]  # 칼럼이 복잡해서 정리

print("최적 파라미터", grid_dtree.best_params_)
print("최고 정확도", grid_dtree.best_score_)  # 0.97

# 여기는 test_size=0.2일떄 정확도 (gridSearchCV 최적파라미터와 차이 거의 없음...)
pred = grid_dtree.predict(X_test)
print("테스트 데이터 정확도: {0:.4f}".format(accuracy_score(y_test, pred)))  # 0.96
