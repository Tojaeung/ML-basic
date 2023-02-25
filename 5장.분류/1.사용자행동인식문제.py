# 데이터 다운로드 링크
# https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

# %% 라이브러리 호출
import pandas as pd

# %% 데이터 불러오기 및 확인
feature_name_df = pd.read_csv(
    "C:\\Users\\tojaeung\\datasets\\human_activity\\features.txt",
    sep="\\s+",  # \s+ 정규표현식으로 공백을 기준으로 나눈다는 의미
    header=None,  # 데이터를 불러올떄 헤더정보(칼럼이름)이 없다는것을 알려주고 따로 넣어준다.
    names=["column_index", "column_name"],
)

feature_name = feature_name_df.iloc[:, 1].values
print("전체 피처명에서 10개만 추출:", feature_name[:10])

# %% 중복되는 칼럼명 확인...
feature_dup_df = feature_name_df.groupby("column_name").count()
print(feature_dup_df[feature_dup_df["column_index"] > 1].count())
feature_dup_df[feature_dup_df["column_index"] > 1].head()

# %% 중복되는 칼럼명 문제를 해결하기 위해 cumcount() 메서드를 사용 (연습)
df = pd.DataFrame([["a"], ["a"], ["a"], ["b"], ["b"], ["a"]], columns=["A"])
print(df.groupby("A").cumcount())


# %% 편의 함수 생성
# 중복되는 칼럼명 끝에 0,1,2 순서대로 붙여서 중복안되게 하기 (판단스에서 칼럼명 중복되면 에러발생하기 떄문에...)
def get_new_feature_name_df(old_feature_name_df):
    feature_dup_df = pd.DataFrame(
        data=old_feature_name_df.groupby("column_name").cumcount(), columns=["dup_cnt"]
    )

    # 인덱스 칼럼 생성해서 인덱스를 부여함 (0부터)
    feature_dup_df = feature_dup_df.reset_index()

    # 두개의 데이터프레임 결합을 위해 merge메서드 사용 (outer는 sql에서 outer 생각하면 된다.)
    new_feature_name_df = pd.merge(
        old_feature_name_df.reset_index(), feature_dup_df, how="outer"
    )

    # 중복되는것은 값 뒤에 0,1,2하나씩 붙여주는 로직 (람다 사용 !!)
    new_feature_name_df["column_name"] = new_feature_name_df[
        ["column_name", "dup_cnt"]
    ].apply(lambda x: x[0] + "_" + str(x[1]) if x[1] > 0 else x[0], axis=1)

    # 쓸데없는 index칼럼은 제거
    new_feature_name_df = new_feature_name_df.drop(["index"], axis=1)

    return new_feature_name_df


def get_human_dataset():
    feature_name_df = pd.read_csv(
        "C:\\Users\\tojaeung\\datasets\\human_activity\\features.txt",
        sep="\\s+",
        header=None,
        names=["column_index", "column_name"],
    )

    # 데이터 전처리를 위한 편의함수 사용 (get_new_feature_name_df)
    new_feature_name_df = get_new_feature_name_df(feature_name_df)

    # 데이터 가져올때 칼럼명으로 사용한다. (리스트)
    feature_name = new_feature_name_df.iloc[:, 1].values.tolist()  # type: ignore

    X_train = pd.read_csv(
        "C:\\Users\\tojaeung\\datasets\\human_activity\\train\\X_train.txt",
        sep="\\s+",
        names=feature_name,
    )
    X_test = pd.read_csv(
        "C:\\Users\\tojaeung\\datasets\\human_activity\\test\\X_test.txt",
        sep="\\s+",
        names=feature_name,
    )
    y_train = pd.read_csv(
        "C:\\Users\\tojaeung\\datasets\\human_activity\\train\\y_train.txt",
        sep="\\s+",
        header=None,
        names=["action"],
    )
    y_test = pd.read_csv(
        "C:\\Users\\tojaeung\\datasets\\human_activity\\test\\y_test.txt",
        sep="\\s+",
        header=None,
        names=["action"],
    )

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = get_human_dataset()

print(X_train.info())
print(y_train["action"].value_counts())

# %% 결정트리 분류
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dt_clf = DecisionTreeClassifier(random_state=156)
dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("결정 트리 예측 정확도:{0:4f}".format(accuracy))
print("결정트리 기본 하이퍼파라미터:\n", dt_clf.get_params())

# %% GridSearchCV
from sklearn.model_selection import GridSearchCV

params = {"max_depth": [6, 8, 10, 12, 16, 20, 24], "min_samples_split": [16]}

grid_cv = GridSearchCV(dt_clf, param_grid=params, scoring="accuracy", cv=5, verbose=1)
grid_cv.fit(X_train, y_train)
print("GridSearchCV 최고 평균 정확도 수치: {0:.4f}".format(grid_cv.best_score_))
print("GridSearchCV 최적 하이퍼 파라미터:", grid_cv.best_params_)

# %% GridSearchCV에서 깊이, split에 따라 평균 정확도가 얼마나 나오는지 확인하자

#  grid_cv.cv_results_가 보기에 불편해서 판다스 데이터프레임으로 변경
cv_results_df = pd.DataFrame(grid_cv.cv_results_)
cv_results_df[["param_max_depth", "mean_test_score"]]
"""
위에서 출력한 GridSearchCV 최적 하이퍼 파라미터와 같이...
[8, 16] 파라미터 조합이 정확도가 제일 좋다.
"""

# %% 결정트리에 적용해보자 !! 결과가 어떨까?
max_depths = [6, 8, 10, 12, 16, 20, 24]
for depth in max_depths:
    dt_clf = DecisionTreeClassifier(
        max_depth=depth, min_samples_split=16, random_state=156
    )
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print("max_depth = {0}, 정확도: {1:.4f}".format(depth, accuracy))

"""
결정트리 분류에서도 GridSearchCV와 마찬가지로
[8,16] 파라미터에서 정확도가 가장 높다 !!
"""

# %% 파라미터 변화
params = {"max_depth": [8, 12, 16, 20], "min_samples_split": [16, 24]}

grid_cv = GridSearchCV(dt_clf, param_grid=params, scoring="accuracy", cv=5, verbose=1)
grid_cv.fit(X_train, y_train)
print("GridSearchCV 최고 평균 정확도 수치: {0:.4f}".format(grid_cv.best_score_))
print("GridSearchCV 최적 하이퍼 파라미터:", grid_cv.best_params_)
"""
파라미터를 변화시켜도 [8,16] 조합이 최적으로 나온다.
"""

# %% grid_cv.best_estimator_ 최적의 파라미터로 학습된 모델을 반환한다.
best_df_clf = grid_cv.best_estimator_
pred = best_df_clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print("결정트리 예측 정확도: {0:.4f}".format(accuracy))

# %% 최적의 학습 모델에서 피쳐의 중요도를 확인하는게 무슨 의미가 있는가??
import matplotlib.pyplot as plt
import seaborn as sns

ftr_importance_values = best_df_clf.feature_importances_

# ftr_importance_values 보기가 안좋아서 판다스 시리즈를 사용했다.
ftr_importance = pd.Series(ftr_importance_values, index=X_train.columns)
ftr_top20 = ftr_importance.sort_values(ascending=False)[:20]

# 뭐지?? 맷플랏이랑 씨본이 다른 객체인데??? 융합이 되서 출력되네????????
plt.figure(figsize=(8, 6))
plt.title("Feature importances Top 20")
sns.barplot(x=ftr_top20, y=ftr_top20.index)
plt.show()
