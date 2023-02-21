import pandas as pd

titanic_df = pd.read_csv("../data/titanic_train.csv")

# iloc 위치기반 인덱싱
data = {
    "Name": ["Apple", "Banana", "Kiwi", "Cherry"],
    "Year": [2011, 2016, 2015, 2015],
    "Gender": ["Male", "Female", "Male", "Male"],
}

data_df = pd.DataFrame(
    data,
    index=[
        "one",
        "two",
        "three",
        "four",
    ],
)

data_df.iloc[1, 0]
data_df.iloc[2, 1]

data_df.iloc[0:2, [0, 1]]
data_df.iloc[0:2, 0:2]

data_df.iloc[:, -1]  # 맨 마지막 칼럼 가져오기
data_df.iloc[:, :-1]  # 맨 마지막 칼럼 제외 모든 칼럼 가져오기

data_df.iloc[data_df.Year >= 2014]  # 불린 인덱싱을 지원하지않아서 에러발생 !!

# loc 명칭기반 인덱싱
data_df.loc["three", "Name"]
data_df.loc["one":"two", ["Name", "Year"]]  # 슬라이싱과 다르게 마지막을 포함한다.(헷갈,,,)
data_df.loc[data_df.Year >= 2014]  # loc는 불링인덱싱이 가능하다.

# 불링인덱싱으로 필터링
titanic_df[titanic_df["Age"] > 60][["Name", "Age"]]

titanic_df[
    (titanic_df["Age"] > 60)
    & (titanic_df["Pclass"] == 1)
    & (titanic_df["Sex"] == "female")
]  # 3개의 조건을 모두 만족하는것

condition1 = titanic_df["Age"] > 60
condition2 = titanic_df["Pclass"] == 1
condition3 = titanic_df["Sex"] == "female"
titanic_df[condition1 & condition2 & condition3]  # 조건을 따로 만들어서 대입
