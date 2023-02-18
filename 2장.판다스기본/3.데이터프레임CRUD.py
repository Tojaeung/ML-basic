import pandas as pd

titanic_df = pd.read_csv("./data/train.csv")

# 데이터프레임 칼럼생성
titanic_df["나다호다"] = 0  # 새로운 칼럼 '나다호다' 생성
titanic_df["HappyNewYear"] = titanic_df["Age"] + 1  # 새로운 칼럼을 기존 칼럼을 이용해서 생성

# 데이터프레임 칼럼 업데이트
titanic_df["Age"] = titanic_df["Age"] + 100

# 데이터프레임 삭제 -> 칼럼, 로우 삭제가능
titanic_drop_df = titanic_df.drop("나다호다", axis=1)  # axis 0은 행 제거, 1은 칼럼 제거

titanic_df.drop(
    "HappyNewYear", axis=1, inplace=True
)  # inplace가 True이면, 데이터프레임 원본에서 삭제된다.
titanic_df.drop(["HappyNewYear", "나다호다"], axis=1, inplace=True)  # 여러 칼럼 삭제

titanic_df.drop([0, 1, 2], axis=0, inplace=True)  # 앞의 3개 행 제거
