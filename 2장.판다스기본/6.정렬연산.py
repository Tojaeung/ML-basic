import pandas as pd

titanic_df = pd.read_csv("../data/titanic_train.csv")

# 정렬
titanic_df.sort_values(by="Name", ascending=False)  # asc는 기본 True

# 연산
titanic_df[["Age", "Fare"]].mean()
titanic_df[["Age", "Fare"]].sum()
titanic_df[["Age", "Fare"]].count()
