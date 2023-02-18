import pandas as pd

titanic_df = pd.read_csv("./data/train.csv")

# isna(): Nan이면 True
titanic_df.isna().head(3)
titanic_df.isna().sum()  # Nan건수를 카운트한다. (count X)

# fillna(): Nan을 특정값으로 채운다

# Inplace 설정을 하지않고 직접 칼럼에 할당했다.
titanic_df["Cabin"] = titanic_df["Cabin"].fillna("C000")
titanic_df["Age"] = titanic_df["Age"].fillna(titanic_df["Age"].mean())
titanic_df["Embarked"] = titanic_df["Embarked"].fillna(
    titanic_df["Embarked"].fillna("S")
)

titanic_df.isna().sum()  # 위의 처리떄문에 남은 Nan이 없다.
