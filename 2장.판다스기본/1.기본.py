import pandas as pd

titanic_df = pd.read_csv("../data/titanic_train.csv")
print(type(titanic_df))  # 데이터프레임


titanic_df.head(3)
titanic_df.tail()  # 기본값 5

titanic_df.shape  # 넘파이 shape와 동일

titanic_df.columns
titanic_df.index
titanic_df.index.values

titanic_df.info()  # 널 건수, 칼럼명 등등
titanic_df.describe()  # 숫자 칼럼에 대해서만 평균 ,표준편차,4분위분포도 제공

titanic_df["Age"].value_counts()  # 각 value가 몇개 중복되는지 카운트
titanic_df["Embarked"].value_counts()  # 기본은 null값 포함 안함
titanic_df["Embarked"].value_counts(dropna=False)  # drona 설정하면 null값도 포함한다.
titanic_df[["Pclass", "Embarked"]].value_counts()  # 복수칼럼 지정 가능
