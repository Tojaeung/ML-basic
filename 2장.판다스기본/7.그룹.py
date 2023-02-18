import pandas as pd

titanic_df = pd.read_csv("./data/train.csv")

# Group by
titanic_groupby_pclass = titanic_df.groupby(by="Pclass")
type(titanic_groupby_pclass)  # 신기하게도... groupby 타입이다

titanic_groupby_pclass.count()
titanic_groupby_pclass[["Age", "Fare"]].count()

# 서로다른 연산 수행하려면 각각해야함
titanic_groupby_pclass["Age"].max()
titanic_groupby_pclass["Age"].min()

# agg메서드를 사용하면 간단하다.
titanic_groupby_pclass["Age"].agg([max, min])

# 서로 다른 칼럼에 다른 연산을 적용하고 싶을떄 어떻게??
format = {"Age": "max", "SibSp": "sum", "Fare": "mean"}  # 딕셔너리로 간단히 !!
titanic_groupby_pclass.agg(format)

# 주의할 점 !!
format = {"Age": "max", "Age": "min", "Fare": "mean"}  # Age칼럼이 중복된다...
titanic_groupby_pclass.agg(format)  # 결국, 마지막 min연산이 적용되서 1개의 Age만 출력되낟.

# Named Aggregation (이름 지어진 연산)
titanic_groupby_pclass.agg(
    age_max=("Age", "max"),
    age_min=("Age", "min"),
    fare_mean=("Fare", "mean"),
)  # 칼럼명을 따로 지어준다.

# 다른 방법
titanic_groupby_pclass.agg(
    age_max=pd.NamedAgg(column="Age", aggfunc="max"),
    age_min=pd.NamedAgg(column="Age", aggfunc="min"),
    fare_mean=pd.NamedAgg(column="Fare", aggfunc="mean"),
)
