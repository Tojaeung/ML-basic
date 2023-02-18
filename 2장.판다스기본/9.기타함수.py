import pandas as pd

titanic_df = pd.read_csv("./data/train.csv")

# nunique(): 칼럼의 고유값이 몇개인지 파악
titanic_df["Pclass"].nunique()  # 3
titanic_df["Survived"].nunique()  # 2
titanic_df["Name"].nunique()  # 891

# replace(): 활용도 높음 !!
titanic_df["Sex"] = titanic_df["Sex"].replace(
    {"male": "man", "female": "woman"}
)  # 복수변경은 딕셔너리로 replace
