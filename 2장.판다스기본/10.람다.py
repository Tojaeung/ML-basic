import pandas as pd

titanic_df = pd.read_csv("../data/titanic_train.csv")

# 파이썬 람다 기본사용
squre = lambda x: x**2
print(squre(3))

a = [1, 2, 3]
squres = map(lambda x: x**2, a)
print(list(squres))

# 판다스에서 람다 사용
titanic_df["Name_len"] = titanic_df["Name"].apply(lambda x: len(x))
titanic_df[["Name", "Name_len"]]

# 람다 조건 분기
titanic_df["Child_or_Adult"] = titanic_df["Age"].apply(
    lambda x: "Child" if x <= 19 else "Adult"  # 19세이하이면 Child, 나머지는 Adult (표현 독특)
)
titanic_df[["Age", "Child_or_Adult"]].head(10)


# 람다 조건 분기가 복잡하다면??
def get_category(age) -> str:
    if age <= 5:
        return "아기"
    elif age <= 12:
        return "어린이"
    elif age <= 18:
        return "학생"
    elif age <= 25:
        return "대학생"
    elif age <= 35:
        return "청년"
    elif age <= 60:
        return "중년"
    else:
        return "노인"


titanic_df["Age_category"] = titanic_df["Age"].apply(lambda x: get_category(x))
titanic_df[["Age", "Age_category"]]
titanic_df["Age_category"].value_counts()  # 칼럼 값마다 갯수
