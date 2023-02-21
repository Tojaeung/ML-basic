# 타이타닉 생존자 예측
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

"""
PassengerId: 탑승자 일련번호
survived: 생존여부 (사망 0 , 생존 1)
Pclass: 선실등급 (1등석 1, 2등석 2, 3등석 3)
sibsp: 같이 탑승한 형제자매 또는 배우자 인원수
parch: 같이 탑승한 부모님 또는 어린이 인원수
ticket: 티켓번호
fare: 요금
cabin: 선실번호
embarked: 중간 정착 항구 (C = Cherbourg, Q = Queenstown, S = Southampton)
"""
titanic_df = pd.read_csv("../data/titanic_train.csv")
titanic_df.head()

# 데이터 분석
titanic_df.info()
titanic_df.describe()

# Null 칼럼에 대한 처리
titanic_df["Age"].fillna(titanic_df["Age"].mean(), inplace=True)
titanic_df["Cabin"].fillna("N", inplace=True)
titanic_df["Embarked"].fillna("N", inplace=True)
print("데이터 세트 Null값 개수:", titanic_df.isnull().sum().sum())

# 각 칼럼 분포보기
titanic_df["Sex"].value_counts()
titanic_df["Cabin"].value_counts()
titanic_df["Embarked"].value_counts()

# Cabin데이터 단순화 (앞글자만 사용)
titanic_df["Cabin"].str[:1].value_counts()
titanic_df["Cabin"] = titanic_df["Cabin"].str[:1]

# 성별, 선실등급에 따른 생존자 확인
titanic_df.groupby(["Sex", "Survived"])["Survived"].count()
sns.barplot(x="Sex", y="Survived", data=titanic_df)
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=titanic_df)


# 연령,성별에 따른 생존자 확인
def get_category(age: int) -> str:
    if age <= -1:
        return "UnKnown"
    elif age <= 5:
        return "Baby"
    elif age <= 12:
        return "Children"
    elif age <= 18:
        return "Teenager"
    elif age <= 25:
        return "Student"
    elif age <= 35:
        return "Young Adult"
    elif age <= 50:
        return "Adult"
    else:
        return "Elderly"


group_names = [
    "UnKnown",
    "Baby",
    "Children",
    "Teenager",
    "Student",
    "Young Adult",
    "Adult",
    "Elderly",
]  # X축을 순차적으로 표시하기 위해

plt.figure(figsize=(10, 6))  # 막대그래프 크기 설정
titanic_df["Age_Cat"] = titanic_df["Age"].apply(lambda x: get_category(x))
sns.barplot(x="Age_Cat", y="Survived", hue="Sex", data=titanic_df, order=group_names)
titanic_df.drop("Age_Cat", axis=1, inplace=True)
"""
Age_Cat 칼럼을 새로 만들어서 정리한 데이터를 두고
씨본을 이용해서 시각화 한뒤...
다시 데이터에서 삭제시켜준다.
"""
