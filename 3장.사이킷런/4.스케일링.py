import pandas as pd
from sklearn.datasets import load_iris


iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)  # type: ignore

print(iris_df.mean())  # 각 칼럼 평균값
print(iris_df.var())  # 각 칼럼 분산값

# 스탠다드 스케일링
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)  # 스케일링된 반환값의 타입은 넘파이 배열이다.
print(type(iris_scaled))

# 넘파이 배열을 다시 데이터프레임으로 변경,,,
iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)  # type: ignore
print(iris_df_scaled.mean())
print(iris_df_scaled.var())

# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
iris_scaled = scaler.fit_transform(iris_df)  # fit, transform 한번에 !!

iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)  # type: ignore
print("각 feature들의 최소값", iris_df_scaled.min())
print("각 feature들의 최대값", iris_df_scaled.max())

#  스케일링 주의사항 !!
import numpy as np

train_array = np.arange(0, 11).reshape(-1, 1)  # 스케일링하려면 2차원 배열
test_array = np.arange(0, 6).reshape(-1, 1)  # 스케일링하려면 2차원 배열

train_scaled = scaler.fit_transform(train_array)
print("원본 train_array 데이터:", np.round(train_array.reshape(-1), 2))  # 1차원으로
print("스케일된 train_array 데이터:", np.round(train_scaled.reshape(-1), 2))  # 1차원으로

test_scaled = scaler.fit_transform(test_array)  # 잘못된 방법 !!!!
print("원본 test_array 데이터:", np.round(test_array.reshape(-1), 2))  # 1차원으로
print("스케일된 test_array 데이터:", np.round(test_scaled.reshape(-1), 2))  # 1차원으로

"""
뭔가 이상하다 !!

스케일된 train_array 데이터: [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]
스케일된 test_array 데이터: [0.  0.2 0.4 0.6 0.8 1]

테스트피쳐의 최대값은 5이다.
그렇다면, 학습피쳐의 스케일링 척도에 맞게
스케일된 test_array 데이터: [0.  0.1 0.2 0.3 0.4 0.5]이 되어야한다. !!

이것을 주의해야한다. !!

스케일링시, 학습피쳐와 테스트피쳐의 스케일링 척도가 일치해야 잘못된 결과가 나오지 않는다.
"""

test_scaled = scaler.transform(test_array)  # 올바른 방법
print("원본 test_array 데이터:", np.round(test_array.reshape(-1), 2))  # 1차원으로
print("스케일된 test_array 데이터:", np.round(test_scaled.reshape(-1), 2))  # 1차원으로

"""
새롭게 fit을 호출해서 스케일링 척도를 초기화하지말고...
학습 피쳐의 스케일링 척도를 이어서 사용한다.

test_scaled = scaler.transform(test_array) # 올바른 방법
"""
