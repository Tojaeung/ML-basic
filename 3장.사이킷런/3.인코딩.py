import imp
import sklearn

from sklearn.preprocessing import LabelEncoder

items = ["TV", "냉장고", "전자렌지", "컴퓨터", "선풍기", "선풍기", "믹서", "믹서"]

encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)

# 레이블인코딩: 요소를 각각 숫자를 정해줌...
print("인코딩 변환값:", labels)
print("인코딩 클래스:", encoder.classes_)
print("디코딩 원본 값:", encoder.inverse_transform([1, 2, 3, 4, 5]))

# 원핫 인코딩: 하나만 1로 해서 레이블을 구분한다.
import numpy as np
from sklearn.preprocessing import OneHotEncoder

items = np.array(items).reshape(-1, 1)  # 원핫 인코딩은 2차원 배열을 받기떄문에 변환

oh_encoder = OneHotEncoder()
oh_encoder.fit(items)
oh_labels = oh_encoder.transform(items)

print("원핫인코딩 데이터:\n", oh_labels.toarray())  # type: ignore

"""
원핫인코딩은 2차원배열을 넣어줘야하는 등...
또 변환과정도 있고 불편하다.

그러니, 판다스의 데이터프레임을 이용하면 간단하다.
"""

# 원핫인코딩보다 간단한 판다스 이용 !!
import pandas as pd

df = pd.DataFrame({"item": ["TV", "냉장고", "전자렌지", "컴퓨터", "선풍기", "선풍기", "믹서", "믹서"]})
pd.get_dummies(df)