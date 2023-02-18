import numpy as np
import pandas as pd

# 리스트, nd배열 -> 데이터프레임 변환
list1 = [[11, 22, 33], [44, 55, 66]]
array1 = np.array(list1)

df_list = pd.DataFrame(list1, columns=["Column_A", "Column_B", "Column_C"])
df_array = pd.DataFrame(array1, columns=["Column_A", "Column_B", "Column_C"])

# 딕셔너리 -> 데이터프레임 변환
dict = {
    "col1": [1, 11],
    "col2": [2, 22],
    "col3": [3, 33],
}

df_dict = pd.DataFrame(dict)

# 데이터프레임 -> 넘파이배열, 리스트, 딕셔너리 변환
array2 = df_dict.values
print(type(array2))

list2 = df_dict.values.tolist()
print(type(list2))

dict2 = df_dict.to_dict("list")  # 딕셔너리 value가 리스트형이라는 뜻
print(type(dict2))
