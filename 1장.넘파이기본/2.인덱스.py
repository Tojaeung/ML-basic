import numpy as np

array1d = np.arange(start=1, stop=10)
array2d = array1d.reshape(3, 3)
print(array2d)

print()

# 슬라이싱 인덱싱 (연속되는 인덱싱)
print(array2d[0:2, 0:2])
print(array2d[1:3, :])
print(array2d[:2, 1:])
print(array2d[:2, 0])  # 단일 값 0떄문에 1차원으로 차원축소

print()

# 팬시 인덱싱 (연속되지 않는 인덱싱)
print(array2d[[0, 2], 2])  # 행을 연속되지 않는 0, 2로 지정하였다.

print()

# 불린 인덱싱 (조건으로 인덱싱)
print(array1d[array1d > 5])
