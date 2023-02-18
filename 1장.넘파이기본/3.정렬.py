import numpy as np

org_array = np.array([3, 1, 9, 5])
print("원본배열", org_array)

# 행렬의 정렬
sort_array = np.sort(org_array)
print(sort_array)  # 새롭게 정렬된 행렬 할당
print(org_array)

sort_array2 = org_array.sort()
print(org_array)  # 정렬
print(sort_array2)  # None

sort_array_desc = np.sort(org_array)[::-1]
print(sort_array_desc)  # 내림차순 정렬

print()

# 행렬의 axis기준으로 정렬
array2d = np.array([[8, 12], [7, 1]])
print("원본행렬\n", array2d)

sort_array_axis0 = np.sort(array2d, axis=0)  # 2차배열의 axis0(행)을 기준으로 정렬
print(sort_array_axis0)

sort_array_axis1 = np.sort(array2d, axis=1)  # 2차배열의 axis1(열)을 기준으로 정렬
print(sort_array_axis1)

print()

# argsort(): 정렬된 행렬의 인덱스를 반환
sort_indices = np.argsort([3, 1, 9, 5])
print(sort_indices)

sort_indices2 = np.argsort([3, 1, 9, 5])[::-1]  # 내림차순
print(sort_indices2)

print()

# argsort()는 넘파이 맵핑에 유용하게 사용된다
name_array = np.array(["토재웅", "나다호다", "안중저격수"])
score_array = np.array([87, 43, 94])

sort_indices_asc = np.argsort(score_array)
print(sort_indices_asc)  # 인덱스 기억
print(name_array[sort_indices_asc])  # 성적순으로 이름 출력으로 사용

print()

# 행렬내적 (선형대수 연산)
# 각각 행과열을 곱한값을 모두 더한 값을 행렬에 넣음
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8], [9, 10], [11, 12]])
dot_product = np.dot(A, B)
print(dot_product)

# 전치행렬 (선형대수 연산)
# 행과 열을 바꿈
A = np.array([[1, 2], [3, 4]])
transpose_mat = np.transpose(A)
print(transpose_mat)
