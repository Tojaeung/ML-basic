import numpy as np

# 넘파이 배열에는 다른타입 못드러감(리스트와 다르다)
# 다른타입이면 상위 타입으로 형변환 된다.
list1 = [1, 2, "test"]
array1 = np.array(list1)
# ['1' '2' 'test'] <U11
# 정수형이 문자열로 형변환
print(array1, array1.dtype)

print()

# 형변환 (정수 -> 실수)
array_int = np.array([1, 2, 3])
array_float = array_int.astype(np.float64)
# array_float = array_int.astype("float64")
print(array_float, array_float.dtype)

print()

# 편리하게 넘파이배열 생성
print(np.arange(10))
print(np.zeros((3, 2), dtype="int32"))  # 빈배열에 0을 넣는다
print(np.ones((3, 2)))  # 빈배열에 1을 넣는다, 기본타입은 float64

print()

# 넘파이배열 shape 변경
array2 = np.arange(10)

print(array2.reshape(2, 5))  # 1차원 배열을 2차원배열로 shape변경

print()

# 넘파이배열 shape -1 가변변경
print(array2.reshape(-1, 2))  # -1은 열의 개수에 따라 행이 가변적으로 변경된다.
print(array2.reshape(-1, 1))  # 2차원으로 변경, 열의 개수는 1
print(  # 1차원으로 변경
    array2.reshape(
        -1,
    )
)
