import pandas as pd

titanic_df = pd.read_csv("./data/train.csv")

# 인덱스
indexes = titanic_df.index  # 인덱스 정보
indexes.values  # 1차원
indexes[:5].values  # 인덱스 자체 슬라이싱 가능
indexes.values[:5]
indexes[5]  # 인덱스 찾기

# 인덱스 재설정
titanic_reset_df = titanic_df.reset_index(
    inplace=False
)  # 기존 인덱스가 칼럼으로 추가된 후, 인덱스가 새로 설정된다.
titanic_reset_df = titanic_df.reset_index(
    drop=True, inplace=False
)  # Drop설정을 하면 칼럼으로 추가하지 않는다.

titanic_df["Pclass"].value_counts()  # value값을 인덱스로 하고 있다.
pclass_count = (
    titanic_df["Pclass"].value_counts().reset_index()
)  # value값 인덱스를 칼럼으로 넣어준다
pclass_count.rename(columns={"index": "Pclass", "Pclass": "Pclass_count"})  # 칼럼명 변경
