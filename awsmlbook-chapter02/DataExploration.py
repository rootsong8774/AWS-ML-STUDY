# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python [conda env:AWS_ML_BOOK] *
#     language: python
#     name: conda-env-AWS_ML_BOOK-py
# ---

# +
import numpy as np
import pandas as pd

# Pandas 데이터프레임 형식으로 파일 내용 불러오기
input_file = './datasets/titanic_dataset/original/train.csv'
df_titanic = pd.read_csv(input_file)
# -

df_titanic.shape

# 12개 열 이름 확인하기
print(df_titanic.columns.values)

# 결측값 확인하기
df_titanic.info()

# 데이터프레임에서 결측값(Missing Value)을 찾는 다른 방법
df_titanic.isnull().sum()

df_titanic.head()

input_file = './datasets/random_30column.csv'
pd.set_option('display.max_columns',None)
df_random30 = pd.read_csv(input_file)
df_random30.head()

pd.set_option('display.max_columns',4)
df_random30.head()

# 데이터프레임의 인덱스 확인
print (df_titanic.index.name)

df_titanic.set_index("PassengerId", inplace=True)

print (df_titanic.index.name)

pd.set_option('display.max_columns', None)
df_titanic.head()

# index 설정 후 데이터프레임 shape 확인
df_titanic.shape

# +
# 목적변수를 추출해 새로운 데이터프레임에 저장하기
df_titanic_target=df_titanic.loc[:,['Survived']]

# 나머지 10개의 피처를 새로운 데이터프레임에 저장하기
df_titanic_features = df_titanic.drop(['Survived'],axis=1)
# -

#목적변숫값 분포 확인하기
df_titanic_target['Survived'].value_counts()

# Embarked 속성 값들의 개수 확인하기(NaN을 포함)
df_titanic_features['Embarked'].value_counts(dropna=False)

# 목적변수 히스토그램
# %matplotlib inline
import matplotlib.pyplot as plt
df_titanic_target.hist(figsize=(5,5))

# 피처 값을 (10, 10) 사이즈의 히스토그램으로 표현하기
df_titanic_target.hist(figsize=(10,10))

# Age 변수의 히스토그램
# 다양한 빈(bin) 크기를 적용해 히스토그램을 그려보는 것은 데이터의 분포를 이해하는 데 도움이 된다.
df_titanic_features.hist(column='Age', figsize=(5,5), bins=8)

# value_counts()의 결과를 통해 나타낸 범주형 피처 'Embarked'의 히스토그램 (NaN 포함)
vc = df_titanic_features['Embarked'].value_counts(dropna=False)
vc.plot(kind='bar')

# 데이터의 통계적 특성 확인하기
df_titanic_features.describe()


