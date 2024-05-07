import pandas as pd
import numpy as np

df_train=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')

print(df_train.dtypes)
print(df_train.isnull().sum())
print(df_test.isnull().sum())