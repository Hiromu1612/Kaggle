import pandas as pd
import numpy as np

df_train=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')

print(df_train.dtypes)
print(df_train.isnull().sum())
print(df_test.isnull().sum())

#object(文字列)・nullがないデータを使う
X=df_train[['Pclass','SibSp','Parch']]
y=df_train[["Survived"]]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

X_test=df_test[['Pclass','SibSp','Parch']]
submit=df_test[['PassengerId']]
submit['Survived']=model.predict(X_test) #submitにSurvived列を追加

#CSVファイルに書き出し
submit.to_csv("submit.csv", index=False) #index=Falseで行番号を出力しない