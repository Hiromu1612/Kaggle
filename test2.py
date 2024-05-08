import pandas as pd
import numpy as np

df_train=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')

X=df_train[['Pclass','SibSp','Parch','Sex']]
y=df_train[["Survived"]]

#! Sex列を追加
#このままだとSexが文字列なので、数値に変換する
X=pd.get_dummies(X, columns=["Pclass","Sex"])

#Trueを1、Falseを0に変換
X=X.astype(np.int64)

#! Fare列をconcatで追加  axis=0(行) 1(縦)
X=pd.concat([X, df_train["Fare"]], axis=1)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=0)


#Fareの値が大きくて、他の列の値との差が大きいので、正規化する
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train) #fitで平均と標準偏差を計算

X_train=scaler.transform(X_train) #transformで正規化
X_test=scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))




#!提出 df_testのデータも同様に加工 
X_test_submit=df_test[['Pclass','SibSp','Parch','Sex']]
y_test_submit=df_test[['PassengerId']]

X_test_submit=pd.get_dummies(X_test_submit, columns=["Pclass","Sex"])
X_test_submit=X_test_submit.astype(np.int64)
X_test_submit=pd.concat([X_test_submit, df_train["Fare"]], axis=1)
X_test_submit["Fare"].fillna(df_test["Fare"].mean()) #Fareの欠損値を平均で埋める inplace=Trueで元のデータを更新
print(X_test_submit.isnull().sum())
print(X_test_submit.head())

scaler=StandardScaler()
scaler.fit(X_test_submit)
X_test_submit=scaler.transform(X_test_submit)

y_test_submit['Survived']=model.predict(X_test_submit) #submitにSurvived列を追加

y_test_submit.to_csv("submit_2.csv", index=False) #index=Falseで行番号を出力しない