import pandas as pd
import numpy as np
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
y=train_data["Survived"]

#このままだとSexが文字列なので、数値に変換する
x=pd.get_dummies(train_data[["Pclass","Sex","SibSp","Parch"]])
X_test=pd.get_dummies(test_data[["Pclass","Sex","SibSp","Parch"]])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
from lightgbm import LGBMClassifier
model=LGBMClassifier(n_estimators=200,max_depth=10,random_state=1)
model.fit(X_train,y_train)
prediction=model.predict(X_test)

output=pd.DataFrame({"PassengerId":test_data.PassengerId,"Survived":prediction})
output.to_csv("Submission.csv",index=False) #index=Falseでindexを出力しないようにする