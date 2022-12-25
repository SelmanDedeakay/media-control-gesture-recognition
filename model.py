import pandas as pd

import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("data.csv")
print(len(df[df["Class"] == 3]))
df = df.drop(["Unnamed: 0"],axis=1)

X_right = pd.concat([df[df["Class"] ==0],df[df["Class"] ==1],df[df["Class"] ==4]],ignore_index=True)
Y_right = X_right["Class"]
print(X_right)
X_right = X_right.drop(["Class"],axis=1)

X_left = pd.concat([df[df["Class"] ==2],df[df["Class"] ==3],df[df["Class"] ==5]],ignore_index=True)

Y_left = X_left["Class"]

print(X_left)
X_left = X_left.drop(["Class"],axis=1)



#For right hand
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X_right,Y_right,test_size=0.10)



#model = KNeighborsClassifier()
#model = RandomForestClassifier(n_estimators=100)
#model = DecisionTreeClassifier(criterion="gini")
model = LogisticRegression(multi_class="ovr")
model = SVC(decision_function_shape="ovo",class_weight="balanced")
model.fit(x_train.values,y_train)

y_pred= model.predict(x_train.values)
print(metrics.accuracy_score(y_train, y_pred))
y_pred= model.predict(x_test.values)
print(metrics.accuracy_score(y_test, y_pred))


filename = 'model_right.sav'
pickle.dump(model, open(filename, 'wb'))

#For left hand

x_train,x_test,y_train,y_test = train_test_split(X_left,Y_left,test_size=0.10)

model = KNeighborsClassifier()
model = LogisticRegression(multi_class="ovr")
model = SVC(decision_function_shape="ovo",class_weight="balanced")
model.fit(x_train.values,y_train)

y_pred= model.predict(x_train.values)
print(metrics.accuracy_score(y_train, y_pred))
y_pred= model.predict(x_test.values)
print(metrics.accuracy_score(y_test, y_pred))
filename = 'model_left.sav'
pickle.dump(model, open(filename, 'wb'))