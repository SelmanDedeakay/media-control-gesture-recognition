from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("data.csv")
df = df.drop(["Unnamed: 0"], axis=1)
X_right = pd.concat([df[df["Class"] == 0], df[df["Class"] == 1],
                    df[df["Class"] == 4]], ignore_index=True)
Y_right = X_right["Class"]
X_right = X_right.drop(["Class"], axis=1)
X_left = pd.concat([df[df["Class"] == 2], df[df["Class"] == 3],
                   df[df["Class"] == 5]], ignore_index=True)
Y_left = X_left["Class"]
X_left = X_left.drop(["Class"], axis=1)

# For right hand

x_train, x_test, y_train, y_test = train_test_split(
    X_right, Y_right, test_size=0.10)


dct = {"Nearest Neighbors (k=5)":KNeighborsClassifier(),
        "Random Forest (Estimators = 100)": RandomForestClassifier(n_estimators=100),
        "Decision Tree": DecisionTreeClassifier(criterion="gini"),
        "Logistic Regression":LogisticRegression(multi_class="ovr"),
        "Support Vector Machine (SVM)":SVC(decision_function_shape="ovo",class_weight="balanced")}

print("For Model Right")
best_test = 0
best_model = None
for i in dct:
    model  = dct[i]
    model.fit(x_train.values, y_train)
    y_pred = model.predict(x_train.values)
    print(f"{i} train: {metrics.accuracy_score(y_train, y_pred)}")
    y_pred = model.predict(x_test.values)
    print(f"{i} test: {metrics.accuracy_score(y_test, y_pred)}")
    if best_test<metrics.accuracy_score(y_test, y_pred):
        best_model = i
print("Best model for right : "+best_model)


filename = 'model_right.sav'
pickle.dump(dct[best_model], open(filename, 'wb'))

# For left hand
print("For Model Left")
x_train, x_test, y_train, y_test = train_test_split(
    X_left, Y_left, test_size=0.05)

best_test = 0
best_model = None
for i in dct:
    model  = dct[i]
    model.fit(x_train.values, y_train)
    y_pred = model.predict(x_train.values)
    print(f"{i} train: {metrics.accuracy_score(y_train, y_pred)}")
    y_pred = model.predict(x_test.values)
    print(f"{i} test: {metrics.accuracy_score(y_test, y_pred)}")
    if best_test<metrics.accuracy_score(y_test, y_pred):
        best_model = i
print("Best model for left : "+best_model)


filename = 'model_left.sav'
pickle.dump(dct[best_model], open(filename, 'wb'))
