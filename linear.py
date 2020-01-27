import pandas as pd

#ucitat fajl

feature_cols = ['bla', 'bla', 'bla']

x = fajl[feature_cols]
y = fajl.label

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state = 0)


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_predict = logreg.predict(X_test)


from sklearn import matrics
cnf_matrix = matrics.confusion_matrix(y_test, y_predict)
cnf_matrix

print("Accuracy: " metric.accuracy_score(y_test, y_predict))
print("Precision: " metric.precision_score(y_test, y_predict))
print("Recall: " metric.recall(y_test, y_predict))