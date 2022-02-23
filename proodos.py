import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from scipy.sparse import data
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import CategoricalNB, GaussianNB
import numpy as np
from sklearn.neural_network import MLPRegressor

# carMarket = pd.read_csv("./testing.csv")

# encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

# encoder.fit(carMarket.loc[:, ['Pool']])
# transformedCarType = encoder.transform(carMarket.loc[:, ['Pool']])
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(transformedCarType, carMarket.loc[:, 'Price'])
# fig = plt.figure()
# tree.plot_tree(clf, class_names=['Low', 'High'], filled=True)
# plt.show()
# print(carMarket)

# absfreq = pd.crosstab(carMarket.Pool, carMarket.Price)
# freq = pd.crosstab(carMarket.Pool, carMarket.Price, normalize='index')
# freqSum = pd.crosstab(carMarket.Pool, carMarket.Price, normalize='all').sum(axis=1)
# print(absfreq)

# GINI_Yes = 1 - freq.loc["Yes", "Low"]**2 - freq.loc["Yes", "High"]**2
# GINI_No = 1 - freq.loc["No", "Low"]**2 - freq.loc["No", "High"]**2
# GINI_Pool = freqSum.loc["Yes"] * GINI_Yes + freqSum["No"] * GINI_No
# print(GINI_Pool)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

testingSet = pd.read_csv("./testing.csv")
trainingSet = pd.read_csv("./training.csv")

# X = trainingSet.loc[:, ["Size", "Rooms", "Floors"]]
# y = trainingSet.loc[:, "Price"]

# Xtest = testingSet.loc[:, ["Size", "Rooms", "Floors"]]
# ytest = testingSet.loc[:, "Price"]

# clf = GaussianNB()
# clf.fit(X, y)
# pred = clf.predict(Xtest)
# predprob = clf.predict_proba(Xtest)

# print("Accuracy: ", accuracy_score(ytest, pred))

# clf =GaussianNB()
# clf.fit(X, y)
# pred = clf.predict(X)
# predprob = clf.predict_proba(X)

# print("Accuracy: ", accuracy_score(y, pred))

X = trainingSet.loc[:, ["Size", "Floors", "Owners"]]
y = trainingSet.loc[:, "NumPrice"]

Xtest = testingSet.loc[:, ["Size", "Floors", "Owners"]]
ytest = testingSet.loc[:, "NumPrice"]

clf = MLPRegressor(hidden_layer_sizes=(20,20), max_iter=10000)# edw exeis 1 layer me 20 nodes
clf = clf.fit(X, y)

MAE = 0
for i in range(10):
    pred = clf.predict(Xtest)
    testingError = [(t - p) for (t, p) in zip(ytest, pred)]
    MAE = MAE + np.mean(np.abs(testingError))
result = MAE/10
print(result)

X = trainingSet.loc[:, ["Size", "Rooms", "Owners"]]
y = trainingSet.loc[:, "Price"]

Xtest = testingSet.loc[:, ["Size", "Rooms", "Owners"]]
ytest = testingSet.loc[:, "Price"]

from sklearn.neighbors import KNeighborsClassifier
for i in range(16,21):
    clf = KNeighborsClassifier(n_neighbors=i)
    clf = clf.fit(X, y)
    pred = clf.predict(Xtest)
    print("F1 Score: ", f1_score(ytest, pred, pos_label='Low'))

from sklearn import svm

accuracies = []
from sklearn.model_selection import cross_val_score
gammavalues = [0.1, 0.01,0.001,0.0001]
for gamma in gammavalues:
    clf = svm.SVC(kernel="rbf", gamma=gamma)
    scores = cross_val_score(clf, Xtest, ytest, cv=7)
    accuracies.append(scores.mean())

print("Best gamma: ", gammavalues[np.argmax(accuracies)])
