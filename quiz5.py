import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# # KNN Part

# X1 = [-2.0, -2.0, -1.8, -1.4, -1.2, 1.2, 1.3, 1.3, 2.0, 2.0, -0.9, -0.5, -0.2, 0.0, 0.0, 0.3, 0.4, 0.5, 0.8, 1.0]
# X2 = [-2.0, 1.0, -1.0, 2.0, 1.2, 1.0, -1.0, 2.0, 0.0, -2.0, 0.0, -1.0, 1.5, 0.0, -0.5, 1.0, 0.0, -1.5, 1.5, 0.0]
# Y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
# alldata = pd.DataFrame({"X1" : X1, "X2" : X2, "Y" : Y})


# X = alldata.loc[:, ["X1", "X2"]]
# y = alldata.Y
# plt.scatter(X[(y == 2)].X1, X[(y == 2)].X2, c="red", marker="+")
# plt.scatter(X[(y == 1)].X1, X[(y == 1)].X2, c="blue", marker="o")
# plt.show()


# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(n_neighbors=3)
# clf = clf.fit(X, y)
# print(clf.predict([[1.5, -0.5]]))
# print(clf.predict_proba([[1.5, -0.5]]))

# clf = KNeighborsClassifier(n_neighbors=5)
# clf = clf.fit(X, y)
# print(clf.predict([[-1.0, 1.0]]))
# print(clf.predict_proba([[-1.0, 1.0]]))


# SVM Part

X1 = [2.0, 2.0, -2.0, -2.0, 1.0, 1.0, -1.0, -1.0]
X2 = [2.0, -2.0, -2.0, 2.0, 1.0, -1.0, -1.0, 1.0]
Y = [1, 1, 1, 1, 2, 2, 2, 2]
alldata = pd.DataFrame({"X1" : X1, "X2" : X2, "Y" : Y})
xx, yy = np.meshgrid(X1, X2)

X = alldata.loc[:, ["X1", "X2"]]
y = alldata.Y
plt.scatter(X[(y == 2)].X1, X[(y == 2)].X2, c="red", marker="+")
plt.scatter(X[(y == 1)].X1, X[(y == 1)].X2, c="blue", marker="o")
plt.show()


from sklearn import svm
from sklearn.metrics import accuracy_score
clf = svm.SVC(kernel="rbf", gamma=1000000)
clf = clf.fit(X, y)
pred = clf.predict(X)

print(accuracy_score(y, pred))

# print(clf.predict([[-2, -1.9]]))
print(clf.predict([[0, 0]]))

pred = clf.predict(np.c_[xx.ravel(), yy.ravel()])
pred = pred.reshape(xx.shape)
plt.contour(xx, yy, pred, colors="blue")
plt.show()
# pred = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# pred = pred.reshape(xx.shape)
# plt.contour(xx, yy, pred, colors="blue")
