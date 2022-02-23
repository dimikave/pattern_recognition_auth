import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import numpy as np

X1 = [2, 2, -2, -2, 1, 1, -1, -1]
X2 = [2, -2, -2, 2, 1, -1, -1, 1]
Y =  [1, 1, 1, 1, 2, 2, 2, 2]

anndata = pd.DataFrame({"X1" : X1, "X2" : X2, "Y" : Y})
X = anndata.loc[:, ["X1", "X2"]]
y = anndata.loc[:, "Y"]

# Oso gemizeis to tuple, ousiastika les posa nodes thes se kathe layer
# clf = MLPRegressor(hidden_layer_sizes=(2,), max_iter=10000)# px edw exeis 1 layer me 2 nodes
clf = MLPRegressor(hidden_layer_sizes=(20,), max_iter=10000)# edw exeis 1 layer me 20 nodes
# clf = MLPRegressor(hidden_layer_sizes=(20,20), max_iter=10000)# edw exeis 2 layers me 20 nodes
clf = clf.fit(X, y)

pred = clf.predict(X)
trainingError = [(t - p) for (t, p) in zip(Y, pred)]
MAE = np.mean(np.abs(trainingError))
print(MAE)