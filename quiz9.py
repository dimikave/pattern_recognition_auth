import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Import data
kmdata = pd.read_csv("./kmdata.txt")
X1 = kmdata.loc[:, "X1"]
X2 = kmdata.loc[:, "X2"]
target = kmdata.loc[:, "Y"]
kmdata = kmdata.drop(["Y"], axis=1)

# Όλα με ένα χρώμα
plt.scatter(kmdata.X1, kmdata.X2)
plt.show()

# Με διαφορετικό χρώμα το καθένα
plt.scatter(kmdata[(target == 1)].X1, kmdata[(target == 1)].X2, c="red", marker="+")
plt.scatter(kmdata[(target == 2)].X1, kmdata[(target == 2)].X2, c="green", marker="o")
plt.scatter(kmdata[(target == 3)].X1, kmdata[(target == 3)].X2, c="blue", marker="x")
plt.show()

## K-means
kmeans = KMeans(n_clusters=3).fit(kmdata)

# plt.scatter(kmdata.X1, kmdata.X2, c=kmeans.labels_)
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="+", s=169, c=range(3))
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.show()

## GMM
epsilon = 0.0001
gm = GaussianMixture(n_components=3, tol=epsilon).fit(kmdata)

# Plot with the grid and gaussians
x = np.linspace(np.min(kmdata.loc[:, "X1"]), np.max(kmdata.loc[:, "X1"]))
y = np.linspace(np.min(kmdata.loc[:, "X2"]), np.max(kmdata.loc[:, "X2"]))
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -gm.score_samples(XX)
Z = Z.reshape(X.shape)

plt.contour(X, Y, Z)
plt.scatter(kmdata[(target == 1)].X1, kmdata[(target == 1)].X2, c="red", marker="+")
plt.scatter(kmdata[(target == 2)].X1, kmdata[(target == 2)].X2, c="green", marker="o")
plt.scatter(kmdata[(target == 3)].X1, kmdata[(target == 3)].X2, c="blue", marker="x")
plt.show()

# Clustering
clusters = gm.predict(kmdata)
centers = gm.means_

# plt.scatter(kmdata.X1, kmdata.X2, c=clusters)
# plt.scatter(centers[:, 0], centers[:, 1], marker="+", s=169, c=range(3))
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.show()

fig, axs = plt.subplots(1,2)
fig.suptitle('Clustering of data : K-Means vs GMM')
axs[0].scatter(kmdata.X1, kmdata.X2, c=kmeans.labels_)
axs[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="+", s=169, c=range(3))
axs[0].set(xlabel='X1', ylabel='X2',title='K-Means Clustering')

axs[1].scatter(kmdata.X1, kmdata.X2, c=clusters)
axs[1].scatter(centers[:, 0], centers[:, 1], marker="+", s=169, c=range(3))
axs[1].set(xlabel='X1', ylabel='X2',title='GMM Clustering')
plt.show()