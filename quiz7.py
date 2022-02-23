from os import sep
import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans

kdata = pd.read_csv("./quiz_data7.csv")
print(kdata)
plt.scatter(kdata.X1, kdata.X2)
plt.show()


cluster_centers = np.array([[-4, 10],[0, 0], [4, 10]])

kmeans = KMeans(n_clusters=3, init=cluster_centers).fit(kdata)
print(kmeans.cluster_centers_)
print(kmeans.labels_)

# Cohesion
print("Cohesion :", kmeans.inertia_)

# Separation
separation = 0
distance = lambda x1, x2: math.sqrt(((x1.X1 - x2.X1) ** 2) + ((x1.X2 - x2.X2) ** 2))
m = kdata.mean()
for i in list(set(kmeans.labels_)):
    mi = kdata.loc[kmeans.labels_ == i, :].mean()
    Ci = len(kdata.loc[kmeans.labels_ == i, :].index)
    separation += Ci * (distance(m, mi) ** 2)
print("Separation: ",separation)


# Silhouette
from sklearn.metrics import silhouette_samples, silhouette_score
print("Silhoutte: " ,silhouette_score(kdata, kmeans.labels_))

# Plot the clustered data
plt.scatter(kdata.X1, kdata.X2, c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="+", s=169, c=range(3))
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()




# Repeat for different initial conditions
cluster_centers = np.array([[-2, 0],[2, 0], [0, 10]])

kmeans = KMeans(n_clusters=3, init=cluster_centers).fit(kdata)
print(kmeans.cluster_centers_)
print(kmeans.labels_)
print("Cohesion :",kmeans.inertia_)
separation = 0
distance = lambda x1, x2: math.sqrt(((x1.X1 - x2.X1) ** 2) + ((x1.X2 - x2.X2) ** 2))
m = kdata.mean()
for i in list(set(kmeans.labels_)):
    mi = kdata.loc[kmeans.labels_ == i, :].mean()
    Ci = len(kdata.loc[kmeans.labels_ == i, :].index)
    separation += Ci * (distance(m, mi) ** 2)
print("Separation: ",separation)

print("Silhoutte: ", silhouette_score(kdata, kmeans.labels_))

plt.scatter(kdata.X1, kdata.X2, c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="+", s=169, c=range(3))
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()