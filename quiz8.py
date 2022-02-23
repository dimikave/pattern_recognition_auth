from numpy.core.numeric import Inf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score , recall_score

hdata = pd.read_csv("./dcdata.txt")
target = hdata.loc[:,"Y"]
hdata = hdata.drop(["Y"], axis=1)


###### Hiearchichal clustering ###########
# plt.scatter(hdata.X1, hdata.X2)
# plt.show()

# clustering = AgglomerativeClustering(n_clusters=None, linkage="single", distance_threshold=0).fit(hdata)

# Hierarchical clustering using the closest point
clustering = AgglomerativeClustering(n_clusters=2, linkage="single").fit(hdata)
plt.scatter(hdata.X1, hdata.X2, c=clustering.labels_, cmap="bwr")
plt.show()
print(accuracy_score(target,clustering.labels_))

# Hierarchical clustering using the furthest point
clustering = AgglomerativeClustering(n_clusters=2, linkage="complete").fit(hdata)
plt.scatter(hdata.X1, hdata.X2, c=clustering.labels_, cmap="bwr")
plt.show()
print(accuracy_score(target,clustering.labels_))

# Hierarchical clustering using the centroid
clustering = AgglomerativeClustering(n_clusters=2, linkage="average").fit(hdata)
plt.scatter(hdata.X1, hdata.X2, c=clustering.labels_, cmap="bwr")
plt.show()
print(accuracy_score(target,clustering.labels_))

###### DBSCAN ###########
clustering = DBSCAN(eps=0.75, min_samples=5).fit(hdata)
# clustering = DBSCAN(eps=1.00, min_samples=5).fit(hdata)
# clustering = DBSCAN(eps=1.25, min_samples=5).fit(hdata)
# clustering = DBSCAN(eps=1.50, min_samples=5).fit(hdata)


clusters = clustering.labels_
plt.scatter(hdata.X1, hdata.X2, c=clusters, cmap="spring")
plt.title("DBSCAN(eps=2, minPts=5)")
plt.show()

####### K-means
kmeans = KMeans(n_clusters=2).fit(hdata)
plt.scatter(hdata.X1, hdata.X2, c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="+", s=169, c=range(2))
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()