from turtle import end_fill, shape
from numpy.core.numeric import Inf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, jaccard_score, silhouette_samples
from sklearn.cluster import DBSCAN
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import math
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import seaborn as sns



# Import data
data1 = pd.read_csv("./pro_data.csv")

# Use Acousticness, Energy and Liveness
data5 = data1[['Acousticness', 'Energy', 'Liveness']]
print(data5.head())

scaler = StandardScaler()
scaler = scaler.fit(data5)
data5n = pd.DataFrame(scaler.transform(data5), columns=data5.columns, index=data5.index)

# Hierarchical clustering
clustering = AgglomerativeClustering(n_clusters=5, linkage="complete").fit(data5n)
# Silhouette score
print("Silhouette Score: ",silhouette_score(data5n, clustering.labels_))
# Silhouette score of the first cluster
print("Silhouette score of first cluster: ",np.mean(silhouette_samples(data5n, clustering.labels_)[clustering.labels_==0]))


### DBSCAN
data6 = data1[['Valence', 'Danceability']]
print(data6.head())

clustering = DBSCAN(eps=0.5, min_samples=100).fit(data6)
clusters = clustering.labels_

plt.scatter(data6.Valence, data6.Danceability, c=clusters, cmap="spring")
plt.title("DBSCAN(eps=0.5, minPts=100)")
plt.show()

# 2 
data7 = data1[['Valence', 'Danceability', 'Liveness']]

# sil = []
# j = 0
# epss = np.linspace(0.01,1,100)
# for i in epss:
#     clustering = DBSCAN(eps=i, min_samples=15).fit(data7)
#     clusters = clustering.labels_
#     sil[j] = silhouette_score(data7, clustering.labels_)
#     j = j+1

# j_max = sil.index(max(sil))
# print("Best eps: ", epss[j_max])

### K-Means
bdata = data1.loc[:, ["Danceability", "Instrumentalness", "Speechiness"]]

sse = []
for i in range(1, 11):
    sse.append(
        KMeans(n_clusters=i, init=bdata.loc[0:i-1, :]).fit(bdata).inertia_)
plt.plot(range(1, 11), sse)
plt.scatter(range(1, 11), sse, marker="o")
plt.show()