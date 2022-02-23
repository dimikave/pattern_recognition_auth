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
from sklearn.decomposition import PCA
import math
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from statistics import mean
from sklearn.mixture import GaussianMixture
import seaborn as sns

# Import data
wdata = pd.read_csv("./wdata.csv")

# Peek
print(wdata.head())

##### 1 - Scaling & PCA, Information Loss
wdata1 = wdata[['Exports', 'Health', 'Income', 'Age']]
print(wdata1.head())

# Scaling & PCA
scaler = StandardScaler()
scaler = scaler.fit(wdata1)
transformed = pd.DataFrame(scaler.transform(wdata1), columns=["Exports", "Health", "Income", "Age"])
pca = PCA()
pca = pca.fit(transformed)
pca_transformed = pca.transform(transformed)
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_
print(eigenvalues)

# Infoloss
info_loss = eigenvalues[3] / sum(eigenvalues)
print("Info Loss: ", info_loss)
# PCs
plt.bar(range(len(eigenvalues)), eigenvalues/sum(eigenvalues))
plt.show()

##### 2 - Silhouette of 4th cluster with K-Means
wdata2 = wdata[['Exports', 'Imports', 'Income', 'Age']]
kmeans = KMeans(n_clusters=5, init=wdata2.loc[0:4, :]).fit(wdata2)
print("Silhouette score of fourth cluster: ",np.mean(silhouette_samples(wdata2, kmeans.labels_)[kmeans.labels_==3]))

##### 3 - K-Means best k
wdata3 = wdata[['ChildMortality', 'Exports', 'Income']]
sse = []
sil = []
for i in range(5, 10):
    kmeansi = KMeans(n_clusters=i, init=wdata3.loc[0:i-1, :]).fit(wdata3)
    sse.append(kmeansi.inertia_)
    sil.append(silhouette_score(wdata3, kmeansi.labels_))
    print("Silhouette Score: ",silhouette_score(wdata3, kmeansi.labels_))
    print(i)

# Plot of SSE to check the elbow for verification
plt.plot(range(5, 10), sse)
plt.scatter(range(5, 10), sse, marker="o")
plt.show()

##### 4 - Single Linkage Hierarchical Clustering
wdata4 = wdata[['ChildMortality', 'Health', 'Age']]
clustering = AgglomerativeClustering(n_clusters=5, linkage="single").fit(wdata4)
print("Silhouette Score - Single Linkage Hierarchical Clustering: ",silhouette_score(wdata4,clustering.labels_))

##### 5 - DBSCAN
wdata5 = wdata[['ChildMortality', 'Imports', 'Age']]
clustering = DBSCAN(eps=2, min_samples=2).fit(wdata5)
clusters = clustering.labels_
print("Silhouette Score - DBSCAN: ",silhouette_score(wdata5,clustering.labels_))
