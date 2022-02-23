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
p2data = pd.read_csv("./p2data.csv")
country = p2data.Country
p2data1 = p2data.drop(["Country"], axis=1)

# # Peek
# print(p2data.head())

# ###### Visuallization
# ## Visualizing in 1 dimension
# p2data1.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
#            xlabelsize=8, ylabelsize=8, grid=False)    
# plt.show()

# ## Correlation Matrix Heatmap
# f, ax = plt.subplots()
# corr = p2data1.corr()
# hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
#                  linewidths=.05)
# # f.subplots_adjust(top=0.93)
# t= f.suptitle('Correlation Heatmap', fontsize=14)
# plt.show()

# ## Pair-wise Scatter Plots
# cols = p2data1.columns
# print(cols)
# pp = sns.pairplot(p2data1[cols], height=1.8, aspect=1.8,
#                   plot_kws=dict(edgecolor="k", linewidth=0.5),
#                   diag_kind="kde", diag_kws=dict(shade=True))

# fig = pp.fig 
# # fig.subplots_adjust(top=0.93, wspace=0.3)
# plt.show()




#### 1 - Scaling & PCA, Information Loss
# p2data1 = p2data[['Population', 'Area', 'PopulationDensity',  'Coastline',  'Migration',  'InfantMortality', 'GDP', 'Literacy','Phones', 'Birthrate', 'Deathrate']]
# print(p2data1.head())

# Scaling & PCA
scaler = StandardScaler()
scaler = scaler.fit(p2data1)
transformed = pd.DataFrame(scaler.transform(p2data1), columns=["Population", "Area", "PopulationDensity",  "Coastline",  "Migration",  "InfantMortality", "GDP", "Literacy","Phones", "Birthrate", "Deathrate"])
pca = PCA()
pca = pca.fit(transformed)
pca_transformed = pca.transform(transformed)
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_
print(eigenvalues)
plt.bar(range(len(eigenvalues)), eigenvalues/sum(eigenvalues))
plt.show()


info_loss = sum(eigenvalues[1:]) / sum(eigenvalues)
print("Info Loss: ", info_loss)

##### K-Means and evaluation with Silhouette Score
sil = []
for i in range(1,20):
    # Scaler according to training data
    scaler = StandardScaler()
    scaler = scaler.fit(p2data1)
    # Apply scaling at training and testing data
    scaled = pd.DataFrame(scaler.transform(p2data1), columns=p2data1.columns)
    # PCA according to training data
    pca = PCA(n_components=5)
    pca = pca.fit(scaled)
    # Apply PCA at training and testing data
    pca_scaled = pd.DataFrame(pca.transform(scaled))
    # K-Means with training data after PCA
    kmeans = KMeans(n_clusters=i, init=scaled.loc[0:i-1, :]).fit(scaled)
    sil.append(silhouette_score(scaled, kmeans.labels_))
print("Max Silhouette with K-Means: ",max(sil))
print("Best k:",sil.index(max(sil))+1)

# ##### DBSCAN and evaluation with Silhouette Score
# sil = []
# epsilon = np.linspace(0.01,3,300)
# for i in range(1,20):
#     # Scaler according to training data
#     scaler = StandardScaler()
#     scaler = scaler.fit(p2data1)
#     # Apply scaling at training and testing data
#     scaled = pd.DataFrame(scaler.transform(p2data1), columns=p2data1.columns)
#     # PCA according to training data
#     pca = PCA(n_components=5)
#     pca = pca.fit(scaled)
#     # Apply PCA at training and testing data
#     pca_scaled = pd.DataFrame(pca.transform(scaled))
#     # K-Means with training data after PCA
#     clustering = DBSCAN(eps=epsilon, min_samples=7).fit(p2data1)
#     clusters = clustering.labels_
#     sil.append(silhouette_score(scaled, clusters))
# print("Max Silhouette with DBSCAN: ",max(sil))
# print("Best epsilon:",sil.index(max(sil))+1)


