from turtle import end_fill
from numpy.core.numeric import Inf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, jaccard_score
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



def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)


# Import data
data1 = pd.read_csv("./data.csv")

# # Ti morfh exoun
# print(data1.head())

# Keep only the employees with low salary
data2 = data1[data1['salary']=='low']
print(data1.head())
# Satisfaction > 0.5 ?
print(data1[["satisfaction"]].mean())
# More than 20 people promoted?
print(data1['promotion'].value_counts())
# Most of people in management department?
print(data1['department'].value_counts())

# Keep only those with satisfaction < 0.6 in the department of sales
data3 = data1[(data1['satisfaction']<0.6) & (data1['department']=='sales')]
print(data3.head())


# GINI Index for accident
absfreq = pd.crosstab(data3.accident, data3.salary)
freq = pd.crosstab(data3.accident, data3.salary, normalize='index')
freqSum = pd.crosstab(data3.accident, data3.salary, normalize='all').sum(axis=1)
print(absfreq)

GINI_Yes = 1 - freq.loc["Yes", "high"]**2 - freq.loc["Yes", "low"]**2 - freq.loc["Yes", "medium"]**2
GINI_No = 1 - freq.loc["No", "high"]**2 - freq.loc["No", "low"]**2 - freq.loc["No", "medium"]**2
GINI_accident = freqSum.loc["Yes"] * GINI_Yes + freqSum["No"] * GINI_No
print(GINI_accident)

# Keep only the employees that haven't been promoted
data4 = data1[(data1['promotion']=='No')]
print(data4.head())

### Fitting a desicion tree using left, promotion, department
# Encoding data
df2, targets = encode_target(data1, "salary")
# print("* df2.head()", df2[["Target", "salary"]].head(),
#       sep="\n", end="\n\n")
# print("* df2.tail()", df2[["Target", "salary"]].tail(),
#       sep="\n", end="\n\n")
# print("* targets", targets, sep="\n", end="\n\n")

features = ['left', 'promotion', 'department']
y = df2["Target"]
X = df2[features]
# Encoding
encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
encoder.fit(X.loc[:, features])
mod_X = encoder.transform(X.loc[:, features])
# Classifier
dt = DecisionTreeClassifier()
dt.fit(mod_X,y)
# Probability 'salary' = 'low' when 'left' = 'No', 'promotion' = 'No', 'department' = 'sales'
probs = dt.predict_proba(mod_X[(X['left']=='No') & (X['promotion']=='No') &(X['department']=='sales')])
print(probs)
sal = dt.predict(mod_X[(X['left']=='No') & (X['promotion']=='No') &(X['department']=='sales')])


#### PCA
# Keeping only employees that got promoted and specific columns
data4 = data1[(data1['promotion']=='Yes')]
data4 = data4[['satisfaction', 'evaluation', 'projects', 'hours', 'history']]

# Scaling
scaler = StandardScaler()
scaler = scaler.fit(data4)
transformed = pd.DataFrame(scaler.transform(data4), columns=["satisfaction", "evaluation", "projects", "hours", "history"])
pca = PCA()
pca = pca.fit(transformed)
pca_transformed = pca.transform(transformed)
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_
print(eigenvalues)

# Infoloss, check how many components to have info loss < 0.5
info_loss = (eigenvalues[2] + eigenvalues[3] + eigenvalues[4]) / sum(eigenvalues)
print("Info Loss: ", info_loss)
# PCs
plt.bar(range(len(eigenvalues)), eigenvalues/sum(eigenvalues))
plt.show()

### Silhouette
data5 = data1[(data1['left']=='No')]
data5 = data5[['satisfaction', 'evaluation', 'projects']]

kmeans = KMeans(n_clusters=3, init=data5.loc[0:2, :]).fit(data5)
# kmeans = KMeans(n_clusters=3, init=data5.iloc[4:14, :]).fit(data5)

# print(data5.loc[4:14, :])
# print(kmeans.cluster_centers_)
# print(kmeans.labels_)
# print(kmeans.inertia_)

from sklearn.metrics import silhouette_samples, silhouette_score
print("Silhouette Score: ",silhouette_score(data5, kmeans.labels_))


data6 = data1[(data1['salary']=='medium')]
data6 = data6[['satisfaction', 'hours', 'history']]

minPts = [51, 41, 50, 30, 31, 40]

for i in range(len(minPts)):
    clustering = DBSCAN(eps=0.01, min_samples=minPts[i]).fit(data6)
    clusters = clustering.labels_
    # print(len(set(clusters)))
    if len(set(clusters))==7:
        print(minPts[i])
