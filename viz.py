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


## Import data
data1 = pd.read_csv("./pro_data.csv")

## Take a peek
print(data1.head())

## Visualizing in 1 dimension
data1.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
           xlabelsize=8, ylabelsize=8, grid=False)    
plt.show()

## Visualizing in pdf
sns.kdeplot(data1['Acousticness'], shade=True, color='steelblue')
plt.show()

## Correlation Matrix Heatmap
f, ax = plt.subplots()
corr = data1.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
# f.subplots_adjust(top=0.93)
t= f.suptitle('Correlation Heatmap', fontsize=14)
plt.show()

## Pair-wise Scatter Plots
cols = data1.columns
print(cols)
pp = sns.pairplot(data1[cols], height=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))

fig = pp.fig 
# fig.subplots_adjust(top=0.93, wspace=0.3)
plt.show()

## Scatter Plot
plt.scatter(data1['Acousticness'], data1['Speechiness'],
            alpha=0.4, edgecolors='w')

plt.xlabel('Acousticness')
plt.ylabel('Speechiness')
plt.show()