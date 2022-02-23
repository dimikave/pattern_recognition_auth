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

P1 = [0,1,0,0,0,0,1,0]
P2 = [1,1,0,0,0,1,0,0]

# Jaccard index
print(jaccard_score(P1,P2))

# SMC score
def SiMoCa(x,y):
    p00 = 0
    p01 = 0
    p10 = 0
    p11 = 0
    for (i,j) in zip(x,y):
        if (i == 0 and j == 0):
            p00 += 1
        elif (i == 0 and j == 1):
            p01 += 1
        elif (i == 1 and j == 0):
            p01 += 1
        else:
            p11 += 1
    smc = (p00+p11)/(p00+p11+p01+p10)
    jac = (p11)/(p11+p01+p10)
    print("Simple Matchin Coefficient - SMC: ",smc)
    print("Jaccard Index: ",jac)
    return smc,jac

SiMoCa(P1,P2)



