from numpy.lib.function_base import cov
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from scipy.sparse import data
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import CategoricalNB, GaussianNB
import numpy as np
from sklearn.neural_network import MLPRegressor
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA

covidData = pd.read_csv("./covid-data.csv")
covidData = covidData.dropna()
print(covidData)

x = covidData.loc[:, ["Fever", "Tiredness", "DryCough", "DifficultyBreathing", "SoreThroat", "RunnyNose", "Diarrhea", "NoneSymptoms", "Age", "Gender" ]]
y = covidData.loc[:, "Severity"]

xtrain = covidData.loc[0:8000, ["Fever", "Tiredness", "DryCough", "DifficultyBreathing", "SoreThroat", "RunnyNose", "Diarrhea", "NoneSymptoms", "Age", "Gender" ]]
ytrain = covidData.loc[0:8000, "Severity"]

xtest = covidData.loc[8000:9999, ["Fever", "Tiredness", "DryCough", "DifficultyBreathing", "SoreThroat", "RunnyNose", "Diarrhea", "NoneSymptoms", "Age", "Gender" ]]
ytest = covidData.loc[8000:9999, "Severity"]

# encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
# encoder = encoder.fit(x)
# x = encoder.transform(x)

# encoder2 = OneHotEncoder(handle_unknown="ignore", sparse=False)
# encoder2 = encoder2.fit(y)
# y = encoder2.transform(y)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le = le.fit(y)
y = le.transform(y)
# # feature extraction
# pca = PCA(n_components=3)
# fit = pca.fit(x)
# print("Explained Variance: %s" % fit.explained_variance_ratio_)
# print(fit.components_)


# encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
# encoder.fit(covidData.loc[:, ['NoneSymptoms']])
# transformedData = encoder.transform(covidData.loc[:, ['NoneSymptoms']])
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(transformedData, covidData.loc[:, 'Severity'])
# fig = plt.figure()
# tree.plot_tree(clf, class_names=['Low', 'High'], filled=True)
# plt.show()
# print(covidData)

# absfreq = pd.crosstab(covidData.NoneSymptoms, covidData.Severity)
# freq = pd.crosstab(covidData.NoneSymptoms, covidData.Severity, normalize='index')
# freqSum = pd.crosstab(covidData.NoneSymptoms, covidData.Severity, normalize='all').sum(axis=1)
# print(absfreq)

# GINI_Yes = 1 - freq.loc["Yes", "Low"]**2 - freq.loc["Yes", "High"]**2
# GINI_No = 1 - freq.loc["No", "Low"]**2 - freq.loc["No", "High"]**2
# GINI_NoneSymptoms = freqSum.loc["Yes"] * GINI_Yes + freqSum["No"] * GINI_No
# print(GINI_NoneSymptoms)


clf = tree.DecisionTreeClassifier(min_impurity_decrease=0)
clf = clf.fit(xtrain, ytrain)

pred = clf.predict(xtest)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

print(f'Confusion Matrix: {confusion_matrix(ytest, pred)}')
print(f'Accuracy: {accuracy_score(ytest, pred)}')
print(f'Precision: {precision_score(ytest, pred, pos_label=1)}')
print(f'Recall: {recall_score(ytest, pred,pos_label=1)}')
print(f'F1 Score: {f1_score(ytest, pred, pos_label=1)}')

