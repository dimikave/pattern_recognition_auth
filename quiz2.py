import pandas as pd
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

carMarket = pd.read_csv("./quiz_data.csv")

encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

# encoder.fit(carMarket.loc[:, ['Sex', 'CarType',  'Budget']])
# transformedCarType = encoder.transform(carMarket.loc[:, ['Sex', 'CarType', 'Budget']])
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(transformedCarType, carMarket.loc[:, 'Insurance'])
# fig = plt.figure(figsize=(10,9))
# tree.plot_tree(clf, class_names=['No', 'Yes'], filled=True)
# plt.show()



# encoder.fit(carMarket.loc[:, ['CarType']])
# transformedCarType = encoder.transform(carMarket.loc[:, ['CarType']])
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(transformedCarType, carMarket.loc[:, 'Insurance'])
# fig = plt.figure()
# tree.plot_tree(clf, class_names=['No', 'Yes'], filled=True)
# plt.show()


# absfreq = pd.crosstab(carMarket.CarType, carMarket.Insurance)
# freq = pd.crosstab(carMarket.CarType, carMarket.Insurance, normalize='index')
# freqSum = pd.crosstab(carMarket.CarType, carMarket.Insurance, normalize='all').sum(axis=1)
# print(absfreq)

# GINI_Family = 1 - freq.loc["Family", "No"]**2 - freq.loc["Family", "Yes"]**2
# GINI_Sport = 1 - freq.loc["Sport", "No"]**2 - freq.loc["Sport", "Yes"]**2
# GINI_Sedan = 1 -freq.loc["Sedan", "No"]**2 - freq.loc["Sedan", "Yes"]**2
# GINI_CarType = freqSum.loc["Family"] * GINI_Family + freqSum["Sport"] * GINI_Sport + freqSum["Sedan"] * GINI_Sedan
# print(GINI_CarType)

# absfreq = pd.crosstab(carMarket.Sex, carMarket.Insurance)
# freq = pd.crosstab(carMarket.Sex, carMarket.Insurance, normalize='index')
# freqSum = pd.crosstab(carMarket.Sex, carMarket.Insurance, normalize='all').sum(axis=1)
# print(absfreq)

# GINI_Male = 1 - freq.loc["M", "No"]**2 - freq.loc["M", "Yes"]**2
# GINI_Female = 1 - freq.loc["F", "No"]**2 - freq.loc["F", "Yes"]**2
# GINI_Sex = freqSum.loc["M"] * GINI_Male + freqSum["F"] * GINI_Female
# print(GINI_Male)

# encoder.fit(carMarket.loc[:, ['Budget']])
# transformedBudget = encoder.transform(carMarket.loc[:, ['Budget']])
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(transformedBudget, carMarket.loc[:, 'Insurance'])
# fig = plt.figure()
# tree.plot_tree(clf, class_names=['No', 'Yes'], filled=True)
# plt.show()


absfreq = pd.crosstab(carMarket.Budget, carMarket.Insurance)
freq = pd.crosstab(carMarket.Budget, carMarket.Insurance, normalize='index')
freqSum = pd.crosstab(carMarket.Budget, carMarket.Insurance, normalize='all').sum(axis=1)
GINI_Low = 1 - freq.loc["Low", "No"]**2 - freq.loc["Low", "Yes"]**2
GINI_Medium = 1 - freq.loc["Medium", "No"]**2 - freq.loc["Medium", "Yes"]**2
GINI_High = 1 -freq.loc["High", "No"]**2 - freq.loc["High", "Yes"]**2
GINI_VeryHigh = 1 -freq.loc["VeryHigh", "No"]**2 - freq.loc["VeryHigh", "Yes"]**2
GINI_Budget = freqSum.loc["Low"] * GINI_Low + freqSum["Medium"] * GINI_Medium + freqSum["High"] * GINI_High + freqSum["VeryHigh"] * GINI_VeryHigh
print(GINI_Budget)