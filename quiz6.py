from numpy.linalg.linalg import eig
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pd.read_csv("./quiz_data6.csv",sep=",")
# print(data.head())

trainingRange = list(range(0,50)) + list(range(90,146))
training = data.loc[trainingRange,:]
trainingType = training.loc[:,"Type"]
training = training.drop(["Type"],axis=1)

testingRange = list(range(50,90))
testing = data.loc[testingRange,:]
testingType = testing.loc[:,"Type"]
testing = testing.drop(["Type"],axis=1)



scaler = StandardScaler()
scaler = scaler.fit(training)
transformed = pd.DataFrame(scaler.transform(training), columns=training.columns)
pca = PCA()
# pca = PCA(n_components=1)
pca = pca.fit(transformed)
pca_transformed = pca.transform(transformed)
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_
plt.bar(range(len(eigenvalues)), eigenvalues/sum(eigenvalues))
plt.show()
# print(pca.components_)
# Ποσοστό πληροφορίας του αρχικού dataset που ενσωματώνει το pc1
print(pca.explained_variance_ratio_[0])

# Ποσοστό απώλειας πληροφορίας αν κρατήσουμε τα πρώτα 4 PC
info_loss = (sum(eigenvalues[4:])) / sum(eigenvalues)
print(info_loss)

from sklearn.metrics import accuracy_score , recall_score
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf = clf.fit(training,trainingType)
pred = clf.predict(testing)
# Accuracy
print(accuracy_score(testingType,pred))
# Recall με θετική κλάση 2
print("Recall: ", recall_score(testingType, pred, pos_label=2))


# Accuracy with different knn models + PCA
acc=[]
for i in range(1,10):
    # Scaler according to training data
    scaler = StandardScaler()
    scaler = scaler.fit(training)
    # Apply scaling at training and testing data
    scaled_training = pd.DataFrame(scaler.transform(training), columns=training.columns)
    scaled_testing = pd.DataFrame(scaler.transform(testing), columns=testing.columns)
    # PCA according to training data
    pca = PCA(n_components=i)
    pca = pca.fit(scaled_training)
    # Apply PCA at training and testing data
    pca_training = pd.DataFrame(pca.transform(scaled_training))
    pca_testing = pd.DataFrame(pca.transform(scaled_testing))
    # Train KNN with training data after PCA
    clf = KNeighborsClassifier(n_neighbors=3)
    clf = clf.fit(pca_training, trainingType)
    # Predict with testing data
    pred = clf.predict(pca_testing)
    acc.append(accuracy_score(testingType,pred))
print("Accuracy:",acc)
print("Index:",acc.index(max(acc))+1)




