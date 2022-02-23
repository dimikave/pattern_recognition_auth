import pandas as pd
from scipy.sparse import data
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import CategoricalNB, GaussianNB
import numpy as np

data = pd.read_csv("./quiz_data2.csv")
X = data.loc[:, ["P_M1", "P_M2"]]
y = data.loc[:, "Class"]


# encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
# encoder = encoder.fit(X)
# X = encoder.transform(X)
predprob = np.array(X)
print(predprob)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc


fpr1, tpr1, thresholds1 = roc_curve(y, predprob[:, 0])
print(tpr1,thresholds1)
fpr2, tpr2, thresholds2 = roc_curve(y, predprob[:, 1])
print("AUC: ", auc(fpr1, tpr1))
print(predprob)
# print("TPR: ", tpr)
# pred1 = []
# for i in predprob[:,1]:
#     print(i)
#     if i<0.5:
#         pred1.append(0)
#     else:
#         pred1.append(1)

pred1 = np.where(predprob[:,0] > 0.5, 1, 0)
print(pred1)
print(f'Recall: {recall_score(y, pred1,pos_label=1)}')# REMEMBER!           Recall == TPR

pred2 = np.where(predprob[:,1] > 0.5, 1, 0)
print(pred2)
print("F1 Score: ", f1_score(y, pred2, pos_label=1))

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic - Μ1')
plt.plot(fpr1, tpr1, 'b', label = 'AUC = %0.2f' % auc(fpr1, tpr1))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

plt.title('Receiver Operating Characteristic - Μ2')
plt.plot(fpr2, tpr2, 'b', label = 'AUC = %0.2f' % auc(fpr2, tpr2))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
