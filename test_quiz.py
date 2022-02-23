patients = 200
# 50 test thetika, 150 arnhtika
tp = 35
fp = 30
tn = 120
fn = 15


sensitivity = tp/(tp+fn)

specificity = tn/(tn+fp)        # auta ta duo xaraktirizoun to test, den allazoun akoma kai na allaksoun oi astheneis


recall = tp/(tp+fn)
prevalence = (tp+fn)/patients
npv = tn/(tn+fn)
tpr = recall
fnr = fn/(tp+fn)
fpr = fp/(fp+tn)
tnr = tn/(fp+tn)
LRplus = tpr/fpr
LRminus = fnr/tnr
dor = LRplus/LRminus

print(sensitivity)
print(specificity)
print(recall)
print(npv)
print(prevalence)
print(f"LR+ =",LRplus)
print(0.25*LRplus)
print(dor)

# positive Pretest prob
preprob = (tp+fn)/patients
preodds = preprob/(1-preprob)
postodds = preodds*LRplus
postprob = postodds/(postodds+1)
print(postprob)

def ginitest(A):
    pn1c1 = A[0][0]/sum(A[0])
    pn1c2 = A[0][1]/sum(A[0])
    pn2c1 = A[1][0]/sum(A[1])
    pn2c2 = A[1][1]/sum(A[1])
    print(pn2c1)
    gini1 = 1-pow(pn1c1,2)-pow(pn1c2,2)
    gini2 = 1-pow(pn2c1,2)-pow(pn2c2,2)
    gini = sum(A[0])*gini1+sum(A[1])*gini2
    return gini/(sum(A[0])+sum(A[1]))

A = [[5,3],[3,3]]
B = [[7,4],[1,1]]
print(ginitest(B))

