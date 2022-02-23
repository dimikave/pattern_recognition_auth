from sklearn import datasets
import statistics
import pandas as pd

iris = datasets.load_iris()
featureIndex = iris.feature_names.index("petal length (cm)")
data = iris.data
# print(data)
mo = statistics.mean(data[featureIndex])
# print(statistics.mean())
df = pd.DataFrame(data, columns=iris.feature_names)
# df.head()
print(df[["petal length (cm)"]].mean())
print(df[["sepal length (cm)"]].var())
print(df[["sepal width (cm)"]].max())

print(df.mean())      # mesos oros kathe sthlhs
# print(df.mean(data[featureIndex]))
# mo = df["petal length (cm)"].mean()
# print(df[["petal length (cm)"]])