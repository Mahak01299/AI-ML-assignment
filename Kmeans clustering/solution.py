import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv("Mall_Customers.csv")

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

model = KMeans(n_clusters=5)
df["Cluster"] = model.fit_predict(X)

print(df.head())
