import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("housing.csv")

print(df.describe())
print(df.corr())

sns.heatmap(df.corr(), annot=False)
plt.show()
