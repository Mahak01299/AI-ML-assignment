import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split

df = pd.read_csv("Kickstarter.csv")

df = df.select_dtypes(include='number').dropna()

X = df.drop("goal", axis=1)
y = df["goal"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)

ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

print("Ridge Score:", ridge.score(X_test, y_test))
print("Lasso Score:", lasso.score(X_test, y_test))
