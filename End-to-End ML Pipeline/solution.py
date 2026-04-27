import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("train.csv")

X = df.drop("Transported", axis=1)
y = df["Transported"]

X = pd.get_dummies(X)
X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier())
])

pipeline.fit(X_train, y_train)

print("Accuracy:", pipeline.score(X_test, y_test))
