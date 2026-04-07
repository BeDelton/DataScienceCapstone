import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("Data/fraud_oracle.csv")

print(df.columns.tolist())

target = "FraudFound_P"


X = df.drop(columns=['FraudFound_P', 'PolicyNumber', 'RepNumber'])
y = df[target]

X = pd.get_dummies(X, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

importance = pd.Series(model.feature_importances_, index=X.columns)

top_features = importance.sort_values(ascending=False)
print("Top Features:", top_features.head(15))

least_features = top_features.sort_values(ascending=True)
print("Least Important Features:", least_features.head(15))
