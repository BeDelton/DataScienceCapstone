import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier

import xgboost as xgb


df = pd.read_csv("Data/fraud_oracle.csv")

y = df['FraudFound_P']
X = df.drop(columns=['FraudFound_P', 'PolicyNumber', 'RepNumber', 'Year', 'Sex', 'Days_Policy_Claim', 'DriverRating', 'MaritalStatus', 'Make'])

results = []

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols     = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results.append({
        "Model":     name,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1 Score":  f1_score(y_test, y_pred),
        "ROC-AUC":   roc_auc_score(y_test, y_prob)
    })


smote = SMOTE(random_state=42)
smoteenn = SMOTEENN(random_state=42)

#Maybe make this a string instead of 2 parameters**
def make_pipeline(classifier, use_smote=False, use_smoteenn=False):
    steps = [('prep', preprocessor)]
    if use_smote:
        steps.append(('smote', smote))
    elif use_smoteenn:
        steps.append(('smoteenn', smoteenn))
    steps.append(('model', classifier))
    return Pipeline(steps)


classifiers = {
    "Logistic Regression":LogisticRegression(max_iter=1000),
    "Decision Tree":DecisionTreeClassifier(random_state=42),
    "Random Forest":RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    ),
}

for name, clf in classifiers.items():
    #Baseline
    model = make_pipeline(clf, use_smote=False)
    model.fit(X_train, y_train)
    evaluate_model(name, model, X_test, y_test)
    print(name + " Finished")

    #SMOTE version
    clf_fresh = type(clf)(**clf.get_params())
    model_smote = make_pipeline(clf_fresh, use_smote=True)
    model_smote.fit(X_train, y_train)
    evaluate_model(name + " + SMOTE", model_smote, X_test, y_test)
    print(name + " + SMOTE Finished")

    #SMOTEENN version
    clf_fresh2 = type(clf)(**clf.get_params())
    model_smoteenn = make_pipeline(clf_fresh2, use_smoteenn=True)
    model_smoteenn.fit(X_train, y_train)
    evaluate_model(name + " + SMOTEENN", model_smoteenn, X_test, y_test)
    print(name + " + SMOTEENN Finished")


brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
brf_pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', brf)
])

brf_pipeline.fit(X_train, y_train)
evaluate_model("Balanced Random Forest", brf_pipeline, X_test, y_test)
print("Balanced Random Forest")


results_df = pd.DataFrame(results).round(3)
results_df = results_df.set_index("Model")
print("\n", results_df.to_string())