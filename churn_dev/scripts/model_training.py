from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

'''
def split_data(df):
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    return train_test_split(X, y, test_size=0.2, random_state=42)
'''

def split_data(X, y, test_size=0.2, random_state=42):
    if y.dtype == 'object':
        y = y.map({'Yes': 1, 'No': 0})
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_and_tune_models(X_train, y_train):
    models = {
        "LogisticRegression": (LogisticRegression(max_iter=1000), {
            'C': [0.01, 0.1, 1, 10]
        }),
        "RandomForest": (RandomForestClassifier(), {
            'n_estimators': [50,100,150],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }),
        "GradientBoosting": (GradientBoostingClassifier(), {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1]
        }),
        "XGBoost": (XGBClassifier(eval_metric='logloss'), {
            'n_estimators': [100, 150],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1],
            'colsample_bytree': [0.8, 1]
        }),
        "KNeighbors": (KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 7]
        }),
        "DecisionTree": (DecisionTreeClassifier(), {
            'max_depth': [5, 10, None]
        }),
        "NaiveBayes": (GaussianNB(), {})
    }

    trained_models = {}
    for name, (model, params) in models.items():
        if params:
            grid = GridSearchCV(model, params, cv=3)
            grid.fit(X_train, y_train)
            trained_models[name] = grid.best_estimator_
        else:
            model.fit(X_train, y_train)
            trained_models[name] = model

    return trained_models
