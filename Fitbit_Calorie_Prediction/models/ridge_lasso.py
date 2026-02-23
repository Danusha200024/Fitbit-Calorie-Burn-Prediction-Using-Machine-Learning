from sklearn.linear_model import Ridge, Lasso
from .model_evaluation import evaluate_model

def train_ridge(X_train, X_test, y_train, y_test):
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return evaluate_model(y_test, y_pred, "Ridge Regression")

def train_lasso(X_train, X_test, y_train, y_test):
    model = Lasso(alpha=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return evaluate_model(y_test, y_pred, "Lasso Regression")