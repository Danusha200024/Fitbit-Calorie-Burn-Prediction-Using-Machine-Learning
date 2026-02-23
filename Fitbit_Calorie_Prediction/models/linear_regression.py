from sklearn.linear_model import LinearRegression
from .model_evaluation import evaluate_model

def train_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return evaluate_model(y_test, y_pred, "Linear Regression")