from sklearn.svm import SVR
from .model_evaluation import evaluate_model

def train_svr(X_train, X_test, y_train, y_test):
    model = SVR(kernel="rbf")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return evaluate_model(y_test, y_pred, "Support Vector Regression")