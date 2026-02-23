from sklearn.neighbors import KNeighborsRegressor
from .model_evaluation import evaluate_model

def train_knn(X_train, X_test, y_train, y_test):
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return evaluate_model(y_test, y_pred, "KNN Regressor")