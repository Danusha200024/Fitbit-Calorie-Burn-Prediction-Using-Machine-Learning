from sklearn.tree import DecisionTreeRegressor
from .model_evaluation import evaluate_model

def train_decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return evaluate_model(y_test, y_pred, "Decision Tree")