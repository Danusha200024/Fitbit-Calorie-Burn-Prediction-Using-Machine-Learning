from xgboost import XGBRegressor
from .model_evaluation import evaluate_model

def train_xgboost(X_train, X_test, y_train, y_test):
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return evaluate_model(y_test, y_pred, "XGBoost Regressor")