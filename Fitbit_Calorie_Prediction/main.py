# main.py

import pandas as pd

from models.linear_regression import train_linear_regression
from models.ridge_lasso import train_ridge, train_lasso
from models.knn_regressor import train_knn
from models.decision_tree import train_decision_tree
from models.random_forest import train_random_forest
from models.svr_model import train_svr
from models.xgboost_model import train_xgboost

# Load processed data
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# Run all models
train_linear_regression(X_train, X_test, y_train, y_test)
train_ridge(X_train, X_test, y_train, y_test)
train_lasso(X_train, X_test, y_train, y_test)
train_knn(X_train, X_test, y_train, y_test)
train_decision_tree(X_train, X_test, y_train, y_test)
train_random_forest(X_train, X_test, y_train, y_test)
train_svr(X_train, X_test, y_train, y_test)
train_xgboost(X_train, X_test, y_train, y_test)