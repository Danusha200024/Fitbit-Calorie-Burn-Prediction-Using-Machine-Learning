🏃 Fitbit Calorie Burn Prediction System
📌 Overview

This project develops a machine learning-based regression system to accurately predict calories burned (kcal) during workouts using physiological and workout-related features.

Multiple regression models were trained, compared, and evaluated to determine the most accurate and robust approach suitable for real-world fitness tracking applications.

🎯 Objective

Predict Calories_Burned (continuous numeric target)

Compare multiple supervised regression models

Identify the best-performing algorithm

Generate actionable business insights for wearable fitness systems

📊 Dataset Summary

Total Records: 14,102

Clean dataset (no missing values)

Mix of physiological and workout-related features

Key Features

Age

Gender

Weight (kg)

Height (m)

Max_BPM

Avg_BPM

Resting_BPM

Session_Duration (hours)

Workout_Type

Fat_Percentage

Water_Intake (liters)

Workout_Frequency (days/week)

Experience_Level

BMI

Base_MET

HR_Intensity

Effective_MET

🎯 Target Variable:

Calories_Burned (kcal)
🔄 Data Preprocessing Steps

Removed unnecessary columns

Verified absence of missing values

Applied one-hot encoding

Split dataset (80% training / 20% testing)

Applied StandardScaler

Prevented data leakage

🤖 Models Implemented

The following regression models were trained and evaluated:

Linear Regression

Ridge Regression

Lasso Regression

K-Nearest Neighbors (KNN)

Decision Tree Regressor

Random Forest Regressor

Support Vector Regression (SVR)

XGBoost Regressor

📈 Model Performance
Model	R² Score
Linear Regression	0.911
Ridge Regression	0.911
Lasso Regression	0.911
KNN	0.943
Decision Tree	0.994
Random Forest	0.998
SVR	0.878
XGBoost	0.999
🏆 Best Model
✅ XGBoost Regressor

MAE: 3.511

RMSE: 5.625

R² Score: 0.999

This model explains approximately 99.9% of variance in calorie burn prediction, making it the most accurate model in this study.
