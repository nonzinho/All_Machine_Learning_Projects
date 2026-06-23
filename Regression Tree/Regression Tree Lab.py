from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

import warnings
warnings.filterwarnings('ignore')

# Load dataset
path = r"C:\Users\PC\All_Machine_Learning_Projects\Datasets\yellow_tripdata.csv"
df = pd.read_csv(path)

print(df.head(15))

# ----------------------------
# Exploratory Data Analysis
# ----------------------------
print(df.info())
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nUnique Values:")
print(df.nunique(axis=0))

print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# ----------------------------
# Correlation Analysis
# ----------------------------
correlation_values = df.corr(numeric_only=True)['tip_amount'].drop('tip_amount')
correlation_values = correlation_values.sort_values(ascending=False)

print("\nCorrelation with Tip Amount:")
print(correlation_values)

correlation_values.plot(kind='barh', figsize=(10, 6))
plt.title("Feature Correlation with Tip Amount")
plt.xlabel("Correlation")
plt.tight_layout()
plt.show()

# ----------------------------
# Feature Matrix and Target
# ----------------------------
y = df['tip_amount']

X = df.drop(columns='tip_amount')

# NOTE:
# PULocationID, DOLocationID, payment_type,
# and RatecodeID are categorical identifiers.
# For a more advanced model, consider encoding them.

# ----------------------------
# Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

# ----------------------------
# Decision Tree Regressor
# ----------------------------
dt_reg = DecisionTreeRegressor(
    criterion='squared_error',
    max_depth=3,
    min_samples_leaf=20,
    random_state=42
)

# Train model
dt_reg.fit(X_train, y_train)

# ----------------------------
# Feature Importance
# ----------------------------
importance = pd.Series(
    dt_reg.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

# ----------------------------
# Predictions
# ----------------------------
y_pred = dt_reg.predict(X_test)

# ----------------------------
# Evaluation Metrics
# ----------------------------
mse_score = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nTrain R² Score:", dt_reg.score(X_train, y_train))
print("Test R² Score :", dt_reg.score(X_test, y_test))

print("MSE Score: %.2f" % mse_score)
print("R² Score: %.2f" % r2)