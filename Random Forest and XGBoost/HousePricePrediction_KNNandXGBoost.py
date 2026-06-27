import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# ----- Import dataset into Pandas dataframe -----
path = r"C:\Users\PC\All_Machine_Learning_Projects\Datasets\California_housing.csv"
df = pd.read_csv(path)
print(df.head())

# ----- Exploratory Data Analysis -----
print(df.info()) # --> All values are float64 datatype
print(df.describe()) # --> Shape: 20640 x 9 (20640 observations)
print(df.isnull().sum()) # --> Check for null values

plt.figure()
correlation_values = df.corr()['Target'].drop('Target')
plt.barh(correlation_values.index, correlation_values.values)
plt.xlabel("Correlation Values")
plt.ylabel("Feature Columns")
plt.show()

plt.figure(figsize=(14,6))
corr_values = df.corr(numeric_only=True)
sns.heatmap(
    corr_values,
    annot=True,
    cmap='coolwarm',
    linewidths=0.5,
    linecolor='blue'
)
plt.title("Feature Correlation with Target")
plt.show()

N_observations, N_features = df.shape
print("Number of observations: %i" % N_observations)
print("Number of features: %i" % N_features)

# ----- Splitting data -----
X = df.drop('Target', axis=1)
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----- Initialize model -----
n_estimators = 100
rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
xgb = XGBRegressor(n_estimators=n_estimators, random_state=42)

# ---- Fit models -----
start_time_rf = time.time()
rf.fit(X_train, y_train)
end_time_rf = time.time()
rf_train_time = end_time_rf - start_time_rf

start_time_xgb = time.time()
xgb.fit(X_train, y_train)
end_time_xgb = time.time()
xgb_train_time = end_time_xgb - start_time_xgb
#Difference in performance time
difference_performance_train = np.abs(rf_train_time - xgb_train_time)

#Compute difference between the training performance of the 2 models
print("Execution time for Random Forest Regressor: %f" % rf_train_time)
print("Execution time for XGBoost: %f" % xgb_train_time) 

if rf_train_time < xgb_train_time:
    print("Random Forest executes training time faster by %f seconds" % difference_performance_train)
elif rf_train_time > xgb_train_time:
    print("XGBoost executes training time faster by %f seconds" % difference_performance_train)
else:
    print("Same performance time of %f seconds" % rf_train_time)
# --> Random Forest: 9.76 seconds, XGBoost: 0.18 (9.57 second difference in performance!)

start_rf_predict_time = time.time()
y_pred_rf = rf.predict(X_test)
stop_rf_predict_time = time.time()
rf_predict_time = stop_rf_predict_time - start_rf_predict_time

start_xgb_predict_time = time.time()
y_pred_xgb = xgb.predict(X_test)
stop_xgb_predict_time = time.time()
xgb_predict_time = stop_xgb_predict_time - start_xgb_predict_time

#Compute difference between the prediction performance of the 2 models
difference_performance_predict = np.abs(rf_predict_time - xgb_predict_time)

# Comparing training performance between 2 models
print("Prediction time for Random Forest Regressor: %f" % rf_predict_time)
print("Prediction time for XGBoost: %f" % xgb_predict_time)

if rf_predict_time < xgb_predict_time:
    print("Random Forest executes prediction time faster by %f seconds" % difference_performance_predict)
elif rf_predict_time > xgb_predict_time:
    print("XGBoost executes prediction time faster by %f seconds" % difference_performance_predict)
else:
    print("Same performance time of %f seconds" % rf_predict_time)
# --> Random Forest: 0.113 seconds, XGBoost: 0.003 seconds (0.11 second difference in performance!)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

mse_difference = np.abs(mse_rf - mse_xgb)
r2_difference = np.abs(r2_rf - r2_xgb)

print("Random Forest MSE: %.2f, R^2: %.2f" % (mse_rf, r2_rf)) # --> MSE: 0.26, r2: 0.80
print("XGBoost MSE: %.2f, R^2: %.2f" % (mse_xgb, r2_xgb)) # --> MSE: 0.22, r2: 0.83

if mse_rf > mse_xgb:
    print("XGB MSE has lesser error by %f" % mse_difference)
elif mse_rf < mse_xgb:
    print("Random Forest MSE has lesser error by %f" % mse_difference)
else:
    print("MSE for both models are the same at %f" % mse_rf)

if r2_rf > r2_xgb:
    print("Random Forest R^2 has more accuracy by %f" % r2_difference)
elif r2_rf < r2_xgb:
    print("XGBoost R^2 has more accuracy by %f" % r2_difference)
else:
    print("R^2 for both models accuracy are the same at %f" % r2_rf)

# --> Overall, XGBoost performed better in terms of both accuracy and speed!

# ----- Calculate std of test data -----
std_y_test = np.std(y_test)
print(std_y_test)

# ----- Visualize the Results -----
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
plt.scatter(y_test, y_pred_rf, alpha=0.5, color='blue', ec='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Model')
plt.plot([y_test.min(), y_test.max()], [y_test.min() - std_y_test, y_test.max() + std_y_test], 'r--', lw=1, label='+/- 1 std dev')
plt.plot([y_test.min(), y_test.max()], [y_test.min() - std_y_test, y_test.max() + std_y_test], 'r--', lw=1)
plt.ylim(0,6)
plt.title("Random Forest Predictions vs Actual")
plt.legend()

plt.figure(figsize=(14,6))
plt.subplot(1,2,2)
plt.scatter(y_test, y_pred_xgb, alpha=0.5, color='blue', ec='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Model')
plt.plot([y_test.min(), y_test.max()], [y_test.min() - std_y_test, y_test.max() + std_y_test], 'r--', lw=1, label='+/- 1 std dev')
plt.plot([y_test.min(), y_test.max()], [y_test.min() - std_y_test, y_test.max() + std_y_test], 'r--', lw=1)
plt.ylim(0,6)
plt.title("XGBoost Predictions vs Actual")
plt.legend()
plt.tight_layout()
plt.show()








