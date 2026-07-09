import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# --- Load Data ---
file_path = 'Datasets/Obesity_level_prediction_dataset (multi-class classification).csv'
df = pd.read_csv(file_path)

TARGET = 'NObeyesdad'

# --- Exploratory Data Analysis ---
print("=== Data Inspection ===")
print(df.isnull().sum())
print("\nData Types & Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

print("\nDistribution of Obesity Levels:")
target_counts = df[TARGET].value_counts()
print(target_counts)

plt.figure(figsize=(8, 5))
sns.countplot(y=TARGET, data=df, order=target_counts.index, saturation=1.0)
plt.title('Distribution of Obesity Levels')
plt.xlabel("Observed Counts")
plt.tight_layout()
plt.show()

# --- Preprocessing ---
print("\n=== Preprocessing ===")

# Encode target separately, keep a mapping for readability later
df[TARGET] = df[TARGET].astype('category')
target_labels = df[TARGET].cat.categories
y = df[TARGET].cat.codes.values

# Identify feature columns (exclude target)
continuous_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
continuous_columns = [c for c in continuous_columns if c != TARGET]
categorical_columns = [c for c in df.columns if df[c].dtype == 'object']

print(f"Continuous columns: {continuous_columns}")
print(f"Categorical columns: {categorical_columns}")

# Correlation check (numeric features only, against encoded target)
print("\nCorrelation with target:")
for col in continuous_columns:
    corr = df[col].corr(pd.Series(y))
    print(f"{col}: {corr:.3f}")

# Scale continuous features
scaler = RobustScaler()
scaled_continuous = scaler.fit_transform(df[continuous_columns])
scaled_continuous_df = pd.DataFrame(scaled_continuous, columns=continuous_columns)

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
encoded_categorical = encoder.fit_transform(df[categorical_columns])
encoded_categorical_df = pd.DataFrame(
    encoded_categorical,
    columns=encoder.get_feature_names_out(categorical_columns)
)

# Combine into final feature matrix
X = pd.concat([scaled_continuous_df, encoded_categorical_df], axis=1)
feature_names = X.columns.tolist()

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42
)

# --- Modeling ---
print("\n=== Modeling ===")

# 1. One-vs-Rest Logistic Regression
print("\n--- One-vs-Rest ---")
model_ovr = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=1000)
model_ovr.fit(X_train, y_train)
y_pred_ovr = model_ovr.predict(X_test)
acc_ovr = accuracy_score(y_test, y_pred_ovr)
print(f"One-vs-Rest Accuracy: {acc_ovr * 100:.2f}%")
print(classification_report(y_test, y_pred_ovr, target_names=[str(l) for l in target_labels]))

# 2. One-vs-One Logistic Regression
print("\n--- One-vs-One ---")
model_ovo = OneVsOneClassifier(LogisticRegression(solver='lbfgs', max_iter=1000))
model_ovo.fit(X_train, y_train)
y_pred_ovo = model_ovo.predict(X_test)
acc_ovo = accuracy_score(y_test, y_pred_ovo)
print(f"One-vs-One Accuracy: {acc_ovo * 100:.2f}%")

# --- Feature Importance (from OVR model) ---
print("\n=== Visualizations ===")

# coef_ shape is (n_classes, n_features) for multi-class OVR
weights = np.mean(np.abs(model_ovr.coef_), axis=0)

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': weights
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, color='skyblue')
plt.xlabel('Importance (Mean Abs Coefficient)')
plt.ylabel('Features')
plt.title("Feature Importance - One-vs-Rest Logistic Regression")
plt.tight_layout()
plt.show()

print("\nLab Completed successfully!")