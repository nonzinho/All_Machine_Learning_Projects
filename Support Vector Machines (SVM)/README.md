# 🧠 Credit Card Fraud Detection using Support Vector Machines (SVM)

## 📌 Overview

This project applies a Linear Support Vector Machine (Linear SVC) to detect fraudulent credit card transactions. Due to the severe class imbalance present in the dataset, class weighting techniques were used during model training and performance was evaluated using the ROC-AUC metric.

---

## 📊 Dataset

The Credit Card Fraud Detection dataset contains anonymized transaction records with:

* 30 input features
* Binary target variable (`Class`)

  * `0` → Legitimate Transaction
  * `1` → Fraudulent Transaction

Class distribution:

* Legitimate Transactions: 284,315
* Fraudulent Transactions: 492

Fraudulent transactions account for approximately **0.172%** of all observations, making this a highly imbalanced classification problem.

---

## 🔍 Exploratory Data Analysis

Performed exploratory analysis including:

* Dataset inspection
* Missing value analysis
* Class distribution visualization
* Feature correlation analysis
* Fraud vs Non-Fraud transaction comparison

---

## ⚙️ Data Preprocessing

### Feature Scaling

Standardized feature values using:

```python
StandardScaler()
```

to ensure each feature contributes equally during model training.

### Feature Normalization

Applied L1 normalization using:

```python
normalize(X, norm='l1')
```

to scale samples consistently.

### Train-Test Split

The dataset was split into:

* 70% Training Data
* 30% Testing Data

using `train_test_split()`.

---

## 🤖 Support Vector Machine Model

A Linear Support Vector Classifier was implemented using Scikit-Learn.

### Hyperparameters

```python
LinearSVC(
    class_weight='balanced',
    loss='hinge',
    fit_intercept=False,
    random_state=31
)
```

### Class Imbalance Handling

The parameter:

```python
class_weight='balanced'
```

automatically assigns higher penalties to misclassified fraudulent transactions, reducing bias toward the majority class.

---

## 📈 Model Evaluation

The model was evaluated using the ROC-AUC metric.

Unlike accuracy, ROC-AUC measures the model's ability to rank fraudulent transactions higher than legitimate transactions across various classification thresholds.

### ROC-AUC Score

```text
Decision Tree ROC-AUC score : 0.939
SVM ROC-AUC score: 0.986
```

---

## 🧠 Key Concepts Demonstrated

* Binary Classification
* Support Vector Machines (SVM)
* Linear Decision Boundaries
* Class Imbalance Handling
* Feature Scaling
* Feature Normalization
* ROC-AUC Evaluation
* Fraud Detection

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn
* Jupyter Notebook

---

## 📚 Learning Outcome

This project demonstrates how Support Vector Machines can be applied to highly imbalanced classification problems while utilizing class weighting and ROC-AUC evaluation to build a more reliable fraud detection model.
