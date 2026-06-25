# 🌳 Regression Tree for Taxi Tip Amount Prediction

## 📌 Overview

This project implements a Regression Tree using the CART (Classification and Regression Trees) algorithm to predict taxi tip amounts based on trip-related features from New York City taxi ride data.

The objective is to explore how tree-based regression models partition the feature space and learn nonlinear relationships between trip characteristics and customer tipping behavior.

---

## 📊 Dataset

The dataset contains information about taxi trips, including:

* VendorID
* passenger_count
* trip_distance
* RatecodeID
* PULocationID
* DOLocationID
* payment_type
* fare_amount
* additional trip-related attributes

### Target Variable

```text
tip_amount
```

The goal is to predict the amount tipped by the passenger at the end of the trip.

---

## 🔍 Exploratory Data Analysis

Several exploratory analysis techniques were performed to better understand the dataset:

* Dataset inspection using `info()`
* Summary statistics using `describe()`
* Missing value analysis
* Unique value analysis
* Duplicate row detection and removal
* Feature correlation analysis with the target variable

A horizontal bar chart was generated to visualize the correlation between each feature and the target variable.

---

## ⚙️ Data Preprocessing

### Data Cleaning

Duplicate rows were removed from the dataset to improve data quality.

```python
df = df.drop_duplicates()
```

### Feature Matrix and Target Variable

Target variable:

```python
y = df['tip_amount']
```

Feature matrix:

```python
X = df.drop(columns='tip_amount')
```

### Train-Test Split

The dataset was split into:

* 70% Training Data
* 30% Testing Data

using:

```python
train_test_split()
```

to evaluate model performance on unseen data.

---

## 🌳 Regression Tree Model

A CART-based Regression Tree was implemented using Scikit-Learn.

### Model Configuration

```python
DecisionTreeRegressor(
    criterion='squared_error',
    max_depth=3,
    min_samples_leaf=20,
    random_state=42
)
```

### Splitting Criterion

The model uses:

```python
criterion='squared_error'
```

which minimizes the weighted Mean Squared Error (MSE) when selecting the optimal feature and threshold for each split.

At each node, the algorithm searches for the split that produces the greatest reduction in prediction error.

---

## 📈 Feature Importance

Feature importance scores were extracted from the trained Regression Tree using:

```python
dt_reg.feature_importances_
```

This helps identify which variables contribute most to the prediction of taxi tip amounts.

---

## 📏 Model Evaluation

The Regression Tree model was evaluated using:

### R² Score

Measures the proportion of variance in the target variable explained by the model.

```text
R² Score: 0.03
```

### Mean Squared Error (MSE)

Measures the average squared prediction error.

```text
MSE: 27.55
```

### Training vs Testing Performance

Both training and testing R² scores were compared to evaluate the model's ability to generalize and detect potential overfitting.

---

## 🧠 Key Concepts Demonstrated

* Supervised Learning
* Regression Trees
* CART Algorithm
* Recursive Binary Splitting
* Weighted Mean Squared Error
* Feature Importance
* Nonlinear Regression
* Model Evaluation
* Bias-Variance Tradeoff

---

## ⚠️ Dataset Limitations

One challenge of this dataset is that several important factors influencing tip amount are not recorded.

While variables such as trip distance, fare amount, passenger count, and payment type may contribute to tipping behavior, many real-world factors are unavailable, including:

* Customer generosity
* Customer income level
* Driver service quality
* Weather conditions
* Customer mood
* Tourist vs local behavior
* Time pressure and urgency

As a result, a significant portion of the variability in tip amounts may not be explainable using the available features.

This limitation can reduce predictive performance and lead to lower R² scores even when the model is correctly implemented.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn
* Jupyter Notebook

---

## 💡 What I Learned

* How Regression Trees recursively partition feature space.
* Why weighted Mean Squared Error is used as the splitting criterion.
* How tree depth affects model complexity and generalization.
* How feature importance helps interpret tree-based models.
* Why model performance is heavily influenced by the quality and relevance of available features.
* The importance of understanding dataset limitations before evaluating model performance.

---

## 🚀 Future Improvements

Potential improvements include:

* Hyperparameter tuning using GridSearchCV
* Random Forest Regression
* Gradient Boosting Regression
* XGBoost Regression
* Cross-Validation
* One-Hot Encoding of categorical features
* Additional feature engineering

---

## 📚 Learning Outcome

This project demonstrates how Regression Trees can be applied to regression problems involving real-world tabular data. It highlights both the strengths of tree-based models in capturing nonlinear relationships and the challenges that arise when important explanatory variables are missing from the dataset.
