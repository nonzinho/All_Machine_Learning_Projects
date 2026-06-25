# 👥 Customer Service Category Prediction using K-Nearest Neighbors (KNN)

## 📌 Overview

This project implements the K-Nearest Neighbors (KNN) classification algorithm to predict a customer's service subscription category based on demographic and customer-related attributes.

The objective is to classify customers into one of four service categories while exploring how the number of nearest neighbors (**k**) affects model performance.

---

## 📊 Dataset

The dataset contains customer demographic and service-related information, including:

* Region
* Age
* Marital Status
* Income
* Education
* Employment
* Retire Status
* Gender
* Residence
* Additional customer attributes

### Target Variable

```text
custcat
```

Customer service category:

* Category 1 — Basic Service
* Category 2 — E-Service
* Category 3 — Plus Service
* Category 4 — Total Service

The dataset is perfectly balanced, with **266 observations** for each class, reducing the likelihood of bias toward any single category.

---

## 🔍 Exploratory Data Analysis (EDA)

Several exploratory analysis techniques were performed before model training:

* Dataset inspection using `info()`
* Summary statistics using `describe()`
* Missing value analysis
* Duplicate row analysis
* Class distribution analysis
* Correlation matrix visualization
* Feature correlation analysis with the target variable

---

## ⚙️ Data Preprocessing

### Feature Matrix

```python
X = df.drop('custcat', axis=1)
```

### Target Variable

```python
y = df['custcat']
```

### Feature Standardization

Since KNN relies on distance calculations, all features were standardized using:

```python
StandardScaler()
```

Standardization transforms each feature to have:

* Mean = 0
* Standard Deviation = 1

This prevents features with larger numerical ranges from dominating the Euclidean distance calculation.

### Train-Test Split

The dataset was divided into:

* 80% Training Data
* 20% Testing Data

using `train_test_split()`.

---

## 🤖 K-Nearest Neighbors (KNN)

A KNN classifier was implemented using Scikit-Learn.

### Initial Model

```python
KNeighborsClassifier(n_neighbors=3)
```

The algorithm predicts the class of a new customer by examining the **k nearest neighbors** in the feature space and assigning the majority class among those neighbors.

---

## 🔧 Hyperparameter Tuning

The number of neighbors (**k**) is one of the most important hyperparameters in KNN.

To identify the optimal value, models were trained for:

```text
k = 1 → 10
```

For each value of **k**, the following were computed:

* Classification Accuracy
* Standard Error of Accuracy

The results were visualized using an Accuracy vs. Number of Neighbors plot.

The shaded region around the accuracy curve represents the **standard error**, providing an estimate of the uncertainty associated with each measured accuracy.

---

## 📈 Model Evaluation

The model was evaluated using:

### Accuracy Score

Measures the proportion of correctly classified observations.

```text
Accuracy ≈ 0.36
```

The highest accuracy was obtained at approximately:

```text
k = 3
```

Although this produced the best performance among the tested values, the overall classification accuracy remained relatively low.

---

## 🧠 Key Concepts Demonstrated

* Supervised Learning
* Multi-Class Classification
* K-Nearest Neighbors (KNN)
* Euclidean Distance
* Feature Scaling
* Hyperparameter Tuning
* Model Evaluation
* Standard Error Visualization
* Exploratory Data Analysis (EDA)

---

## ⚠️ Dataset Limitations

Although the dataset is perfectly balanced across all four service categories, the overall classification accuracy remained relatively low.

This suggests that the available demographic and customer-related features may not sufficiently distinguish between the service categories. Many important factors influencing customer service selection, such as user preferences, purchasing behavior, historical service usage, or marketing exposure, are not included in the dataset.

As a result, customers from different categories may appear similar in the feature space, making it difficult for the KNN algorithm to identify clear neighborhood boundaries.

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn
* Jupyter Notebook

---

## 💡 What I Learned

* How the K-Nearest Neighbors algorithm performs classification using distance-based learning.
* Why feature standardization is essential for KNN models.
* How changing the value of **k** affects model complexity and prediction performance.
* How hyperparameter tuning can improve model performance.
* How to interpret an Accuracy vs. Number of Neighbors graph.
* The importance of evaluating model limitations when predictive performance is low.

---

## 🚀 Future Improvements

Potential improvements include:

* Cross-Validation for selecting the optimal **k**
* Weighted KNN (`weights='distance'`)
* Alternative distance metrics (Manhattan, Minkowski)
* Feature Engineering
* Principal Component Analysis (PCA)
* Comparison with Logistic Regression, Decision Trees, and Support Vector Machines

---

## 📚 Learning Outcome

This project demonstrates the implementation of the K-Nearest Neighbors algorithm for multi-class classification, emphasizing the importance of feature scaling, hyperparameter tuning, and model evaluation. It also highlights how dataset characteristics and feature quality directly influence the predictive performance of distance-based machine learning models.
