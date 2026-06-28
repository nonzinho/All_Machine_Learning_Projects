# 🏠 California Housing Price Prediction using Random Forest and XGBoost

A machine learning regression project that compares the performance of **Random Forest Regressor** and **XGBoost Regressor** on the California Housing dataset. The project includes exploratory data analysis (EDA), feature correlation analysis, model training, evaluation, and performance benchmarking.

---

## 📌 Project Overview

Predicting housing prices is a classic supervised machine learning problem with many real-world applications in real estate, finance, and urban planning.

This project aims to compare two ensemble learning algorithms:

- 🌲 Random Forest Regressor
- 🚀 XGBoost Regressor

The comparison focuses on:

- Exploratory Data Analysis (EDA)
- Feature correlation analysis
- Model training
- Prediction performance
- Training & inference speed
- Regression evaluation metrics
- Visualization of model predictions

---

## 📂 Dataset

**Dataset:** California Housing Dataset

The dataset contains **20,640 observations** and **8 numerical features** describing demographic and geographic information of California housing districts.

### Features

| Feature | Description |
|---------|-------------|
| MedInc | Median income |
| HouseAge | Median house age |
| AveRooms | Average number of rooms |
| AveBedrms | Average number of bedrooms |
| Population | Population of the block |
| AveOccup | Average occupancy |
| Latitude | Geographic latitude |
| Longitude | Geographic longitude |

**Target Variable**

- Median House Value

---

# 🛠 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

---

# 📊 Exploratory Data Analysis

Before model training, the dataset was analyzed to better understand its structure and feature relationships.

The following analyses were performed:

- Dataset information
- Summary statistics
- Missing value inspection
- Feature correlation analysis
- Correlation heatmap

---

## Feature Correlation with Target

![Feature Correlation](Random%20Forest%20and%20XGBoost/images/feature_correlation_bar.png)

This visualization illustrates the Pearson correlation coefficient between each feature and the target variable.

### Key Findings

- **Median Income** has the strongest positive relationship with house prices.
- **Average Rooms** and **House Age** have weak positive correlations.
- **Latitude** shows a weak negative correlation.
- **Population** and **Average Occupancy** contribute little linear information.

---

## Correlation Heatmap

![Correlation Heatmap](Random%20Forest%20and%20XGBoost/images/correlation_heatmap.png)

The correlation heatmap provides insight into relationships among all numerical features.

### Observations

- **Average Rooms** and **Average Bedrooms** are highly correlated (0.85).
- **Latitude** and **Longitude** have a strong negative correlation (-0.92).
- **Median Income** exhibits the strongest correlation with the target variable.

Understanding these relationships helps identify important predictors before model training.

---

# 🤖 Machine Learning Models

## Random Forest Regressor

Random Forest is an ensemble learning algorithm that constructs multiple decision trees using bootstrap sampling and averages their predictions.

### Advantages

- Handles nonlinear relationships
- Reduces overfitting
- Robust against noisy data
- Requires minimal preprocessing

---

## XGBoost Regressor

XGBoost (Extreme Gradient Boosting) builds decision trees sequentially, where each new tree learns from the errors of previous trees.

### Advantages

- Excellent predictive performance
- Efficient training
- Fast prediction
- Built-in regularization
- Handles complex nonlinear relationships

---

# ⚙️ Machine Learning Workflow

```
Load Dataset
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Train/Test Split (80/20)
      │
      ▼
Train Random Forest
      │
      ▼
Train XGBoost
      │
      ▼
Evaluate Performance
      │
      ▼
Visualize Predictions
```

---

# 📈 Evaluation Metrics

Both models were evaluated using:

- Mean Squared Error (MSE)
- R² Score
- Training Time
- Prediction Time

---

# 📊 Model Predictions

![Prediction Comparison](Random%20Forest%20and%20XGBoost/images/model_predictions.png)

The scatter plots compare predicted house prices against the true values.

### Interpretation

- The **black dashed line** represents perfect predictions.
- The **red dashed lines** indicate ±1 standard deviation from the ideal prediction.
- Points closer to the black line indicate more accurate predictions.

The XGBoost model demonstrates a tighter clustering around the ideal prediction line, indicating better predictive accuracy.

---

# 📊 Performance Comparison

| Metric | Random Forest | XGBoost |
|---------|--------------:|---------:|
| Mean Squared Error | **0.26** | **0.22** |
| R² Score | **0.80** | **0.83** |
| Training Time | **9.76 s** | **0.18 s** |
| Prediction Time | **0.113 s** | **0.003 s** |

---

# 🏆 Conclusion

Both models achieved strong predictive performance on the California Housing dataset.

However, **XGBoost consistently outperformed Random Forest** by:

- Achieving a lower Mean Squared Error
- Producing a higher R² Score
- Training significantly faster
- Making predictions more efficiently

Overall, XGBoost proved to be the better model for this regression task.

---

# 🚀 How to Run

Clone the repository

```bash
git clone https://github.com/nonzinho/All_Machine_Learning_Projects.git
```

Navigate to the project directory

```bash
cd "Random Forest and XGBoost"
```

Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

Run the project

```bash
python HousePricePrediction_RF_XGBoost.py
```

---

# 📁 Project Structure

```
Random Forest and XGBoost/
│
├── README.md
├── HousePricePrediction_RF_XGBoost.py
├── RandomForest_vs_XGBoost.ipynb
│
├── images/
│   ├── feature_correlation_bar.png
│   ├── correlation_heatmap.png
│   └── model_predictions.png
```

---

# 📚 Machine Learning Concepts Demonstrated

- Supervised Learning
- Regression
- Ensemble Learning
- Random Forest
- Gradient Boosting
- XGBoost
- Exploratory Data Analysis (EDA)
- Correlation Analysis
- Model Evaluation
- Performance Benchmarking
- Data Visualization

---

## 👤 Author

**Thanachat Kanthanon**

Computer Engineering (Artificial Intelligence)

Passionate about Machine Learning, Data Science, and Artificial Intelligence.
