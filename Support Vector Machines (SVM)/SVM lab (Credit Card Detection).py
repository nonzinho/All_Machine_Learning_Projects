import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ----- Extracting dataset into Pandas dataframe -----
dir = r"C:\Users\PC\Documents\Machine Learning Lab\Datasets\creditcard.csv"
df = pd.read_csv(dir)
print(df.head(15))

# ----- Exploratory Data Analysis (EDA) -----
print(df.info()) # --> Every feature column (except target column) is float64 datatype.
print(df.describe()) #--> As we can see, the standard deviation for each column value is not very well distributed, suggesting a soft margin, which could potentially lead to missclassifications.
print(df.isnull().sum()) # --> No null values for all feature columns.
print(df.Class.unique()) # --> 2 unique values for target value 'Class' : Fraud, not Fraud.
print(df.Class.value_counts()) # --> Counts number of entries per each class in target column: 0: 284315, 1:492, meaning about 0.172% of the credit cards in the dataset are classified as fraud.

# ----- Plotting value counts -----
sizes = df.Class.value_counts()
labels = df.Class.unique()
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels)
ax.set_title("Target Value counts")
fig.tight_layout()
plt.show()

# ----- Plotting correlation values -----
correlation_values = df.corr()['Class'].drop('Class')
correlation_values.plot(kind='barh', figsize=(10,6)) 
plt.show()

# ----- Data Preprocessing -----
df.iloc[:, 1:30] = StandardScaler().fit_transform(df.iloc[:, 1:30]) # --> Standardize features by removing the mean and scaling to unit variance.
data_matrix = df.values

# ----- X, y values -----
X = data_matrix[:, 1:30]
X = normalize(X, norm='l1')
y = data_matrix[:, 30]

# ----- Data Splitting (train, test) -----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----- Build Decision Tree Classifier model -----
w_train = compute_sample_weight('balanced', y_train) # --> Compute the sample weights to be used as input to the train routine so that it takes into account the class imbalance present in this dataset.
dt = DecisionTreeClassifier(max_depth=4, random_state=35)
dt.fit(X_train, y_train, sample_weight=w_train)
# --> You can also do dt = DecisionTreeClassifier(max_depth=4, random_state=35, class_weight = 'balanced) without having to use compute_sample_weight

# ----- Build Support Vector Machine (SVM) -----
svm = LinearSVC(class_weight='balanced', random_state=31, loss='hinge', fit_intercept=False)
svm.fit(X_train, y_train)

# ----- Predictions for each model -----
y_pred_dt = dt.predict_proba(X_test)[:,1] # --> Compute the probabilities of the test samples belonging to the class of fraudulent transactions. 
roc_auc_dt = roc_auc_score(y_test, y_pred_dt)
print('Decision Tree ROC-AUC score : {0:.3f}'.format(roc_auc_dt))

y_pred_svm = svm.decision_function(X_test)
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
print("SVM ROC-AUC score: {0:.3f}".format(roc_auc_svm))
# --> ROC-AUC evaluates a model's ability to rank positive samples higher than negative samples and therefore requires continuous confidence scores rather than hard class labels. For the Decision Tree, predict_proba() returns the probability of each sample belonging to the fraud class, while for LinearSVC, decision_function() returns the signed distance of each sample from the decision boundary. 
# --> Using predict() would only return binary class labels (0 or 1), losing valuable confidence information needed to compute an accurate ROC-AUC score.