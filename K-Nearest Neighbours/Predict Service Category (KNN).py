import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ----- Extracting data into Pandas dataframe -----
path = r"C:\Users\PC\All_Machine_Learning_Projects\Datasets\teleCust1000t.csv"
df = pd.read_csv(path)
print(df.head(15)) 
# --> As we can see, the dataset has demographic data, such as region, age, and marital, with custcat being the target value with 4 cateogories.

# ----- Examining the data -----
print(df.info()) # --> Notice that there's only int64 and float64 values - including gender, marital, etc?!
print(df.describe())
print(df.isnull().sum()) # --> No null values!
print(df.duplicated().sum()) # --> No duplicate values!
print(df.custcat.value_counts().sort_values(ascending=False)) # --> All four classes contain 266 samples each, resulting in a perfectly balanced dataset. This helps prevent the KNN classifier from favoring one class over another due to class imbalance.

# ----- Visualize correlation matrix -----
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5) # --> Examine how different features relate to each other.
plt.show()

# ----- Visualize correlation matrix (for target value) -----
correlation_values = df.corr()['custcat'].drop('custcat').sort_values(ascending=False)
print(correlation_values.values)
fig, ax = plt.subplots()
ax.barh(correlation_values.index, correlation_values.values)
ax.set_xlabel("Correlation Values Towards Target Value")
ax.set_ylabel("Feature Columns")
fig.tight_layout()
plt.show()

# ----- Separate input and target features -----
X = df.drop('custcat', axis=1)
y = df['custcat']

# ----- Normalize data -----
# --> Data normalization is important for the KNN model. KNN makes predictions based on the distance between data points (samples).
# --> The algorithm finds the k-nearest neighbors by measuring the distance between the test point and other data points in the dataset.
# --> By normalizing / standardizing the data, you ensure that all features contribute equally to the distance calculation. Since normalization scales each feature to have zero mean and unit variance, it puts all features on the same scale (with no feature dominating due to its larger range).
X_norm = StandardScaler().fit_transform(X)

# ----- Splitting dataset -----
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

# ----- K-NN Classification Algorithm -----
k = 3 # --> number of neighbours
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_model = knn_classifier.fit(X_train, y_train)

# ----- Predict Model -----
y_hat = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_hat)
print("Prediction accuracy: %.2f" % accuracy) # --> Accuracy = 0.33 (Pretty Low)

# ----- Choosing the correct value for k -----
Ks = 10
acc = np.zeros((Ks))
std_acc = np.zeros((Ks))

for n in range(1, Ks+1):
    # --> Train model and predict
    knn_model_n = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat = knn_model_n.predict(X_test)
    acc[n-1] = accuracy_score(y_test, yhat)
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0]) # --> Calculate standard deviation. (for yhat == y_test)

# ----- Visualize iteration for the optimal k value -----
plt.plot(range(1,Ks+1),acc,'g')
plt.fill_between(range(1,Ks+1),acc - 1 * std_acc,acc + 1 * std_acc, alpha=0.10) # shaded blue region (standard error/deviation region)
plt.legend(('Accuracy value', 'Standard Deviation'))
plt.ylabel('Model Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", acc.max(), "with k =", acc.argmax()+1) 