import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import *

import warnings
warnings.filterwarnings('ignore')

path = '/Users/thanachatkanthanon/All_Machine_Learning_Projects/Datasets/drug200.csv'
my_data = pd.read_csv(path)
print(my_data.head())

#info and describe
print(my_data.info())
print(my_data.describe()) #--> 
print(my_data.dtypes) # dtypes: int64(1), float64(1), object(4)
print(my_data.isnull().sum()) #--> no null values present

#Use LabelEncoder to encode categorical columns to numerical columns
encoder = LabelEncoder()
my_data['Sex'] = encoder.fit_transform(my_data['Sex'])
my_data['BP'] = encoder.fit_transform(my_data['BP'])
my_data['Cholesterol'] = encoder.fit_transform(my_data['Cholesterol'])

print(my_data.head(15))
print("Unique Drug values:", my_data['Drug'].unique())

custom_map = {
    'drugX' : 0,
    'drugY' : 1,
    'drugC' : 2
}

my_data['Drug'] = my_data['Drug'].map(custom_map)
print("NaN count after mapping:", my_data['Drug'].isna().sum())

# Remove rows with NaN in Drug column
my_data = my_data.dropna(subset=['Drug'])
print("Data shape after dropping NaN:", my_data.shape)
print(my_data.head(15))

#Check correlation between input values
print(my_data.drop(columns='Drug').corr())

#distribution of the dataset by plotting the count of the records with each drug recommendation
categorical_counts = my_data['Drug'].value_counts()

#plot the bar to visualize distribution
plt.bar(categorical_counts.index, categorical_counts.values,color='blue')
plt.xlabel("Drug")
plt.ylabel("Count")
plt.title("Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#We notice that distribution for drugY > drugX > drugC, now we separate data for traning and testing
X = my_data.drop(columns='Drug')
y = my_data.loc[:,'Drug']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#define the model with constraints to prevent overfitting
drug_tree = DecisionTreeClassifier(
    criterion='entropy', 
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
drug_tree.fit(X_train,y_train)

#predict values
tree_predictions = drug_tree.predict(X_test)
train_predictions = drug_tree.predict(X_train)

#check for model accuracy
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, tree_predictions)
print("Decision Tree Train Accuracy: %.2f " % train_accuracy)
print("Decision Tree Test Accuracy: %.2f " % test_accuracy)

# Add cross-validation for more reliable accuracy
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(drug_tree, X, y, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean CV Accuracy: %.2f " % cv_scores.mean())