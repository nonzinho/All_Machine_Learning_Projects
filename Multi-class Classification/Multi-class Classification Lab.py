import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score
import warnings 
warnings.filterwarnings('ignore') # Suppress warnings for cleaner output

df = pd.read_csv(r"C:\Users\PC\Documents\Machine Learning Lab\Datasets\Obesity_level_prediction_dataset (multi-class classification).csv")

#print necessary information
print(df.isnull().sum())
print(df.info())
print(df.describe())
print(df.head())
# --> all categorical columns are object type, whereas others are float64


#exploratory data analysis (EDA)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    correlation = df[col].corr(df['Weight'])
    print(f"Correlation between {col} and Weight: {correlation}") # --> for loop to select only numeric columns and analize their correlation

print(df['NObeyesdad'].value_counts().sort_values(ascending = True)) # --> counts the number of entries for each categorical type, and sorts it in ascending order
sns.countplot(y='NObeyesdad',data=df,saturation=1.0) # --> visualizes number of entries for each categorical type
plt.title('Distribution of Obesity Levels')
plt.tight_layout()
plt.show()

# Standardizing continuous numerical features
continuous_columns = df.select_dtypes(include=['float64']).columns.tolist()
print(continuous_columns)

#initialize standardscaler and fit_transform continuous columns in df
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[continuous_columns])

#new pd dataframe with data as scaled_features and columns as continuous columns using scaler.get_feature_names_out(input_featurs : ArrayLike)
scaled_continuous_cols = pd.DataFrame(data=scaled_features, columns=scaler.get_feature_names_out(input_features = continuous_columns))

#combining with the original dataset
scaled_df = pd.concat([df.drop(columns=continuous_columns), scaled_continuous_cols], axis=1) # --> drop original continuous columns 
#and concatenate with scaled continuous columns. axis=1 means concatenate along columns

#identifying categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('NObeyesdad')
print(categorical_columns)

#initializing onehotencoder
encoder = OneHotEncoder(sparse_output=False, drop='first') # --> sparse_output=false returns 2d numpy array, drop first categorical column to prevent multicollinearity
encoded_features = encoder.fit_transform(df[categorical_columns])

#new dataframe consisting of encoded categorical columns
encoded_categorical_columns = pd.DataFrame(data=encoded_features, columns=encoder.get_feature_names_out(input_features=categorical_columns))
encoded_df = pd.concat([df.drop(columns=categorical_columns),encoded_categorical_columns], axis=1)

#encoding target variable
encoded_df['NObeyesdad'] = encoded_df['NObeyesdad'].astype('category').cat.codes

#data splitting
X = encoded_df.drop('NObeyesdad',axis=1)
y = encoded_df['NObeyesdad']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

#Logistic Regression with OnevAll --> trains single binary classifier for each class: N class, N classifiers 
#simple, easier to implement and more efficient in terms of number of classifiers (k) but may struggle with highly imbalanced datasets
model_ova = LogisticRegression(multi_class='ovr', max_iter=1000) # --> multi_class='ovr' specifies one-vs-rest strategy, we iterate 1000 times for the model to converge (find the best parameters)
model_ova.fit(X_train,y_train)

#predictions and accuracy for onevrest/onevall
y_pred_ova = model_ova.predict(X_test)
print("One v All Strategy")
print(f"Accuracy {np.round(100* accuracy_score(y_test, y_pred_ova),2)}%") #--> 73.05% accuracy

#Logistic Regression for OnevOne Classifier
model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ovo.fit(X_train,y_train)

#predictions and accuracy for onevone
y_pred_ovo = model_ovo.predict(X_test)
print("One v One Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test,y_pred_ovo),2)}%")

#as we can see, onevone performs better in this case with 84.87% accuracy score, whereas 73.05% for onevrest/onevall

#bar chart of feature importance using the coefficients from the One vs All logistic regression model. Also try for the One vs One model
feature_importance_ova = np.mean(np.abs(model_ova.coef_),axis=0) # --> calculate mean absolute value of coefficients across all classes to get overall feature importance, returns an array with importance values for each feature which is 1d array with shape (n_features,)
print(feature_importance_ova) # --> print feature importance values for onevall model
plt.barh(X.columns, feature_importance_ova) #arguments: X.columns for y-axis labels, feature_importance_ova for bar lengths, horizontal bar chart
plt.title("Feature Importance (One v All)")
plt.xlabel("Importance")
plt.show()

# For One vs One model
coefs = np.array([est.coef_ for est in model_ovo.estimators_]) # --> extract coefficients from each binary classifier and stack vertically
feature_importance_ovo = np.mean(np.abs(coefs), axis=0) # --> calculate mean absolute value of coefficients across all classifiers
feature_importance_ovo = feature_importance_ovo.flatten()  # Ensure shape matches number of features
print(feature_importance_ovo) # --> print feature importance values for onevone model
plt.barh(X.columns, feature_importance_ovo)
plt.title("Feature Importance (One v One)")
plt.xlabel("Importance")
plt.show()