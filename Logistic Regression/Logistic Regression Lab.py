import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss 

#Telecommunications company concerned with the number of customers leaving their land-line business for cable competitors. 
#They need to understand who is more likely to leave the company.
#Telco Churn is a hypothetical data file that concerns a telecommunications company's efforts to reduce turnover in its customer base.
#Each case corresponds to a separate customer and it records various demographic and service usage information.

#Load the dataset
df = pd.read_csv(r"C:\Users\PC\Documents\Machine Learning Lab\Datasets\Logistic Regression Dataset.csv")
print(df.head())
#select until 7th column and add the last column to it
df = df[['tenure','age','address','income','ed','employ','equip','churn']]
df['churn'] = df['churn'].astype(int) # --> Convert the 'churn' column to integer type from float
print(df.head())

X = np.asarray(df[['tenure','age','address','income','ed','employ','equip']])
print(X[0:5]) #--> Print the first 5 rows of the feature matrix X
y = np.asarray(df['churn'])
print(y[0:5]) #--> Print the first 5 values of the target vector

#normalize feature matrix X
X_norm = StandardScaler().fit(X).transform(X)
print(X_norm[0:5]) #--> Print the first 5 rows of the normalized feature matrix X_norm

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)

#fit data into logistic regression model
LR = LogisticRegression().fit(X_train, y_train)
y_hat = LR.predict(X_test)
print(y_hat[0:5].reshape(-1,1)) #--> Print the first 5 predicted values of y_hat
y_hat_proba = LR.predict_proba(X_test)
print(y_hat_proba[0:5].reshape(-1,2)) #--> Print the first 5 predicted probabilities of y_hat_proba

coefficients = pd.Series(LR.coef_[0], index=df.columns[:-1]) #--> create series to store first row of coefficients and assign column as y-axis labels
coefficients.sort_values().plot(kind='barh') #--> sorts the coefficients and plots them as a horizontal bar chart
plt.title('Logistic Regression Coefficients')
plt.xlabel('Coefficient Value') #--> Set the x-axis label to 'Coefficient Value' for the first row of coefficients
plt.ylabel('Feature') #--> Set the y-axis label to 'Feature' for the first row of coefficients
plt.tight_layout() #--> Adjust the layout of the plot to prevent overlapping of labels and title
plt.show()

#Large positive value of LR Coefficient for a given field indicates that increase in this parameter will lead to better chance of a positive, i.e. 1 class. 
#A large negative value indicates the opposite, which means that an increase in this parameter will lead to poorer chance of a positive class. 
#A lower absolute value indicates weaker affect of the change in that field on the predicted class. Let us examine this with the following exercises.
#this basically means that the higher the value of tenure, the more likely the customer is to churn,
#while the higher the value of age, the less likely the customer is to churn.
#we can see that the coefficient for 'tenure' is positive, which means that as tenure increases, the likelihood of churn also increases.
#On the other hand, the coefficient for 'age' is negative, which means that as age increases, the likelihood of churn decreases.

print(f"Log Loss: {log_loss(y_test, y_hat_proba)}")

#Let us assume we add the feature 'callcard' to the original set of input features. What will the value of log loss be in this case?
#To add the feature 'callcard' to the original set of input features, 
# you would first need to ensure that the 'callcard' column is present in your dataset and then include it in the feature matrix X. 
# After that, you would need to normalize the new feature, split the data into training and testing sets, 
#fit the logistic regression model again, and calculate the log loss with the new feature included.

#visualizing the 3d plot of the first three features (tenure, age) against the target variable (churn)
#add surfaces to the plot to visualize the decision boundary of the logistic regression model
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_norm[:, 0], X_norm[:, 1], X_norm[:, -1])
ax.set_xlabel('Tenure')
ax.set_ylabel('Age')
ax.set_zlabel('Churn')
plt.show()