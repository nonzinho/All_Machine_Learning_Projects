import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#import dataset
url = r"C:\Users\PC\Documents\Machine Learning Lab\Datasets\FuelConsumptionCo2.csv"
df = pd.read_csv(url)

#verify data load
print(df.head())
print(df.describe())

#selecting indicative features 
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.sample(5))

#visualize the data
viz = cdf[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
viz.hist()
plt.show()

#larger engine size means more emission
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.xlim(0,27) #set x limit to 27 to better visualize the data
plt.show()

#cylinder against CO2 emission
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color="blue")
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.xlim(0,10) #set x limit to 10 to better visualize the data
plt.show()

X = cdf.ENGINESIZE.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()
print(X[:10])


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

print(X_train[:10])
print(type(X_train))
print(np.shape(X_train)) #853 rows, 1 column
print(type(y_train))
print(y_train[:10])
print(np.shape(y_train)) #853 rows, 1 column

regressor = LinearRegression()
regressor.fit(X_train.reshape(-1,1), y_train) #reshape X_train to be a 2D array with one column; -1 means that the number of rows will be inferred from the length of the ar

#print the coefficients and intercept
print("Coefficients: ", regressor.coef_[0]) #regressor.coef_ is a 2D array with one row and one column, so we access the first element to get the coefficient value
print("Intercept: ", regressor.intercept_) #regressor.intercept_ is a 1D array with one element, so we access the first element to get the intercept value

#visualize regression line
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, regressor.coef_[0]*X_train + regressor.intercept_, color='red') #regressor.coef_[0] is the slope of the line, and regressor.intercept_ is the y-intercept
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.xlim(0,27) #set x limit to 27 to better visualize the
plt.show()

#measure the accuracy of the model
y_pred = regressor.predict(X_test.reshape(-1,1))
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#the more accurate the model, the lower the MAE, MSE, and the higher the R2 score. An R2 score of 1 indicates a perfect fit, 
#while an R2 score of 0 indicates that the model does not explain any of the variance in the target variable.
#r2 score should be between 0 and 1, with higher values indicating a better fit of the model to the data.

print("Mean Absolute Error: %.2f" % mae) 
print("Mean Squared Error: %.2f" % mse)
print("R2 Score: %.2f" % r2)


#visualize the predicted values against the actual values
plt.scatter(X_test, y_test, color='blue', label='Actual') #actual values in blue
plt.scatter(X_test, y_pred, color='red', label='Predicted') #predicted values in red
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.xlim(0,27) #set x limit to 27 to better visualize the data
plt.legend()
plt.show()