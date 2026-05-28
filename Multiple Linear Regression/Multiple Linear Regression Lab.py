import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from termcolor import colored
from sklearn.metrics import mean_squared_error, r2_score


#import dataset

df = pd.read_csv(r"C:\Users\PC\Documents\Machine Learning Lab\Datasets\FuelConsumptionCo2_MultipleRegression.csv")


print(df.columns)#check dataset clmns
print(df.head()) #check dataset head
print(df.describe()) #describe dataset --> mean, std, min, max, 25%, 50%, 75%

#drop categorical data/columns
df = df.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE'], axis=1) #drop categorical data/columns (axis = 1 for clmns)
print(df.head()) #check dataset head after dropping categorical data/columns

# we have to eliminate strong dependency between independent variables by selecting best ones from each group of highly correlated variables 
# (why? because of multicollinearity)
#multicollinearity is a phenomenon in which two or more independent variables in a multiple regression model are highly correlated with each other,
#which can lead to unreliable and unstable estimates of the regression coefficients. To check for multicollinearity,
#we can calculate the correlation matrix of the independent variables and look for high correlations (e.g., above 0.8 or below -0.8) between them.

print(df.corr()) #check correlation between independent variables
#notice that engine size and cylinders are highly correlated (0.9), so we can drop one of them 
# (e.g., cylinders because engine size is a more correlated variable to the target) to eliminate multicollinearity
df = df.drop(['CYLINDERS'], axis=1) #drop cylinders column
#since fuel_consumption_comb_mlb is the most correlated variable to the target (co2emissions), 
# we will keep it and drop the other fuel consumption variables
df = df.drop(['FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB'], axis=1)
print(df.head()) #check dataset head after dropping multicollinear variables

#To help with selecting predictive features that are not redundant, consider the following scatter matrix, 
# which shows the scatter plots for each pair of input features. The diagonal of the matrix shows each feature's histogram.
axes = pd.plotting.scatter_matrix(df,alpha=0.2) #scatter matrix to show the scatter plots for each pair of input features, alpha = 0.2 for transparency
print(axes)
for ax in axes.flatten(): #axes.flatten() to flatten the 2D array of axes into a 1D array for easier iteration
# basically axes is [row,columns] and flatten() will convert it to [plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8, plot9] for easier iteration 
#or row * columns = total number of plots, so flatten will convert it to a 1D array of plots for easier iteration
    ax.xaxis.label.set_rotation(90) #rotate x-axis labels by 90 degrees for better readability
    ax.yaxis.label.set_rotation(0) #rotate y-axis labels by 0 degrees for better readability
    ax.yaxis.label.set_ha('right') #align y-axis labels to the right for better readability

plt.tight_layout() #adjust layout to prevent overlap of labels and titles, 
#tight_layout automatically adjusts the subplot parameters to give specified padding
plt.gcf().subplots_adjust(wspace=0, hspace=0) #adjust the spacing between subplots to 0, wspace for width and hspace for height,
plt.show() #show scatter matrix

#split dataset into training and testing sets
X = df.iloc[:,[0,1]].to_numpy() #independent variables (engine size and fuel consumption combined)
print(X) #check independent variables
y = df.iloc[:,[2]].to_numpy() #dependent variable (co2 emissions)
print(y) #check dependent variable

#standardize X to have mean = 0 and std = 1 for better performance of the model
std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(X)
y_std = std_scaler.fit_transform(y) #standardize y as well to have mean = 0 and std = 1 for better performance of the model

#split dataset
X_train, X_test, y_train, y_test = train_test_split(X_std, y_std, test_size=0.2, random_state=42) #split dataset into training and testing sets, 
#test_size = 0.2 for 20% testing data, random_state = 42 for reproducibility

regressor = LinearRegression() #create linear regression model
regressor.fit(X_train, y_train) #fit the model to the training data

coef_ = regressor.coef_ #get coefficients of the model
intercept_ = regressor.intercept_ #get intercept of the model

print("Coefficients: ", coef_) #print coefficients
print("Intercept: ", intercept_) #print intercept

mean_ = std_scaler.mean_ #get mean of the standardized X
std_devs_ = np.sqrt(std_scaler.var_) #get standard deviations of the standardized X by taking the square root of the variance

# Ensure X1, X2, and y_test have compatible shapes for 3D plotting
# If X_test is 2D, select the first and second columns as X1 and X2; otherwise, handle as 1D.
X1 = X_test[:, 0] if X_test.ndim > 1 else X_test
X2 = X_test[:, 1] if X_test.ndim > 1 else np.zeros_like(X1) # If X_test is 1D, set X2 to an array of zeros with the same shape as X1
print(X_test.shape)

#we can write a for loop for this too:
#X1 = []
#X2 = []
#for i in range(len(X_test)):
#    if X_test.ndim > 1:
#        X1.append(X_test[i, 0]) #append first column of X_test to X1
#        X2.append(X_test[i, 1]) #append second column of X_test to X2
#    else:
#        X1.append(X_test[i]) #append X_test to X1 if X_test is 1D
#        X2.append(0) #append 0 to X2 if X_test is 1D

# Create a mesh grid for plotting the regression plane,
#meshgrid is a function in numpy that creates a rectangular grid out of two given linear spaces. 
# It takes two 1D arrays as input and returns two 2D arrays representing the grid of points for each variable.
x1_surf, x2_surf = np.meshgrid(np.linspace(X1.min(), X1.max(), 100), #what this does is it 
                               #creates a grid of points for X1 and X2 to plot the regression plane, 100 points for each variable for smoothness
                               np.linspace(X2.min(), X2.max(), 100))

y_surf = intercept_ + coef_[0,0]*x1_surf + coef_[0,1]*x2_surf #calculate the predicted y values 
#for the regression plane using the coefficients and intercept in the form of a linear equation: 
# y = b0 + b1*x1 + b2*x2, where b0 is the intercept and b1, b2 are the coefficients for x1 and x2 respectively

# Predict y values using trained regression model to compare with actual y_test for above/below plane colors
y_pred = regressor.predict(X_test.reshape(-1, 1)) if X_test.ndim == 1 else regressor.predict(X_test) 
#if X_test is 1D, reshape it to 2D for prediction; otherwise, use it as is for prediction
above_plane = y_test >= y_pred #above_plane is True where actual y_test is greater than or equal to the prediction.
below_plane = y_test < y_pred #below_plane is True where actual y_test is below the prediction
above_plane = above_plane[:,0] #since above_plane is a 2D array with one column, we select the first column to get a 1D array of boolean values 
#for coloring the points above the plane
below_plane = below_plane[:,0] #since below_plane is a 2D array with one column, we select the first column to get a 1D array of boolean values
#for coloring the points below the plane

# Plotting
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111, projection='3d') # Create a 3D scatter plot of the test data points, 111 means 1 row, 1 column, and this is the first plot, projection='3d' for 3D plot 
#colored by whether they are above or below the regression plane

# Plot the data points above and below the plane in different colors
ax.scatter(X1[above_plane], X2[above_plane], y_test[above_plane],  label="Above Plane",s=70,alpha=.7,ec='k')
ax.scatter(X1[below_plane], X2[below_plane], y_test[below_plane],  label="Below Plane",s=50,alpha=.3,ec='k')
#.scatter() is used to create a scatter plot of the data points, where X1 and X2 are the independent variables, y_test is the dependent variable,
#label is used to label the points for the legend, s is the size of the points, alpha is the transparency of the points, and ec is the edge color of the points

# Plot the regression plane
ax.plot_surface(x1_surf, x2_surf, y_surf, color='k', alpha=0.21,label='plane')

# Set view and labels
ax.view_init(elev=10)

ax.legend(fontsize='x-large',loc='upper center')
# Remove x, y, z ticks for a cleaner look since we are focusing on the plane and 
# the relative positions of the points
# x, y, z ticks are the marks on the axes that indicate the scale of the plot, 
# and removing them can help to focus on the overall shape of the data and the regression plane 
# without being distracted by the specific values on the axes.
ax.set_xticks([]) 
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(None, zoom=0.75) # Adjust the aspect ratio of the plot to make it more visually appealing,
# where zoom controls the overall size of the plot and None allows for automatic scaling of the axes
ax.set_xlabel('ENGINESIZE', fontsize='xx-large')
ax.set_ylabel('FUELCONSUMPTION', fontsize='xx-large')
ax.set_zlabel('CO2 Emissions', fontsize='xx-large')
ax.set_title('Multiple Linear Regression of CO2 Emissions', fontsize='xx-large')
plt.tight_layout()
plt.show()

#Instead of making a 3D plot, which is difficult to interpret, you can look at 
# vertical slices of the 3D plot by plotting each variable separately as a best-fit line
# using the corresponding regression parameters.

plt.scatter(X_train[:,0], y_train,  color='blue')
plt.plot(X_train[:,0], coef_[0,0] * X_train[:,0] + intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show() #how co2 emissions change with engine size, while keeping fuel consumption constant

plt.scatter(X_train[:,1], y_train,  color='blue')
plt.plot(X_train[:,1], coef_[0,1] * X_train[:,1] + intercept_[0], '-r')
plt.xlabel("FUELCONSUMPTION_COMB_MPG")
plt.ylabel("Emission")
plt.show() #how co2 emissions change with fuel consumption, while keeping engine size constant

mse = mean_squared_error(y_test, y_pred) # squared difference between actual and predicted values of all data points, mse of 0 means perfect fit, higher mse means worse fit, and mse can be used to compare different models (lower mse is better) 
r2 = r2_score(y_test, y_pred) # checks the proportion of the variance. an r2 score of 1 -> perfect fit, 0 -> no fit, negative -> worse than a horizontal line fit
n = len(y_test) # gets all y values in the test set, which is used to calculate the adjusted R^2 score
k = X_test.shape[1] # gets number of independdent variables (features) in the test set, which is used to calculate the adjusted R^2 score
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1)) # Adjusted R^2 score is a modified version of R^2 that takes into account the number of independent variables in the model

print("Mean Squared Error: ", mse)
print("R^2 Score: ", r2)
print("Adjusted R^2 Score: ", adj_r2) 