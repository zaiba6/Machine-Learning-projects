import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

#this is data 
data = {
    'Age': [25, 55, 27, 35, 60, 65, 45, 40, 50, 30, 30, 26, 58, 29, 40, 27, 33, 61, 27, 48],
    'Salary': [135000, 260000, 105000, 220000, 240000, 265000, 270000, 300000, 265000, 105000, 166000, 78000, 310000, 100000, 260000, 150000, 140000, 220000, 86000, 276000]
}
#turning it into a data frame
df = pd.DataFrame(data)

#separating the data into its test and training sets
dataTrain, dataTest = df.iloc[:10], df.iloc[10:20]
xRaw, yRaw = dataTrain[['Age']], dataTrain[['Salary']]
xRawTest, yRawTest = dataTest[['Age']], dataTest[['Salary']]

# Polynomial regression with degree 2 and 5 
poly = PolynomialFeatures(degree = 2)
xPoly = poly.fit_transform(xRaw)
xPolyTest = poly.transform(xRawTest)

# fit the model
model = LinearRegression()
model.fit(xPoly, yRaw)
#printing the intercept and coefficiens
print("Intercept: ", model.intercept_[0])
print("Coefficients: ", model.coef_)

# model performance and printing their errors using std
yFitted = model.predict(xPoly)
errTrain = np.std(yRaw.values - yFitted, ddof=1)
print("Training Error: ", errTrain)

yTestFitted = model.predict(xPolyTest)
errTest = np.std(yRawTest.values - yTestFitted, ddof=1)
print("Testing Error: ", errTest)

# Plotting the original data
plt.scatter(xRaw, yRaw, color='blue')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
plt.title('Salary vs. Age')
plt.xlabel('Age')
plt.ylabel('Salary')

# Plot the regression polynomial line byt making a range of values of age, and uses the training information to fit it to out 
#dataset and dataframe we are working with
x_range = np.linspace(min(xRaw.values), max(xRaw.values), 100).reshape(-1, 1)
x_range_poly = poly.transform(x_range)
y_range_pred = model.predict(x_range_poly)
plt.plot(x_range, y_range_pred, color='red', linewidth=2)
plt.show()