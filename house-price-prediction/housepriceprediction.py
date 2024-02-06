# Importing necessary packages
import os # enable interaction with the operating system
import pandas as pd # python's data handling package
import numpy as np # python's scientific computing package
import matplotlib.pyplot as plt # python's plotting package
import random # random numbers package
from sklearn.linear_model import Lasso # Lasso regression

#Both features and target have already been scaled: mean = 0; SD = 1
DATA_FOLDER = '/Users/zaiba/Desktop/ML in bizz'
FILENAME = 'Houseprice_data_scaled.csv'
data = pd.read_csv(os.path.join(DATA_FOLDER, FILENAME)) 

# First 1800 data items are training set; the next 600 are the validation set
train = data.iloc[:1800] 
test = data.iloc[1800:2400]
# Creating the "X" and "y" variables. We drop sale price from "X"
X_train, X_test = train.drop('Sale Price', axis=1), test.drop('Sale Price', axis=1)
y_train, y_test = train[['Sale Price']], test[['Sale Price']] 

#empty array for all the bagged predictions 
bagged_predictions = []
#empty array for the prediction errors
prediction_errors = []

#making 100 Lasso models
for i in range(100):
    random.seed(i)
    
    #randomly sample 900 data points from the original 1800 data points in the train set
    sample_indices = random.sample(range(1800), 900)
    X_sampled = X_train.iloc[sample_indices]
    y_sampled = y_train.iloc[sample_indices]

    #create and fit a Lasso model with alpha
    lasso_model = Lasso(alpha = 0.05)
    lasso_model.fit(X_sampled, y_sampled)

    #make predictions using the model
    prediction = lasso_model.predict(X_test)

    #store the prediction in the bagged_predictions list
    bagged_predictions.append(prediction)

    #calculate the error for the current prediction
    error = np.std(y_test['Sale Price'] - np.mean(bagged_predictions, axis=0))
    
    #append the error to the list of errors
    prediction_errors.append(error)

#ploting the errors 
plt.plot(range(1, 100 + 1), prediction_errors)
plt.xlabel('Number of Models Averaged')
plt.ylabel('Prediction Error (Standard Deviation)')
plt.title('Bagging Error vs. Number of Models')
plt.show()
