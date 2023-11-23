#Polynomial Regression Analysis
This project utilizes polynomial regression to model the intricate relationship between age and salary. 
Two polynomial regressions of 2nd and 5th degrees are employed to determine the most effective model for the given dataset.

#Data Preprocessing
The dataset is evenly split into training and testing sets, with each comprising 50% of the data. 
The scikit-learn class 'PolynomialFeatures' is then utilized to transform the input features (Age) into polynomial features of specified degrees (2, 5) for both training and testing.

#Model Implementation
The model is based on linear regression and is trained using the transformed polynomial features alongside the corresponding target variables. 
Training and testing errors are calculated by determining the standard deviation of each dataset to evaluate the model's performance.

#Visualization
To provide a visual understanding, the data is displayed through a scatter plot, with the regression lines plotted separately. Each section of the code generates distinct visualizations for the various regressions.

#Conclusion
After careful examination of the graphs for both the 2nd and 5th-degree polynomial regressions, it is evident that the 2nd-degree polynomial model is the most suitable for this dataset. 
The linear regression model appears to be underfitting, indicating a generalized trend but lacking specificity. 
On the other hand, the 5th-degree polynomial regression exhibits signs of overfitting, capturing specific data trends but risking accuracy in predictions. 
The low training error for the 5th degree reinforces this notion, indicating an overfitting scenario. 
In contrast, the 2nd-degree polynomial regression strikes a balance, demonstrating accuracy and effective generalization of the data.

In conclusion, the 2nd-degree polynomial regression model emerges as the optimal choice for accurately modelling the relationship between age and salary in this dataset.
