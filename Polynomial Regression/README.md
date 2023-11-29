# Polynomial Regression Analysis <br />
This project utilizes polynomial regression to model the intricate relationship between age and salary. 
Two polynomial regressions of 2nd and 5th degrees are employed to determine the most effective model for the given dataset.

## Data Preprocessing <br />
The dataset is evenly split into training and testing sets, with each comprising 50% of the data. 
The scikit-learn class 'PolynomialFeatures' is then utilized to transform the input features (Age) into polynomial features of specified degrees (2, 5) for both training and testing.

## Model Implementation <br />
The model is based on linear regression and is trained using the transformed polynomial features alongside the corresponding target variables. 
Training and testing errors are calculated by determining the standard deviation of each dataset to evaluate the model's performance.

### Training and Testing Errors <br />
After model training, the project evaluates its performance on both the training and testing datasets. 
Here are the training and testing errors for the 2nd-degree and 5th-degree polynomial regressions:

#### Degree 2 Polynomial Regression: <br />
**Training Error:** 32,932.08  
**Testing Error:** 33,553.77

#### Degree 5 Polynomial Regression: <br />
**Training Error:** 12,902.20  
**Testing Error:** 38,793.93

## Visualization <br />
To provide a visual understanding, the data is displayed through a scatter plot, with the regression lines plotted separately.
Each section of the code generates distinct visualizations for the various regressions.

## Conclusion <br />
After careful examination of the graphs for both the 2nd and 5th-degree polynomial regressions, it is evident that the 2nd-degree polynomial model is the most suitable for this dataset. 

**Degree 2 Model:** 
Demonstrates a balanced performance with moderate errors on both training and testing sets.
The errors are reasonably close, indicating effective generalization without overfitting or underfitting.

**Degree 5 Model:** 
Exhibits low training error but a significantly higher testing error.
Suggests overfitting, where the model fits the training data too closely but struggles to generalize to new, unseen data.

## Sources <br /> 
This is a citation for the dataset that I used for this project 
Mitchell, T. M. (n.d.). In Machine Learning Third Edition Resources. Rotman School of Management, University of Toronto. https://www-2.rotman.utoronto.ca/~hull/MLThirdEditionFiles/mlindex1_3rdEd.html 
