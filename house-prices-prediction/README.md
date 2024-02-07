# Bagging with Lasso Models: A Machine Learning Experiment <br />
This project contains code for a machine-learning experiment involving the use of many lasso regressions to predict house prices. 
The primary objective is to demonstrate how the error of predictions typically decreases as the number of models used in each prediction increases through bagging. 
The experiment involves creating and training 100 Lasso models, each on a different random sample of 900 observations from the original 1800 data points in the training set.

## Requirements <br />
The following libraries are needed in Python: NumPy, ScikitLearn, Matplotlib, Pandas

## Data Preprocessing <br />
The dataset is split into training, testing and validation sets. 
The training set includes the first 1800 rows, the Validation set includes rows 600 rows, and the testing set includes the remaining 500 rows. 

## Code Integration <br />
The heart of the experiment lies in the script bagging_lasso_experiment.py. 

1. **Random Sampling:**
- Random sampling is performed using the random package.
- To ensure reproducibility, the random number generator is fixed using random.seed(i), where i corresponds to the model number.

2. **Lasso Model Training:**
- Lasso models are trained with a penalty parameter (alpha) set to 0.05.

3. **Bagging Approach:**
- Predictions are based on aggregating the outputs of multiple Lasso models.
- The first prediction is derived from the output of the first Lasso model.
- Subsequent predictions involve averaging the outputs of an increasing number of Lasso models.

4. **Error Calculation:**
- Prediction errors are calculated for each iteration of the bagging process.
- The code concludes with a visualization that illustrates how the prediction error, measured in standard deviation, changes as more models are averaged during the bagging process.

## Conclusion <br />
It can be seen that the error typically decreases as the number of models used in each prediction increases (bagging). 
The experiment aims to visualize the reduction in error as a function of the number of models averaged to make the prediction.
A decision tree model for individual predictions and its information is printed and visualized.

## Sources <br /> 
This is a citation for the dataset that I used for this project.  
Mitchell, T. M. (n.d.). In Machine Learning Third Edition Resources. Rotman School of Management, University of Toronto. https://www-2.rotman.utoronto.ca/~hull/MLThirdEditionFiles/mlindex1_3rdEd.html 
