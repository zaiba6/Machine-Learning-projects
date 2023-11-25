# Decision Tree Regressor <br>

This project implements a decision tree approach to predict house prices using the data provided in the `Data_DecisionTree.xlsx` file. The decision tree is constructed based on the Quality and Area features, with the goal of maximizing the expected Mean-Squared Error (MSE) for each threshold value.

## Python Code <br>

The Python code is implemented in the `decisionTree.py` file. The key components include:

- **Data Loading**: The data is loaded from the Excel file using the Pandas library.

- **Data Splitting**: The dataset is split into training, validation, and test sets.

- **Threshold Evaluation**: The script iterates over potential thresholds for each feature (Quality and Area), calculates the expected MSE for each threshold, and determines the threshold that maximizes the expected MSE for each feature.

- **Decision Tree Model**: A Decision Tree Regressor model is trained on the training set using the features Quality and Area.

- **Model Evaluation**: The model is evaluated on the training and validation sets, and the Mean-Squared Error (MSE) is calculated.

- **Tree Visualization**: The script visualizes the decision tree using the `plot_tree` function from scikit-learn.

## Results <br>

- **Mean-Squared Error (MSE):**
  ```
  Qual MSE: 0.001494626964920379
  Area MSE: 0.0016399055483021327
  ```
  These values indicate how well the model fits the training data. Lower MSE values suggest better model performance.

- **Thresholds for Root Node:**
  ```
  Threshold for Qual: 7.5
  Threshold for Area: 7.5
  Root Feature: Qual
  ```
  The decision tree's root node is split based on the Quality feature with a threshold of 7.5. This means that when making predictions, the model first evaluates the Quality feature, and if it's greater than 7.5, it goes to the right branch; otherwise, it goes to the left branch.

- **Price Predictions:**
  ```
  Price Predictions: [0.12546719 0.16556307 0.12546719 0.16556307 0.16556307 0.16556307 0.16556307 0.12546719]
  ```
  These are the predicted prices for the first eight samples in your validation set.

The decision tree will be visualized and displayed using Matplotlib.
