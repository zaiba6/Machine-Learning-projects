import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Read data from Excel
data = pd.read_excel(r'/Users/zaiba/Documents/GitHub/Machine-Learning-projects/DecisionTreeRegressor/Data_DecisionTree.xlsx') 
print(data.head())

# Split the data into train, val, and test sets
train = data.iloc[:1800]
val = data.iloc[1801:2400]
test = data.iloc[2401:]

features = ['Qual', 'Area']

#thresholds for 'Qual' and 'Area'
max_expected_mse = {}
best_threshold = {}

for feature in features:
    #thresholds
    potential_thresholds = np.unique(train[feature])

    #Initialize variables for the best threshold and minimum MSE
    best_threshold[feature] = None
    max_mse = -1

    #Iterate over potential thresholds
    for threshold in potential_thresholds:
        #Split the data based on the threshold
        split_below = train[train[feature] <= threshold]
        split_above = train[train[feature] > threshold]

        #Check if splits are not empty
        if len(split_below) == 0 or len(split_above) == 0:
            continue  
            
        #Calculate MSE for each split
        mse_below = mean_squared_error(split_below['Price'], np.full_like(split_below['Price'], np.mean(split_below['Price'])))
        mse_above = mean_squared_error(split_above['Price'], np.full_like(split_above['Price'], np.mean(split_above['Price'])))

        #Calculate the weighted average MSE
        total_samples = len(train)
        weight_below = len(split_below) / total_samples
        weight_above = len(split_above) / total_samples
        expected_mse = weight_below * mse_below + weight_above * mse_above

        #Update the best threshold if the expected MSE is higher
        if expected_mse > max_mse:
            max_mse = expected_mse
            best_threshold[feature] = threshold

    #Store the results 
    max_expected_mse[feature] = max_mse


#decision tree model
model = DecisionTreeRegressor(max_depth = 3)
model.fit(train[features], train['Price'])

#Accessing info about the actual root node
tree = model.tree_
root_feature_index = tree.feature[0]  # Assuming root node is at index 0
root_feature = features[root_feature_index]
root_threshold = tree.threshold[0]

#Evaluating the model on the training set
train_predictions = model.predict(train[features])
train_mse = mean_squared_error(train['Price'], train_predictions)
print(f"Qual MSE: {train_mse}")

#Evaluating the model on the validation set
val_predictions = model.predict(val[features])
val_mse = mean_squared_error(val['Price'], val_predictions)
print(f"Area MSE: {val_mse}")

#Printing information about the actual root node
print(f"Threshold for {root_feature}: {root_threshold}")

#Print the other feature as well
other_feature_index = 1 - root_feature_index  # Assuming only two features
other_feature = features[other_feature_index]
other_threshold = tree.threshold[0]

print(f"Threshold for {other_feature}: {other_threshold}")
print(f"Root Feature: {root_feature}")

# Print thresholds for each branch of the tree
def print_tree_thresholds(node, depth=0):
    if tree.feature[node] == -2:  # Leaf node
        return

    feature_index = tree.feature[node]
    feature_name = features[feature_index]
    threshold = tree.threshold[node]

    print(f"Depth {depth}: Node {node}, Split on {feature_name}, Threshold: {threshold}")

    # Recursively print thresholds for left and right children
    print_tree_thresholds(tree.children_left[node], depth + 1)
    print_tree_thresholds(tree.children_right[node], depth + 1)

# Start printing thresholds from the root node
print_tree_thresholds(0)

# Obtain price predictions for the validation set
print("Price Predictions:")
print(val_predictions[:8])

# Ploting the decision tree 
plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=features, filled=True, rounded=True, fontsize=10)
plt.show()

