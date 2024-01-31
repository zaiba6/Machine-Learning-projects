# Self-Organizing Map (SOM) Clustering with Kohonen's Algorithm (Kohonen.py) <br>

This Python script implements Kohonen's Self-Organizing Map (SOM) clustering algorithm to partition a synthetic dataset into distinct clusters. The generated clusters are visualized in a 3D space, where data points are coloured based on their assigned clusters, and centroids are marked with black 'X' markers. 

## Dependencies
NumPy: Used for efficient array manipulation. <br>
Matplotlib: Employed for data visualization.

## Project Details
Data Generation
The script creates a synthetic dataset with four clusters, each representing a distinct colour in the RGB space. Each cluster is generated using NumPy's random functions.

## Kohonen's Algorithm
The SOM clustering algorithm is applied for a specified number of epochs. The centroids are initialized using random data points, and during each epoch, they are updated based on the proximity of data points. The learning rate decreases over epochs to fine-tune the clustering.

## Visualization
The resulting clusters and centroids are visualized in a 3D plot using Matplotlib. Data points are coloured based on their assigned clusters, creating an informative and visually appealing representation of the clustering results.

## Inspiration
This project draws inspiration from the Kohonen_SOM project on GitHub. Their work's visualization techniques and clustering approaches have been adapted and expanded upon for this project.

# Kohonen Self Organizing Maps for Penguin Clustering (SOM_penguins.py) <br>

## Overview
This Python script utilizes the MiniSom library to implement a Self Organizing Map (SOM) for clustering penguins based on their features. The dataset used for this project is obtained from a CSV file containing penguin data. The SOM is trained to identify patterns and group similar penguins together.

## Dependencies
The following Python libraries were used: 
- `minisom`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `pandas`

## Dataset
The penguin dataset was downloaded from Kaggle.com. 
https://www.kaggle.com/code/youssefaboelwafa/clustering-penguins-species-k-means-clustering/input 

## Data Preprocessing
1. Any rows with missing values are removed from the dataset.
2. Dummy values are created for the 'sex' attribute, replacing it with binary values (0 or 1).
3. The data is normalized using the StandardScaler from scikit-learn to ensure all features are within a comparable range.

## SOM Initialization
The script calculates the grid size for the SOM based on the number of samples in the dataset and the number of features. The SOM is then initialized using MiniSom with specified parameters such as grid size, input length, sigma, learning rate, neighborhood function, and random seed.

## SOM Training
The SOM is trained on the preprocessed data using the PCA initialization method. The training is verbose, displaying progress over the specified number of iterations.

## Clustering and Visualization
1. The script computes the quantization error, providing a measure of how well the SOM has learned the input data.
2. Each data point is mapped to its winning neuron, and cluster labels are assigned based on the coordinates of the winning neurons.
3. The results are visualized in a scatter plot, where each point represents a penguin, colored and organized according to its assigned cluster label.
