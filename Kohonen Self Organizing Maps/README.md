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

# Self-Organizing Map (SOM) species clustering penguins (SOM_penguins.py) <br>
