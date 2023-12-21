Self-Organizing Map (SOM) Clustering with Kohonen's Algorithm
Overview
This Python script implements Kohonen's Self-Organizing Map (SOM) clustering algorithm to partition a synthetic dataset into distinct clusters. The generated clusters are visualized in a 3D space, where data points are colored based on their assigned clusters, and centroids are marked with black 'X' markers. This project is inspired by the Kohonen_SOM project on GitHub.

Dependencies
NumPy: Used for efficient array manipulation.
Matplotlib: Employed for data visualization.
Usage
Install the required dependencies:

Copy code
pip install numpy matplotlib
Run the script:

Copy code
python kohonen_clustering.py
View the generated 3D plot displaying clustered data points and centroids.

Project Details
Data Generation
The script creates a synthetic dataset with four clusters, each representing a distinct color in the RGB space. Each cluster is generated using NumPy's random functions.

Kohonen's Algorithm
The SOM clustering algorithm is applied for a specified number of epochs. The centroids are initialized using random data points, and during each epoch, they are updated based on the proximity of data points. The learning rate decreases over epochs to fine-tune the clustering.

Visualization
The resulting clusters and centroids are visualized in a 3D plot using Matplotlib. Data points are colored based on their assigned clusters, creating an informative and visually appealing representation of the clustering results.

Inspiration
This project draws inspiration from the Kohonen_SOM project on GitHub. The visualization techniques and clustering approaches from their work have been adapted and expanded upon for this project.

Acknowledgment
Special thanks to the contributors of Kohonen_SOM for their impactful work, serving as a valuable reference and inspiration for enhancing the implementation and visualization aspects of this project.
