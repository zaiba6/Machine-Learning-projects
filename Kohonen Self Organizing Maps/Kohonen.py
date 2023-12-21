import numpy as np
import matplotlib.pyplot as plt

# Generate the data for 4 clusters
cluster1 = np.random.rand(250, 3) * 0.5
cluster2 = np.random.rand(250, 3) * 0.5 + [1, 0, 0]
cluster3 = np.random.rand(250, 3) * 0.5 + [0, 1, 0]
cluster4 = np.random.rand(250, 3) * 0.5 + [0, 0, 1]

# Concatenate the clusters to form the dataset
dataset = np.vstack([cluster1, cluster2, cluster3, cluster4]).astype(np.float32)

# Parameters for Kohonen's clustering algorithm
epochs = 10  # Number of iterations for clustering
alpha = np.flip(np.linspace(0.1, 1, epochs))  # Decreasing learning rate

# Initialize centroids using random data points
centroids = dataset[np.random.choice(dataset.shape[0], size=4, replace=False)]

# Kohonen's clustering algorithm
for j in range(epochs):
    # For each data point
    for i in range(dataset.shape[0]):
        # Calculate distance between the data point and each centroid
        dist = np.sum((centroids - dataset[i, None])**2, axis=1)

        # Find the closest centroid
        idx = np.argmin(dist)

        # Update the closest centroid towards the data point
        centroids[idx] += alpha[j] * (dataset[i] - centroids[idx])

# Assign colors to clusters with lighter colors
colors = np.zeros(dataset.shape[0])
for i in range(dataset.shape[0]):
    dist = np.sum((centroids - dataset[i])**2, axis=1)
    colors[i] = np.argmin(dist)

# Plot the data and cluster centroids with lighter colors
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection='3d')
scatter = ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c=colors, cmap='viridis', alpha=0.5, s=100, marker='.')
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='black', marker='X', s=200, label='Centroids', alpha=1)
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
ax.legend()

# Create a custom legend to differentiate between data points and centroids
legend_labels = [plt.Line2D([0], [0], marker='.', color='w', markerfacecolor='white', markersize=10, label='Data Points'),
                 plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='black', markersize=10, label='Centroids')]
ax.legend(handles=legend_labels)

plt.show()

print("Cluster Centroids:")
print(centroids)


#SIDE NOTE i used this github as inspo as it was really cool and they presented it with colours and i love colours, 
#so full credit to their github for inspo
#https://github.com/Kursula/Kohonen_SOM 
