from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd


df = pd.read_csv(r'/Users/zaiba/Documents/GitHub/Machine-Learning-projects/Kohonen Self Organizing Maps/penguins.csv') 

#notice there are a lot of null values, so i remove them hehe
df = df.dropna()

#now making dummy values for the sex of the penguins, and give them values of 0,1
df = pd.get_dummies(df).drop("sex_.", axis=1)
print(df.head())

# Normalize the data to a range between 0 and 1
scaler = StandardScaler()
df = scaler.fit_transform(df)

# Calculate the grid size based on the number of samples

#how many penguins
num_rows = df.shape[0]
#how many features we are looking at
num_columns = df.shape[1]

#finding the grid size
grid_size = int(np.sqrt(5 * num_rows))

# Initialize the som_shape tuple
som_shape = (grid_size, grid_size)
#print(num_rows, num_columns, som_shape)

# Initialize the SOM
som = MiniSom(som_shape[0], som_shape[1], input_len=num_columns, sigma=1.5, learning_rate = 1.5,
              neighborhood_function='gaussian', random_seed=123)

# Initialize SOM weights using PCA
som.pca_weights_init(df)

# Train the SOM
som.train(df,1000, verbose=True) 

# Get the U-matrix (unified distance matrix)
#u_matrix = som.distance_map()

# Visualize the U-matrix
#plt.figure(figsize=(10, 8))
#plt.pcolor(u_matrix, cmap='viridis')
#plt.colorbar()
#plt.title('Unified Distance Matrix (U-matrix)')
#plt.show()

# Compute the quantization error
quantization_error = som.quantization_error(df)

# Map each data point to its winning neuron
winner_coordinates = np.array([som.winner(x) for x in df])

# Convert coordinates to integers by rounding
winner_coordinates = np.round(winner_coordinates).astype(int)

# Assign cluster labels based on the winning neuron coordinates
clusters = np.ravel_multi_index(winner_coordinates.T, som_shape)

# Create a scatter plot with colors based on cluster labels
plt.figure(figsize=(10, 8))
scatter = plt.scatter(winner_coordinates[:, 0], winner_coordinates[:, 1], c=clusters, cmap='viridis')
plt.colorbar(scatter, label='Cluster Labels')
plt.title('Penguin Clusters')
plt.show()
