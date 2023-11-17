import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate sample data (replace with your own data)
X = np.random.rand(100)  # X coordinates
Y = np.random.rand(100)  # Y coordinates
Z = np.random.rand(100)  # Z coordinates

# Create a 3D figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a 3D scatter plot
ax.scatter(X, Y, Z, c='b', marker='o', label='Data Points')

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot')

# Show the legend
ax.legend()

# Display the 3D plot
plt.show()

