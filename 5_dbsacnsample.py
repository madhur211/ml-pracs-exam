import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # For rainbow colors
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# --- 1. Create Sample Data ---
# X = coordinates, y = the "true" group (we won't use y)
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# --- 2. Run DBSCAN ---
# eps=0.2: Max distance between two points to be neighbors.
# min_samples=5: Min number of neighbors a point needs to be a "core" point.
clustering = DBSCAN(eps=0.2, min_samples=5)
labels = clustering.fit_predict(X) # Run the algorithm and get labels

# --- 3. Plot the Results ---
unique_labels = set(labels)
markers = ['o', 's', '^', 'v', 'D', '*', 'P', 'X', '+'] # Marker styles
colors = cm.rainbow(np.linspace(0, 1, len(unique_labels))) # Rainbow colors

for k, color in zip(unique_labels, colors):
    # Find all points belonging to this cluster 'k'
    cluster_points = X[labels == k]
    
    if k == -1:
        # Plot noise points as black 'x'
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    marker='x', color='black', s=30, label='Noise')
    else:
        # Plot a core cluster
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    marker=markers[k % len(markers)],
                    color=color, s=50, label=f'Cluster {k}', edgecolor='k')

plt.title('DBSCAN on make_moons Dataset')
plt.xlabel('Feature 1'); plt.ylabel('Feature 2')
plt.legend(); plt.grid(True); plt.show()