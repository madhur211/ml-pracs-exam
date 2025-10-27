import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Get user input
try:
    eps = float(input("Enter epsilon (ε) (e.g., 1.9): "))
    min_samples = int(input("Enter minPts (e.g., 4): "))
except:
    print("Invalid input"); exit()

# Get points
points = []
print("\nEnter points as 'x, y'. Type 'done' to finish.")
while True:
    line = input("Point: ")
    if line.lower() == 'done': break
    try:
        x, y = map(float, line.split(','))
        points.append([x, y])
    except:
        print("Use 'x, y' format")

if not points:
    print("No points"); exit()

# Run DBSCAN
X = np.array(points)
labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)

# Show results
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"\nClusters: {n_clusters}, Noise: {list(labels).count(-1)}")

# Plot
plt.figure(figsize=(9, 7))
for label in set(labels):
    if label == -1:
        mask = labels == label
        plt.scatter(X[mask, 0], X[mask, 1], s=30, c='black', marker='x', label='Noise')
    else:
        mask = labels == label
        plt.scatter(X[mask, 0], X[mask, 1], s=50, edgecolor='k', label=f'Cluster {label}')

plt.title(f'DBSCAN (ε={eps}, minPts={min_samples})')
plt.xlabel('X'); plt.ylabel('Y')
plt.legend(); plt.grid(True); plt.show()