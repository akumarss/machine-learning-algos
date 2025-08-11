import numpy as np
from sklearn.cluster import DBSCAN

# Use Case: Detects fraud transactions.
# Finding Outliers in Transactions using DBSCAN (Density-Based Spatial Clustering)

# DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups points
# that are closely packed together (high density) and marks points that lie alone in
# low-density regions as outliers (noise).

# Input data: 2D points
X = np.array([
    [1, 2],     # A
    [2, 3],     # B
    [3, 3],     # C
    [8, 7],     # D
    [8, 8],     # E
    [25, 80]    # F (very far away)
])

# Apply DBSCAN clustering
db = DBSCAN(eps=3, min_samples=2).fit(X)
#eps=3: Maximum distance to be considered "neighbors"
#min_samples=2: Minimum number of neighbors (including the point itself) required to
# form a dense region

# Output the cluster labels (-1 = noise/outlier)
print("Cluster Labels:", db.labels_)

# Cluster 0: Points around (1–3, 2–3)
# Cluster 1: Points around (8, 7–8)
# Label -1: Outlier (e.g., [25, 80])