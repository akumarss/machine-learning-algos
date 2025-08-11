import numpy as np
from sklearn.cluster import KMeans

# Sample RGB color data: [R, G, B]
X = np.array([
    [255, 0, 0],     # Red
    [250, 10, 10],   # Red-ish
    [0, 255, 0],     # Green
    [0, 250, 10],    # Green-ish
    [0, 0, 255],     # Blue
    [10, 10, 250]    # Blue-ish
])

# Train KMeans with 3 clusters
model = KMeans(n_clusters=3, random_state=0)
model.fit(X)

# Predict the color group for a new color
new_color = [[5, 5, 240]]
prediction = model.predict(new_color)

print("Color group:", prediction[0])
