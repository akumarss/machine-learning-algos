from sklearn.neighbors import KNeighborsClassifier

# Features: [explosions_count, love_scenes_count]
X = [
    [5, 3],  # likely Action
    [4, 2],  # likely Action
    [3, 5],  # likely Romance
    [2, 4]   # likely Romance
]

# Labels: Movie genres
Y = ["Action", "Action", "Romance", "Romance"]

# Train the KNN model with 2 nearest neighbors
model = KNeighborsClassifier(n_neighbors=2)
model.fit(X, Y)

# Predict genre for a movie with 4 explosions and 3 love scenes
prediction = model.predict([[4, 3]])

# Output result
print("Recommended Movie Genre:", prediction[0])
