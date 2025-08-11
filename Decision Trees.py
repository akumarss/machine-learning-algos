from sklearn.tree import DecisionTreeClassifier

# Features: [Is it raining?, Is it cold?]
X = [[1, 1],  # raining & cold
     [1, 0],  # raining & warm
     [0, 1],  # not raining & cold
     [0, 0]]  # not raining & warm

# Labels: 0 = T-shirt, 1 = Jacket, 2 = Umbrella
Y = [2, 2, 1, 0]

# Train the model
model = DecisionTreeClassifier()
model.fit(X, Y)

# Predict what to wear when it's raining and cold
prediction = model.predict([[1, 1]])

# Output interpretation
outfits = {0: "T-shirt", 1: "Jacket", 2: "Umbrella"}
print("Wear:", outfits[prediction[0]])
