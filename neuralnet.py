from sklearn.neural_network import MLPClassifier

# Input features: [sensor1, sensor2]
X = [
    [0, 0],  # no input
    [0, 1],  # sensor2 triggered
    [1, 0],  # sensor1 triggered
    [1, 1]   # both sensors triggered
]

# Labels: 0 = Stop Car, 1 = Move Car
Y = [0, 1, 1, 0]  # XOR logic

# Define a neural network with one hidden layer of 5 neurons
model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=42)
model.fit(X, Y)

# Predict on new input: both sensors triggered
prediction = model.predict([[1, 1]])

# Output decision
print("Move Car" if prediction[0] == 1 else "Stop Car")
