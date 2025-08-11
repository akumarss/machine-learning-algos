import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data: [Square Feet] and corresponding [Price]
X = np.array([[1000], [1500], [2000], [2500]])  # Features: size of the house
Y = np.array([50000, 75000, 100000, 125000])    # Labels: price of the house

# Create and train the model
model = LinearRegression()
model.fit(X, Y)

# Predict the price for a new house with 1800 square feet
predicted_price = model.predict([[1800]])

# Output the prediction
print("Predicted Price:", predicted_price[0])
