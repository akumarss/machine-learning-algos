from sklearn.ensemble import GradientBoostingRegressor

# Simple time-series-like input: e.g., day number → stock price
X = [[1], [2], [3], [4], [5]]       # e.g., days
Y = [100, 200, 300, 400, 500]       # e.g., stock prices

# Train the Gradient Boosting Regressor
model = GradientBoostingRegressor()
model.fit(X, Y)

# Predict the price for day 6
prediction = model.predict([[6]])

# Output the predicted price
print("Predicted Stock Price:", prediction[0])
