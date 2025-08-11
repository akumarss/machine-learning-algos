from sklearn.ensemble import RandomForestClassifier

# Features: [fever, cough, fatigue]
X = [[1, 1, 1],   # likely flu
     [1, 0, 1],   # likely flu
     [0, 1, 0],   # likely cold
     [0, 0, 0]]   # likely cold

# Labels: 1 = Flu, 0 = Cold
Y = [1, 1, 0, 0]

# Train the model
model = RandomForestClassifier()
model.fit(X, Y)

# Predict the disease for a new patient
new_patient = [[1, 0, 1]]
prediction = model.predict(new_patient)

# Output result
print("Disease:", "Flu" if prediction[0] == 1 else "Cold")
