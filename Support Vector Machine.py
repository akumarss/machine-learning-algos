from sklearn import svm

# Features: [has_loops, is_connected]
X = [
    [1, 1],  # likely cursive
    [1, 0],  # likely cursive
    [0, 1],  # likely printed
    [0, 0]   # likely printed
]

# Labels: 1 = Cursive, 0 = Printed
Y = [1, 1, 0, 0]

# Train the SVM model
model = svm.SVC()
model.fit(X, Y)

# Predict for a new handwriting sample
new_sample = [[1, 0]]
prediction = model.predict(new_sample)

# Output result
print("Cursive" if prediction[0] == 1 else "Printed")
