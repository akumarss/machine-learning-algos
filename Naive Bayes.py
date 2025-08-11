from sklearn.naive_bayes import GaussianNB

# Features: [has_link, has_spam_words, is_short_message]
X = [
    [1, 1, 1],  # likely spam
    [1, 0, 1],  # likely spam
    [0, 1, 0],  # not spam
    [0, 0, 0]   # not spam
]

# Labels: 1 = Spam, 0 = Not Spam
Y = [1, 1, 0, 0]

# Train the model
model = GaussianNB()
model.fit(X, Y)

# Predict for a new email
new_email = [[1, 1, 0]]
prediction = model.predict(new_email)

# Output result
print("Spam?" if prediction[0] == 1 else "Not Spam")
