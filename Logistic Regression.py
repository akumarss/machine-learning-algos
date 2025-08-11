from sklearn.linear_model import LogisticRegression

# Sample binary features: [has_link, has_spam_words, is_short_message]
X = [[1, 1, 1],   # likely spam
     [1, 0, 1],   # likely spam
     [0, 1, 0],   # not spam
     [0, 0, 0]]   # not spam

# Labels: 1 = Spam, 0 = Not Spam
Y = [1, 1, 0, 0]

# Create and train the model
model = LogisticRegression()
model.fit(X, Y)

# Predict for a new email
new_email = [[1, 1, 0]]
prediction = model.predict(new_email)

# Output the result
print("Spam" if prediction[0] == 1 else "Not Spam")
