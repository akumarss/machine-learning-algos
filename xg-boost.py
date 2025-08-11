import numpy as np
import xgboost as xgb

# Generate random data: 10 samples, 5 features
#X =
#[[0.12, 0.34, 0.56, 0.78, 0.90],
# [0.22, 0.11, 0.99, 0.45, 0.33],
# [0.56, 0.72, 0.81, 0.60, 0.44],
# ...
# (total 10 rows)]
# Each row = one sample
# Each column = a feature (e.g., amount spent, time of day, merchant type, etc.)

X = np.random.rand(10, 5)


# Binary labels: 0 = Not Fraud, 1 = Fraud
# Y = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
# These are the correct answers (what you want the model to learn to predict):
Y = np.random.randint(0, 2, 10)

# Train the XGBoost classifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X, Y)

# Predict for a new random input
new_input = np.random.rand(1, 5)
#This is one new case you want to classify:

prediction = model.predict(new_input)

# Output result
print("Fraud?" if prediction[0] == 1 else "Not Fraud")
