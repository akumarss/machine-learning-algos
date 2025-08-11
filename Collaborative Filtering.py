from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# Sample data: (user, item, rating)
ratings_data = [
    ('User1', 'Movie1', 5),
    ('User1', 'Movie2', 3),
    ('User2', 'Movie1', 4),
    ('User2', 'Movie3', 2),
    ('User3', 'Movie2', 4),
    ('User3', 'Movie3', 5),
]

# Step 1: Load data using Surprise's Reader
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(
    pd.DataFrame(ratings_data, columns=["user", "item", "rating"]),
    reader
)

# Step 2: Train/Test Split
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Step 3: User-based collaborative filtering using KNN
sim_options = {
    "name": "cosine",
    "user_based": True  # change to False for item-based
}

model = KNNBasic(sim_options=sim_options)
model.fit(trainset)

# Step 4: Predict on test set
predictions = model.test(testset)

# Step 5: Print predictions
for pred in predictions:
    print(f"User: {pred.uid}, Item: {pred.iid}, Actual: {pred.r_ui}, Predicted: {round(pred.est, 2)}")

# Step 6: Measure RMSE
print("RMSE:", round(accuracy.rmse(predictions), 3))
