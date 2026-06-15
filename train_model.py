import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dataset.csv", header=None)

# Separate features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Feature count:", X.shape[1])

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create model
model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

import pickle

with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved ✅")