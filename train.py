import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load iris dataset
df = pd.read_csv("data/iris_data.csv")

# Prepare features and target
# For demonstration, we create a dummy target (0 or 1) based on sepal length
X = df[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]]
y = (df["sepal length (cm)"] > df["sepal length (cm)"].median()).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training accuracy: {train_score:.4f}")
print(f"Testing accuracy: {test_score:.4f}")

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to model.pkl")
