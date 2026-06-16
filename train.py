import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
data = pd.read_csv("data/housing.csv")

# Features and target
X = data[["area", "bedrooms", "bathrooms"]]
y = data["price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"Model trained successfully!")
print(f"Mean Absolute Error: {mae:.2f}")

# Save model
joblib.dump(model, "house_price_model.pkl")

print("Model saved as house_price_model.pkl")
