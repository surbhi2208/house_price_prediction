import joblib
import numpy as np

# Load model
model = joblib.load("house_price_model.pkl")

# Example house
area = 2200
bedrooms = 4
bathrooms = 3

features = np.array([[area, bedrooms, bathrooms]])

prediction = model.predict(features)

print(f"Predicted House Price: ${prediction[0]:,.2f}")
