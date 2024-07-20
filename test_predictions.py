import pickle
import numpy as np
import pandas as pd

# Load the fitted model and scaler
with open('house_price_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Example feature for testing
sample_features = pd.DataFrame([[1600, 3, 2]], columns=['Area', 'No_Of_Bedrooms', 'No_Of_Bathrooms'])

# Transform the sample features using the fitted scaler
sample_features_scaled = scaler.transform(sample_features)

# Make a prediction
prediction = model.predict(sample_features_scaled)

print(f"Predicted Price: {prediction[0]}")
