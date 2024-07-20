from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('house_price_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"Received data: {data}")  # Debug print

        area = data['Area']
        bedrooms = data['No_Of_Bedrooms']
        bathrooms = data['No_Of_Bathrooms']
        features = np.array([[area, bedrooms, bathrooms]])
        print(f"Features: {features}")  # Debug print

        # Scale features
        features_scaled = scaler.transform(features)
        print(f"Scaled Features: {features_scaled}")  # Debug print

        # Make prediction
        prediction = model.predict(features_scaled)
        print(f"Prediction: {prediction}")  # Debug print

        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run(debug=True)
