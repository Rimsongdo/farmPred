from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize the Flask application
application = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_model.pkl')

@application.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the POST request (as JSON)
        data = request.get_json(force=True)

        # Extract features from the incoming request
        features = np.array([data['soil_humidity_2'], data['air_temperature'], data['air_humidity']]).reshape(1, -1)

        # Make the prediction
        prediction = model.predict(features)

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        # Return error message if something goes wrong
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run the app on the default Flask port (5000)
    application.run(debug=True)
