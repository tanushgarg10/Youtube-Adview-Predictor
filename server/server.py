from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model
with open('server/artifacts/random_forest_youtubeadview-2.pkl', 'rb') as file:
    random_forest = pickle.load(file)

# Load the fitted scaler
with open('server/artifacts/scaler.pickle', 'rb') as file:
    scaler = pickle.load(file)
    
with open('server/artifacts/label_encoder.pickle', 'rb') as file:
    le = pickle.load(file)

# Initialize and fit LabelEncoder (using known categories)
categories = ['A', 'B', 'C', 'D', 'E', 'F']
le = LabelEncoder()
le.fit(categories)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Define the function to preprocess the user input data
def preprocess_data(data):
    df = pd.DataFrame([data])
    category={3: 0, 1: 1, 2: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}
    df["category"]=df["category"].map(category)
    
    # Normalize the numerical variables using the loaded MinMaxScaler
    numeric_cols = ['views', 'likes', 'dislikes', 'comment', 'duration']
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    return df

# Define the endpoint to take user input and return the predicted number of ad views
@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        try:
            # Get the user input data from the request body
            data = request.get_json()
            logging.debug(f"Received data: {data}")
            
            # Preprocess the user input data
            preprocessed_data = preprocess_data(data)
            logging.debug(f"Preprocessed data: {preprocessed_data}")
            
            # Make predictions on the user input data using the trained Random Forest Regressor model
            numeric_cols = ['views', 'likes', 'dislikes', 'comment', 'duration']
            features = preprocessed_data[numeric_cols + ['category']]
            prediction = random_forest.predict(features)
            logging.debug(f"Prediction: {prediction}")
            
            # Return the prediction as a JSON response
            return jsonify({'prediction': float(prediction[0])})
        except Exception as e:
            logging.error(f"Error occurred: {str(e)}")
            return jsonify({'error': str(e)}), 400
    else:
        return jsonify({'error': 'Request content type must be application/json'}), 400
if __name__ == '__main__':
    app.run(debug=True)
