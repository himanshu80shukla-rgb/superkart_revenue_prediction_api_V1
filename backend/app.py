
# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize Flask app
superkart_revenue_prediction_api = Flask("Superkart Sales Revenue Prediction")
CORS(superkart_revenue_prediction_api)

# Load the trained model pipeline (preprocessing + model)
model = joblib.load("/content/drive/MyDrive/Colab Notebooks/SuperKartProject/deployment_files/superkart_sales_forecast_model_v1_0.joblib")

# Define the route for the home page (GET Request)
@superkart_revenue_prediction_api.get('/')
def home():
    """
    This function handles GET requests to the root url ('/') of the API
    It returns a simple welcome message
    """
    return "✅ Welcome to the SuperKart Sales Prediction API"

# Define an endpoint for single sales prediction (POST request))
@superkart_revenue_prediction_api.post('/v1/singlesalespredict')
def predict_sales_single():
    """
    This function handles POST requests to the '/v1/singlesalespredict' endpoint.
    It expects a JSON payload containing property details and returns
    the predicted sales as a JSON response.
    """
    try:
        # Get the JSON data from the request body
        data = request.get_json()
        print("Raw incoming data:", data)

        # Validate required fields
        required_fields = [
            'Product_Weight',
            'Product_Sugar_Content',
            'Product_Allocated_Area',
            'Product_MRP',
            'Store_Size',
            'Store_Location_City_Type',
            'Store_Type',
            'Store_Age_Years',
            'Product_Type_Category'
        ]
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({'error': f"Missing fields: {missing_fields}"}),

        # Extract relevant features from the JSON data
        inputrequest = {
            'Product_Weight': float(data['Product_Weight']),
            'Product_Sugar_Content': data['Product_Sugar_Content'],
            'Product_Allocated_Area_Log': np.log1p(float(data['Product_Allocated_Area'])),
            'Product_MRP': float(data['Product_MRP']),
            'Store_Size': data['Store_Size'],
            'Store_Location_City_Type': data['Store_Location_City_Type'],
            'Store_Type': data['Store_Type'],
            'Store_Age_Years': int(data['Store_Age_Years']),
            'Product_Type_Category': data['Product_Type_Category']
        }

        # Convert the extracted data into a Pandas DataFrame
        input_df = pd.DataFrame([inputrequest])
        print("Transformed input for model:\n", input_df)

        # Make prediction
        prediction = model.predict(input_df).tolist()[0]
        return jsonify({'Predicted_Sales': prediction})

    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500


# Define an endpoint for batch prediction (POST request)
@superkart_revenue_prediction_api.post('/v1/batchsalespredict')
def predict_rental_price_batch():
    """
    This function handles POST requests to the '/v1/batchsalespredict' endpoint.
    It expects a CSV file containing property details for multiple products and returns
    the predicted sales as a dictionary as a JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for all products sales in the DataFrame
    predicted_batch_sales = model.predict(input_data).tolist()

    # Create a dictionary of predictions with product Product_Type_Category as keys
    product_ids = input_data['Product_Type_Category'].tolist()  # Assuming 'iProduct_Type_Categoryd' is the prodict ID column
    output_dict = dict(zip(product_ids, predicted_batch_sales))  # Use actual sales

    # Return the predictions dictionary as a JSON response
    return output_dict


# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_revenue_prediction_api.run(debug=True)
