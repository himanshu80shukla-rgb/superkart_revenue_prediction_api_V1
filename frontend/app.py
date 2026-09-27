# Saving the frontend app


# Streamlit Web App for SuperKart Sales Forecasting
import streamlit as st
import requests
import numpy as np

# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"

# App Title
st.title("🛒 SuperKart Sales Forecasting App")

# Section for online prediction
st.subheader("Superkart - Online Sales Prediction")

# Instructions
st.markdown("🔍 Enter product and store attributes to forecast **monthly product sales revenue**.\n\n_All sales are reported in ($) USD._")

# User Inputs
Product_Weight = st.number_input("Product Weight (oz)", min_value=0.0, value=12.66)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.number_input("Product Allocated Area (linear in.)", min_value=0.0, value=100.0)
Product_MRP = st.number_input("Maximum Retail Price (USD)", min_value=0.0, value=150.0)
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
Store_Age_Years = st.slider("Store Age (years)", min_value=0, max_value=30, value=10)
Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])


# Apply log1p transform (must match backend model training)
Product_Allocated_Area_Log = np.log1p(Product_Allocated_Area)

# Prepare JSON payload for the backend
product_data = {
    "Product_Weight": str(Product_Weight),
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": str(Product_Allocated_Area),
    "Product_MRP": str(Product_MRP),
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Store_Age_Years": str(Store_Age_Years),
    "Product_Type_Category": Product_Type_Category
}

# Make prediction when the "Predict" button is clicked
if st.button("Predict", type="primary"):
    try:
       response = requests.post(f"{BACKEND_URL}/v1/singlesalespredict", json=product_data.to_dict(orient='records')[0])  # Send data to Flask API
       if response.status_code == 200:
            result = response.json()
            predicted_sales = result["Predicted_Sales"]
            st.success(f"📈 Predicted Monthly Sales: **${predicted_sales:,.2f} USD**")
       else:
            st.error("❌ API Error: Please verify input values or try again later.")
    except Exception as e:
        st.error(f"⚠️ Connection error: {e}")


# Section for batch prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file for batch prediction
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:
    if st.button("Predict Batch", type="primary"):
       try:
          response = requests.post(f"{BACKEND_URL}/v1/batchsalespredict", files={"file": uploaded_file})  # Send file to Flask API
          if response.status_code == 200:
            predictions = response.json()
            st.success("Batch predictions completed!")
            st.write(predictions)  # Display the predictions
          else:
            st.error("Unable to connect to the prediction API.")
       except Exception as e:
        st.error(f"⚠️ Connection error: {e}")

