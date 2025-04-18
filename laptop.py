import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load the dataframe
df = pd.read_csv("data.csv")
df = df.drop(df.columns[[0, 1, 9]], axis=1)

# Rename columns based on the dataset structure
df.rename(columns={
    'brand': 'brand',
    'name': 'name',
    'price': 'price',
    'spec_rating': 'spec_rating',
    'processor': 'processor',
    'CPU': 'cpu',
    'Ram': 'ram',
    'ROM': 'rom',
    'ROM_type': 'rom_type',
    'GPU': 'gpu',
    'display_size': 'display_size',
    'resolution_width': 'resolution_width',
    'resolution_height': 'resolution_height',
    'OS': 'os',
    'warranty': 'warranty'
}, inplace=True)

# Encode categorical variables
categorical_columns = ['brand', 'name', 'processor', 'cpu', 'ram', 'rom', 'rom_type', 'gpu', 'os']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Train-test split
X = df[['brand', 'name', 'processor', 'cpu', 'ram', 'rom', 'rom_type', 'gpu', 'display_size', 'resolution_width', 'resolution_height', 'os', 'warranty']]
y = np.log(df['price'])  # Log transformation for price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model and encoders
pickle.dump(rf_model, open("rf_model.pkl", "wb"))
pickle.dump(label_encoders, open("label_encoders.pkl", "wb"))

st.title("Laptop Price Predictor")

# Check if model file exists
model_path = "rf_model.pkl"
encoder_path = "label_encoders.pkl"
if os.path.exists(model_path) and os.path.exists(encoder_path):
    model = pickle.load(open(model_path, "rb"))
    label_encoders = pickle.load(open(encoder_path, "rb"))
else:
    st.error(f"Error: Model or encoder file not found. Please ensure the files are in the correct directory.")
    st.stop()

# User Inputs - Show original categorical names from dataset
brand = st.selectbox('Brand', label_encoders['brand'].classes_)
laptop_name = st.selectbox("Laptop Model", label_encoders['name'].classes_)
processor = st.selectbox("Processor", label_encoders['processor'].classes_)
cpu = st.selectbox("CPU", label_encoders['cpu'].classes_)
ram = st.selectbox("RAM (in GB)", label_encoders['ram'].classes_)
rom = st.selectbox("Storage (ROM)", label_encoders['rom'].classes_)
rom_type = st.selectbox("Storage Type", label_encoders['rom_type'].classes_)
gpu = st.selectbox("GPU", label_encoders['gpu'].classes_)
display_size = st.number_input("Display Size (in inches)")
resolution_width = st.selectbox("Screen Resolution Width", df['resolution_width'].unique())
resolution_height = st.selectbox("Screen Resolution Height", df['resolution_height'].unique())
os = st.selectbox("Operating System", label_encoders['os'].classes_)
warranty = st.selectbox("Warranty (years)", df['warranty'].unique())

# Function to safely encode categorical values
def safe_encode(label_encoder, value):
    if value in label_encoder.classes_:
        return label_encoder.transform([value])[0]
    else:
        return -1  # Assign unseen label a fixed number

# Prediction
if st.button('Predict Price'):
    try:
        ppi = ((resolution_width ** 2) + (resolution_height ** 2)) ** 0.5 / display_size
        query = np.array([
            safe_encode(label_encoders['brand'], brand),
            safe_encode(label_encoders['name'], laptop_name),
            safe_encode(label_encoders['processor'], processor),
            safe_encode(label_encoders['cpu'], cpu),
            safe_encode(label_encoders['ram'], ram),
            safe_encode(label_encoders['rom'], rom),
            safe_encode(label_encoders['rom_type'], rom_type),
            safe_encode(label_encoders['gpu'], gpu),
            display_size,
            resolution_width,
            resolution_height,
            safe_encode(label_encoders['os'], os),
            warranty
        ])
        query = query.reshape(1, -1)
        prediction = str(int(np.exp(model.predict(query)[0])))
        st.title(f"The predicted price of this configuration is ₹{prediction}")
    except Exception as e:
        st.error(f"Error occurred during prediction: {str(e)}")

# streamlit run "c:/Users/DELL/Desktop/FR lab mst/laptop.py"