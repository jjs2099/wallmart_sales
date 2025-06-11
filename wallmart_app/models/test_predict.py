import sys
import os
import torch
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
from lstm import SalesLSTM  # Adjust this import if needed
from model_with_normalization_layer import ComplexRegressionModel  # Import the complex regression model

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up base directory and model directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR)  # Adjust as per your directory structure
DATA_FILE_PATH = os.path.join(BASE_DIR, 'Walmart.csv')  # Path to your original data file

# Load the historical data at the top level so it can be accessed globally
sales_data = pd.read_csv(DATA_FILE_PATH)
sales_data['Date'] = pd.to_datetime(sales_data['Date'], format='%d-%m-%Y')

def calculate_average_features(sales_data):
    """Calculate average values for lag features and other derived features."""
    avg_lag1 = sales_data['Weekly_Sales'].shift(1).mean()
    avg_lag2 = sales_data['Weekly_Sales'].shift(2).mean()
    avg_ma4 = sales_data['Weekly_Sales'].rolling(window=4).mean().mean()
    avg_ma12 = sales_data['Weekly_Sales'].rolling(window=12).mean().mean()
    avg_temp_fuel = (sales_data['Temperature'] * sales_data['Fuel_Price']).mean()
    avg_cpi_unemployment = (sales_data['CPI'] * sales_data['Unemployment']).mean()

    # Ensure 'Day' is present in the DataFrame
    sales_data['Day'] = sales_data['Date'].dt.day
    avg_temp_day = (sales_data['Temperature'] * sales_data['Day']).mean()

    return avg_lag1, avg_lag2, avg_ma4, avg_ma12, avg_temp_fuel, avg_cpi_unemployment, avg_temp_day

def preprocess_input(data, sales_data):
    # Calculate average values for the features
    avg_lag1, avg_lag2, avg_ma4, avg_ma12, avg_temp_fuel, avg_cpi_unemployment, avg_temp_day = calculate_average_features(sales_data)

    # Convert form data into a DataFrame
    input_data = pd.DataFrame(data, index=[0])
    input_data['Date'] = pd.to_datetime(input_data['Date'])
    
    # Extract date-related features
    input_data['Year'] = input_data['Date'].dt.year
    input_data['Month'] = input_data['Date'].dt.month
    input_data['Day'] = input_data['Date'].dt.day
    input_data['WeekOfYear'] = input_data['Date'].dt.isocalendar().week
    
    # Dropping 'Date' as it won't be used directly in prediction
    input_data.drop(['Date'], axis=1, inplace=True)
    
    # Generating a random SP500_Close value
    random_sp500 = random.uniform(3000, 4500)  # Assuming a reasonable range for SP500
    input_data['SP500_Close'] = random_sp500
    
    # Apply the average values for features that can't be directly calculated from the input
    input_data['Temp_Fuel'] = input_data['Temperature'] * input_data['Fuel_Price']
    input_data['CPI_Unemployment'] = input_data['CPI'] * input_data['Unemployment']
    input_data['Temp_Day'] = input_data['Temperature'] * input_data['Day']
    input_data['Sales_Lag1'] = avg_lag1
    input_data['Sales_Lag2'] = avg_lag2
    input_data['Sales_MA_4'] = avg_ma4
    input_data['Sales_MA_12'] = avg_ma12
    
    # Ensure the order of columns matches the model expectation
    input_data = input_data[['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 
                             'Unemployment', 'Year', 'Month', 'Day', 'WeekOfYear', 'SP500_Close', 
                             'Temp_Fuel', 'CPI_Unemployment', 'Temp_Day', 'Sales_Lag1', 'Sales_Lag2', 
                             'Sales_MA_4', 'Sales_MA_12']]

    return input_data

# Simulate form data
form_data = {
    'Store': 1,
    'Holiday_Flag': 0,  # Assuming 0 for 'No'
    'Temperature': 55.0,
    'Fuel_Price': 2.5,
    'CPI': 211.096358,
    'Unemployment': 8.1,
    'Date': '2023-08-13',  # Example date
}

# Preprocess input data
input_data = preprocess_input(form_data, sales_data)

# Model selection
model_type = 'Complex Regression Model'  # Options: 'LSTM Model', 'Random Forest Model', 'Complex Regression Model'

if model_type == 'LSTM Model':
    model_path = os.path.join(MODEL_DIR, 'lstm_model.pth')
    model = SalesLSTM(input_size=input_data.shape[1], hidden_size=100, num_layers=3, output_size=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Scaling input data as done during training
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Fit the scaler_y with the original target data from sales_data
    y_train = sales_data['Weekly_Sales'].values.reshape(-1, 1)
    scaler_y.fit(y_train)

    X_scaled = scaler_X.fit_transform(input_data)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    X_tensor = X_tensor.unsqueeze(0)  # Adding batch dimension

    with torch.no_grad():
        prediction = model(X_tensor).item()

    # Inverse scaling the predicted value
    prediction = scaler_y.inverse_transform([[prediction]])[0][0]
    print(f"LSTM Model Prediction: {prediction}")

elif model_type == 'Random Forest Model':
    model_path = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
    
    # Load the Random Forest model
    model = joblib.load(model_path)

    # Scaling input data as done during training
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Fit the scaler_y with the original target data from sales_data
    y_train = sales_data['Weekly_Sales'].values.reshape(-1, 1)
    scaler_y.fit(y_train)

    # Ensure the input data has the same columns as the training data
    X_scaled = scaler_X.fit_transform(input_data)
    input_data_scaled_df = pd.DataFrame(X_scaled, columns=input_data.columns)
    
    prediction = model.predict(input_data_scaled_df)[0]

    # Inverse scaling the predicted value
    prediction = scaler_y.inverse_transform([[prediction]])[0][0]
    print(f"Random Forest Model Prediction: {prediction}")

elif model_type == 'Complex Regression Model':
    model_path = os.path.join(MODEL_DIR, 'complex_regression_model.pth')
    
    # Initialize the Complex Regression Model
    input_size = input_data.shape[1]  # Ensure this matches the model's expected input size
    model = ComplexRegressionModel(input_size=input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Scaling input data as done during training
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Fit the scaler_y with the original target data from sales_data
    y_train = sales_data['Weekly_Sales'].values.reshape(-1, 1)
    scaler_y.fit(y_train)

    X_scaled = scaler_X.fit_transform(input_data)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    X_tensor = X_tensor.unsqueeze(0)  # Adding batch dimension if necessary

    # Ensure the tensor has the correct shape
    X_tensor = X_tensor.view(-1, input_size)

    with torch.no_grad():
        prediction = model(X_tensor).item()

    # Inverse scaling the predicted value
    prediction = scaler_y.inverse_transform([[prediction]])[0][0]
    print(f"Complex Regression Model Prediction: {prediction}")