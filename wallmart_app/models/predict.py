import os
import torch
import pandas as pd
import yfinance as yf  # Import yfinance to fetch SP500 data
from django.shortcuts import render
from sklearn.preprocessing import StandardScaler
import joblib
from .lstm import SalesLSTM  # Adjust this import if needed
from .model_with_normalization_layer import ComplexRegressionModel  # Import the complex regression model
from .random_forest import run_random_forest_pipeline  # Import the random forest functions

# Set up base directory and model directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')  # Adjust as per your directory structure
DATA_FILE_PATH = os.path.join(MODEL_DIR, 'Walmart.csv')  # Path to your original data file

# Load the historical data
sales_data = pd.read_csv(DATA_FILE_PATH)
sales_data['Date'] = pd.to_datetime(sales_data['Date'], format='%d-%m-%Y')

# Cached SP500 value and date
cached_sp500_value = None
cached_sp500_date = None

def fetch_sp500_value():
    global cached_sp500_value, cached_sp500_date
    today = pd.Timestamp('today').normalize()
    
    if cached_sp500_date != today:
        # Fetch the latest SP500 value from Yahoo Finance
        sp500_data = yf.download('^GSPC', period='1d', interval='1d')
        cached_sp500_value = sp500_data['Close'].iloc[-1]
        cached_sp500_date = today
    
    return cached_sp500_value

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
    data['Store'] = int(data['Store'])
    data['Holiday_Flag'] = int(data['Holiday_Flag'])
    data['Temperature'] = float(data['Temperature'])
    data['Fuel_Price'] = float(data['Fuel_Price'])
    data['CPI'] = float(data['CPI'])
    data['Unemployment'] = float(data['Unemployment'])
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
    
    # Fetch the SP500 value using the cached function
    input_data['SP500_Close'] = fetch_sp500_value()
    
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

def predict_view(request):
    if request.method == 'POST':
        # Gather data from the form
        form_data = {
            'Store': request.POST['Store'],
            'Holiday_Flag': request.POST['Holiday_Flag'],
            'Temperature': request.POST['Temperature'],
            'Fuel_Price': request.POST['Fuel_Price'],
            'CPI': request.POST['CPI'],
            'Unemployment': request.POST['Unemployment'],
            'Date': request.POST['Date']
        }

        # Preprocess the input data
        input_data = preprocess_input(form_data, sales_data)
        
        # Model selection (You can customize this based on user input as well)
        model_type = request.POST['Model']  # Get model type from the form
        
        if model_type == 'LSTM Model':
            model_path = os.path.join(MODEL_DIR, 'lstm_model.pth')
            model = SalesLSTM(input_size=input_data.shape[1], hidden_size=100, num_layers=3, output_size=1)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            # Scaling input data as done during training
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            y_train = sales_data['Weekly_Sales'].values.reshape(-1, 1)
            scaler_y.fit(y_train)

            X_scaled = scaler_X.fit_transform(input_data)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            X_tensor = X_tensor.unsqueeze(0)  # Adding batch dimension

            with torch.no_grad():
                prediction = model(X_tensor).item()

            prediction = scaler_y.inverse_transform([[prediction]])[0][0]

        elif model_type == 'Random Forest Model':
            # Load the preprocessed data file
            preprocessed_file_path = os.path.join(MODEL_DIR, 'preprocessed_dataframe.csv')
            
            # Run the Random Forest pipeline to get model metrics (MSE and R2) and the trained model
            mse, r2, graph_path, trained = run_random_forest_pipeline(DATA_FILE_PATH, MODEL_DIR)
            
            # Since Random Forest is a tree-based model, it typically does not require feature scaling.
            # However, ensure that the input data columns match what was used during training.
            model = joblib.load(os.path.join(MODEL_DIR, 'random_forest_model.pkl'))
            
            # Convert the input data to the correct format (i.e., no scaling needed)
            input_data_rf = input_data.values.reshape(1, -1)
            
            # Make a prediction using the Random Forest model
            prediction = model.predict(input_data_rf)[0]
            
            # Optionally, return MSE and R2 as part of the result (for model evaluation purposes)
            result_message = f"Predicted Weekly Sales: {prediction}\nMSE: {mse}, R2: {r2}"
            
            # If you want to display the graph as well, you can add it to the context:
            context = {
                'prediction': result_message,
                'graph_path': graph_path
            }
            
            # Render the result page with the prediction and other details
            return render(request, 'result.html', context)

        elif model_type == 'Complex Regression Model':
            model_path = os.path.join(MODEL_DIR, 'complex_regression_model.pth')
            input_size = input_data.shape[1]
            model = ComplexRegressionModel(input_size=input_size)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            y_train = sales_data['Weekly_Sales'].values.reshape(-1, 1)
            scaler_y.fit(y_train)

            X_scaled = scaler_X.fit_transform(input_data)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            X_tensor = X_tensor.view(-1, input_size)

            with torch.no_grad():
                prediction = model(X_tensor).item()

            prediction = scaler_y.inverse_transform([[prediction]])[0][0]

        # Return the prediction result
        return render(request, 'result.html', {'prediction': prediction})
    
    # If GET request, render the prediction form
    return render(request, 'predict.html')
