import os
from django.shortcuts import render
from .models.lstm import run_lstm_pipeline
from .models.random_forest import run_random_forest_pipeline
from .models.model_with_normalization_layer import run_complex_regression_pipeline
from .models.predict import preprocess_input, calculate_average_features, SalesLSTM, ComplexRegressionModel
import torch
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')  # Directly refer to the static/images folder
MODEL_DIR = os.path.join(BASE_DIR, 'models')  # Adjust as per your directory structure
DATA_FILE_PATH = os.path.join(MODEL_DIR, 'Walmart.csv')  # Path to your original data file

# Load the historical data
sales_data = pd.read_csv(DATA_FILE_PATH)
sales_data['Date'] = pd.to_datetime(sales_data['Date'], format='%d-%m-%Y')

def model_results_view(request):
    file_path = os.path.join(BASE_DIR, 'models/preprocessed_dataframe.csv')
    model_output_dir = STATIC_DIR  # Output directory for saving images

    # Run LSTM model pipeline
    lstm_mse, lstm_r2, lstm_graph_path, lstm_trained = run_lstm_pipeline(file_path, model_output_dir)

    # Run Random Forest model pipeline
    rf_mse, rf_r2, rf_graph_path, rf_trained = run_random_forest_pipeline(file_path, model_output_dir)

    # Run Complex Regression model pipeline
    cr_mse, cr_r2, cr_graph_path, cr_trained = run_complex_regression_pipeline(file_path, model_output_dir)

    models = [
        {
            'name': 'LSTM Model',
            'mse': lstm_mse,
            'r2': lstm_r2,
            'graph': os.path.basename(lstm_graph_path),  # Only filename since it's in the static folder
            'trained': lstm_trained,
        },
        {
            'name': 'Random Forest Model',
            'mse': rf_mse,
            'r2': rf_r2,
            'graph': os.path.basename(rf_graph_path),  # Only filename since it's in the static folder
            'trained': rf_trained,
        },
        {
            'name': 'Complex Regression Model',
            'mse': cr_mse,
            'r2': cr_r2,
            'graph': os.path.basename(cr_graph_path),  # Only filename since it's in the static folder
            'trained': cr_trained,
        },
    ]

    context = {
        'models': models,
    }

    return render(request, 'index.html', context)

def predict(request):
    if request.method == 'POST':
        # Debug: Print the form data being submitted
        print(request.POST)

        # Adjust the form_data dictionary to match the capitalized field names
        form_data = {
            'Store': request.POST.get('Store', ''),
            'Holiday_Flag': request.POST.get('IsHoliday', ''),
            'Temperature': request.POST.get('Temperature', ''),
            'Fuel_Price': request.POST.get('Fuel_Price', ''),
            'CPI': request.POST.get('CPI', ''),
            'Unemployment': request.POST.get('Unemployment', ''),
            'Date': request.POST.get('Date', '')
        }

        # Preprocess the input data
        input_data = preprocess_input(form_data, sales_data)

        # Model selection
        model_type = request.POST.get('Model', '')  # Get model type from the form

        if model_type == 'LSTM Model':
            model_path = os.path.join(MODEL_DIR, 'lstm_model.pth')
            model = SalesLSTM(input_size=input_data.shape[1], hidden_size=100, num_layers=3, output_size=1)
            model.load_state_dict(torch.load(model_path))
            model.eval()

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
            model_path = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
            model = joblib.load(model_path)

            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            y_train = sales_data['Weekly_Sales'].values.reshape(-1, 1)
            scaler_y.fit(y_train)

            X_scaled = scaler_X.fit_transform(input_data)
            input_data_scaled_df = pd.DataFrame(X_scaled, columns=input_data.columns)
            
            prediction = model.predict(input_data_scaled_df)[0]
            prediction = scaler_y.inverse_transform([[prediction]])[0][0]

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
