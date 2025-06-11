import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import joblib

class ComplexRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(ComplexRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

def load_and_preprocess_data(file_path, output_dir):
    merged_data = pd.read_csv(file_path)
    X = merged_data.drop(['Weekly_Sales'], axis=1)
    y = merged_data['Weekly_Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

    # Save the scalers
    scaler_X_path = os.path.join(output_dir, 'scaler_X_complex.pkl')
    scaler_y_path = os.path.join(output_dir, 'scaler_y_complex.pkl')
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).view(-1, 1)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler_y, scaler_X


def train_model(X_train_tensor, y_train_tensor, model_path, input_size, num_epochs=300, learning_rate=0.001):
    epoch_losses = []  # Initialize epoch_losses as an empty list
    
    # Check if the model parameters already exist
    if os.path.exists(model_path):
        model = ComplexRegressionModel(input_size)
        model.load_state_dict(torch.load(model_path))
        trained = True
    else:
        model = ComplexRegressionModel(input_size=input_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
        for epoch in range(num_epochs):
            model.train()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Save the trained model
        torch.save(model.state_dict(), model_path)
        trained = False
    
    return model, epoch_losses, trained

    

def evaluate_model(model, X_test_tensor, y_test_tensor, scaler_y):
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        y_pred = y_pred_tensor.numpy()
        y_test_actual = y_test_tensor.numpy()
        y_pred_original = scaler_y.inverse_transform(y_pred)
        y_test_original = scaler_y.inverse_transform(y_test_actual)
        mse = mean_squared_error(y_test_original, y_pred_original)
        r2 = r2_score(y_test_original, y_pred_original)
    return mse, r2, y_pred_original, y_test_original

def plot_actual_vs_predicted(y_test, y_pred, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual Sales", color="blue")
    plt.plot(y_pred, label="Predicted Sales", color="red", linestyle="--")
    plt.xlabel("Sample")
    plt.ylabel("Weekly Sales")
    plt.title("Actual vs Predicted Weekly Sales")
    plt.legend()

    graph_path = os.path.join(output_dir, 'complex_regression_graph.png')
    plt.savefig(graph_path)
    plt.close()
    
    return graph_path

def run_complex_regression_pipeline(file_path, output_dir):
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler_y, scaler_X = load_and_preprocess_data(file_path, output_dir)
    input_size = X_train_tensor.shape[1]
    model_path = os.path.join(output_dir, 'complex_regression_model.pth')
    model, epoch_losses, trained = train_model(X_train_tensor, y_train_tensor, model_path, input_size)
    mse, r2, y_pred, y_test = evaluate_model(model, X_test_tensor, y_test_tensor, scaler_y)
    graph_path = plot_actual_vs_predicted(y_test, y_pred, output_dir)
    return mse, r2, graph_path, trained




import torch
import pandas as pd
import joblib

def predict_complex_regression(single_row_dict, model_path, scaler_X_path, scaler_y_path):
    # Convert input to DataFrame
    input_df = pd.DataFrame([single_row_dict])
    
    # Load scalers
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    # Preprocess input data
    input_features = input_df.values
    input_scaled = scaler_X.transform(input_features)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    # Load model
    model = ComplexRegressionModel(input_size=input_tensor.shape[1])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Predict
    with torch.no_grad():
        y_pred = model(input_tensor).numpy()
        y_pred_original = scaler_y.inverse_transform(y_pred)

    return y_pred_original[0][0]

