import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt

class SalesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SalesLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = torch.relu(self.fc1(out[:, -1, :]))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def load_and_preprocess_data(file_path, sequence_length=10):
    merged_data = pd.read_csv(file_path)
    X = merged_data.drop(['Weekly_Sales'], axis=1).values
    y = merged_data['Weekly_Sales'].values
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
    X_sequences = []
    y_sequences = []
    for i in range(len(X_tensor) - sequence_length):
        X_sequences.append(X_tensor[i:i+sequence_length])
        y_sequences.append(y_tensor[i+sequence_length])
    X_sequences = torch.stack(X_sequences)
    y_sequences = torch.tensor(y_sequences, dtype=torch.float32)
    X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler_y

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

def train_lstm_model(X_train, y_train, model_path, input_size, hidden_size=100, num_layers=3, output_size=1, num_epochs=300, learning_rate=0.0001):
    # Check if the model parameters already exist
    if os.path.exists(model_path):
        model = SalesLSTM(input_size, hidden_size, num_layers, output_size)
        model.load_state_dict(torch.load(model_path))
        trained = True
    else:
        model = SalesLSTM(input_size, hidden_size, num_layers, output_size)
        model.apply(weights_init)
        criterion = nn.SmoothL1Loss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        epoch_losses = []

        for epoch in range(num_epochs):
            model.train()
            outputs = model(X_train)
            loss = criterion(outputs, y_train.view(-1, 1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        # Save the trained model
        torch.save(model.state_dict(), model_path)
        trained = False
    
    return model, trained


def evaluate_lstm_model(model, X_test, y_test, scaler_y):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).numpy()
        y_test_actual = y_test.numpy()
        y_pred_original = scaler_y.inverse_transform(y_pred)
        y_test_original = scaler_y.inverse_transform(y_test_actual.reshape(-1, 1))
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

    graph_path = os.path.join(output_dir, 'lstm_graph.png')
    plt.savefig(graph_path)
    plt.close()
    
    return graph_path

def run_lstm_pipeline(file_path, output_dir):
    X_train, X_test, y_train, y_test, scaler_y = load_and_preprocess_data(file_path)
    input_size = X_train.shape[2]
    model_path = os.path.join(output_dir, 'lstm_model.pth')
    model, trained = train_lstm_model(X_train, y_train, model_path, input_size)
    mse, r2, y_pred, y_test = evaluate_lstm_model(model, X_test, y_test, scaler_y)
    graph_path = plot_actual_vs_predicted(y_test, y_pred, output_dir)
    return mse, r2, graph_path, trained

