# import os
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split
# import joblib
# import matplotlib.pyplot as plt

# def load_and_prepare_data(file_path):
#     merged_data = pd.read_csv(file_path)
#     X = merged_data.drop(['Weekly_Sales'], axis=1)
#     y = merged_data['Weekly_Sales']
#     return X, y

# def train_random_forest(X_train, y_train, model_path, n_estimators=100, random_state=42):
#     if os.path.exists(model_path):
#         model = joblib.load(model_path)
#         trained = True
#     else:
#         model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
#         model.fit(X_train, y_train)
#         joblib.dump(model, model_path)
#         trained = False
#     return model, trained

# def plot_actual_vs_predicted(y_test, y_pred, output_dir, graph_filename='random_forest_graph.png'):
#     plt.figure(figsize=(10, 6))
#     plt.plot(y_test.reset_index(drop=True), label="Actual Sales", color="blue")
#     plt.plot(y_pred, label="Predicted Sales", color="red", linestyle="--")
#     plt.xlabel("Sample")
#     plt.ylabel("Weekly Sales")
#     plt.title("Actual vs Predicted Weekly Sales")
#     plt.legend()
#     plt.tight_layout()
    
#     graph_path = os.path.join(output_dir, graph_filename)
    
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     plt.savefig(graph_path)
#     plt.close()
    
#     return graph_path

# def run_random_forest_pipeline(file_path, output_dir, model_filename='random_forest_model.pkl'):
#     X, y = load_and_prepare_data(file_path)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     model_path = os.path.join(output_dir, model_filename)
#     model, trained = train_random_forest(X_train, y_train, model_path)
    
#     # Generate predictions and calculate metrics
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)

#     # Generate and save the graph
#     graph_path = plot_actual_vs_predicted(y_test, y_pred, output_dir)
    
#     return mse, r2, graph_path, trained

import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

# Set up base directory for saving models and other files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')  # Directory to save models

def load_and_prepare_data(file_path):
    """
    Load and prepare the dataset by separating features and the target variable.
    """
    data = pd.read_csv(file_path)
    X = data.drop(['Weekly_Sales'], axis=1)
    y = data['Weekly_Sales']
    return X, y

def train_random_forest(X_train, y_train, model_path, n_estimators=100, random_state=42):
    """
    Train the Random Forest model on the provided training data. If a model already exists at model_path,
    load it instead of retraining.
    """
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        trained = False  # The model was loaded, not trained
        print("Loaded existing model.")
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        trained = True  # The model was trained
        print(f"Model saved to {model_path}.")
    
    return model, trained

    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set and return performance metrics.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_pred

def plot_actual_vs_predicted(y_test, y_pred, output_dir, graph_filename='random_forest_graph.png'):
    """
    Plot actual vs. predicted sales and save the plot to a file.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.reset_index(drop=True), label="Actual Sales", color="blue")
    plt.plot(y_pred, label="Predicted Sales", color="red", linestyle="--")
    plt.xlabel("Sample")
    plt.ylabel("Weekly Sales")
    plt.title("Actual vs Predicted Weekly Sales")
    plt.legend()
    plt.tight_layout()
    
    graph_path = os.path.join(output_dir, graph_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(graph_path)
    plt.close()
    
    print(f"Plot saved to {graph_path}.")
    return graph_path

def run_random_forest_pipeline(file_path, output_dir, model_filename='random_forest_model.pkl'):
    """
    Execute the full pipeline for training, evaluating, and plotting with the Random Forest model.
    """
    # Load and prepare the data
    X, y = load_and_prepare_data(file_path)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the path to save/load the model
    model_path = os.path.join(output_dir, model_filename)
    
    # Train or load the Random Forest model
    model, trained = train_random_forest(X_train, y_train, model_path)
    
    # Evaluate the model
    mse, r2, y_pred = evaluate_model(model, X_test, y_test)
    
    # Plot the actual vs predicted values
    graph_path = plot_actual_vs_predicted(y_test, y_pred, output_dir)
    
    # Return the evaluation metrics, graph path, and whether the model was trained
    return mse, r2, graph_path, trained

