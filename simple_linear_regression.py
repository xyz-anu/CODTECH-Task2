# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
def load_dataset(file_name):
    """Loads the dataset and returns a DataFrame."""
    data = pd.read_csv(file_name)
    print("Dataset loaded successfully.")
    print(data.head())
    return data

# Data exploration
def explore_data(data):
    """Explores the dataset by checking for missing values and displaying info."""
    print("\nDataset Information:")
    print(data.info())
    print("\nMissing Values:\n", data.isnull().sum())

# Split data into features and target
def split_data(data):
    """Splits the dataset into features and target variable."""
    X = data[['TV']]  # Predictor variable
    y = data['Sales']  # Target variable
    print("\nFeature and Target Variables Selected.")
    return X, y

# Train-test split
def split_train_test(X, y):
    """Splits the data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("\nTraining and Testing Sets Created:")
    print("Training Set:", X_train.shape, y_train.shape)
    print("Testing Set:", X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test

# Train the linear regression model
def train_model(X_train, y_train):
    """Trains a Linear Regression model and returns the trained model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("\nModel Training Complete.")
    print("Slope (Coefficient):", model.coef_[0])
    print("Intercept:", model.intercept_)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    """Evaluates the model using Mean Squared Error and R-squared."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\nModel Evaluation Metrics:")
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R²):", r2)
    return y_pred

# Visualize the regression line
def plot_regression_line(X, y, model):
    """Plots the regression line over the actual data."""
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color="blue", label="Actual Data")
    plt.plot(X, model.predict(X), color="red", linewidth=2, label="Regression Line")
    plt.title("Simple Linear Regression: TV Budget vs Sales")
    plt.xlabel("TV Advertising Budget ($)")
    plt.ylabel("Sales")
    plt.legend()
    plt.show()

# Plot actual vs predicted values
def plot_actual_vs_predicted(y_test, y_pred):
    """Plots actual vs. predicted values for test data."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color="purple", alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             color="orange", linewidth=2, linestyle="--")
    plt.title("Actual vs Predicted Sales")
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.show()

# Main function to run the pipeline
def main():
    file_name = "Advertising.csv"  # Replace with your dataset file name
    data = load_dataset(file_name)
    explore_data(data)
    X, y = split_data(data)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    model = train_model(X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test)
    plot_regression_line(X, y, model)
    plot_actual_vs_predicted(y_test, y_pred)

# Run the main function
if __name__ == "__main__":
    main()
