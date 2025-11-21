"""
Assignment 6 Part 1: Student Performance Prediction
Name: _______________
Date: _______________

This assignment predicts student test scores based on hours studied.
Complete all the functions below following the in-class ice cream example.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def load_and_explore_data(filename):
    """
    Load the student scores data and explore it
    
    Args:
        filename: name of the CSV file to load
    
    Returns:
        pandas DataFrame containing the data
    """
    data = pd.read_csv(filename)

    print("=== Student Scores Data ===")
    print(f"\nFirst 5 rows:")
    print(data.head())

    print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")

    print(f"\nBasic statistics:")
    print(data.describe())

    return data


def create_scatter_plot(data):
    """
    Create a scatter plot to visualize the relationship between hours studied and scores
    
    Args:
        data: pandas DataFrame with Hours and Scores columns
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Hours'], data['Scores'], color='purple', alpha=0.6)

    plt.xlabel('Hours Studied', fontsize=12)
    plt.ylabel('Test Scores', fontsize=12)
    plt.title('Student Test Scores vs Hours Studied', fontsize=14, fontweight='bold')

    plt.grid(True, alpha=0.3)
    plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')

    plt.show()


def split_data(data):
    """
    Split data into features (X) and target (y), then into training and testing sets
    """
    X = data[['Hours']]     # features as DataFrame
    y = data['Scores']      # target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n=== Data Splitting ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Create and train a linear regression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\n=== Model Parameters ===")
    print(f"Slope (coefficient): {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance on test data
    """
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    print("\n=== Model Evaluation ===")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    return predictions


def visualize_results(X_train, y_train, X_test, y_test, predictions, model):
    """
    Visualize the model's predictions against actual values
    """
    plt.figure(figsize=(12, 6))

    # training data
    plt.scatter(X_train, y_train, color='blue', label='Training Data')

    # actual test data
    plt.scatter(X_test, y_test, color='green', label='Test Data (Actual)')

    # predictions
    plt.scatter(X_test, predictions, color='red', marker='x', s=80, label='Predictions')

    # line of best fit
    X_range = np.linspace(min(X_train['Hours']), max(X_train['Hours']), 100).reshape(-1, 1)
    y_range_pred = model.predict(X_range)
    plt.plot(X_range, y_range_pred, color='black', linewidth=2, label='Best Fit Line')

    plt.xlabel('Hours Studied')
    plt.ylabel('Test Score')
    plt.title('Model Predictions vs Actual Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig('predictions_plot.png', dpi=300, bbox_inches='tight')
    plt.show()


def make_prediction(model, hours):
    """
    Make a prediction for a specific number of hours studied
    """
    hours_array = np.array([[hours]])
    predicted_score = model.predict(hours_array)[0]

    print(f"\nA student who studies {hours} hours is predicted to score: {predicted_score:.2f}")

    return predicted_score


if __name__ == "__main__":
    print("=" * 70)
    print("STUDENT PERFORMANCE PREDICTION - YOUR ASSIGNMENT")
    print("=" * 70)

    # Step 1: Load and explore the data
    data = load_and_explore_data('student_scores.csv')

    # Step 2: Visualize the relationship
    create_scatter_plot(data)

    # Step 3: Split the data
    X_train, X_test, y_train, y_test = split_data(data)

    # Step 4: Train the model
    model = train_model(X_train, y_train)

    # Step 5: Evaluate the model
    predictions = evaluate_model(model, X_test, y_test)

    # Step 6: Visualize results
    visualize_results(X_train, y_train, X_test, y_test, predictions, model)

    # Step 7: Make a prediction for 7 hours studied
    make_prediction(model, 7)

    print("\n" + "=" * 70)
    print("✓ Assignment complete! Check your saved plots.")
    print("Don't forget to complete a6_part1_writeup.md!")
    print("=" * 70)


