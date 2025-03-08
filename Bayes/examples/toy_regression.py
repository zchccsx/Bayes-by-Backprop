import sys
sys.path.append("./")  # Replace with your actual path
# print(sys.path)
import Bayes  # Now you can import and use it
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from Bayes.models import BayesianRegressor
from Bayes.utils import (
    BayesianRegressionTrainer,
    plot_1d_regression_with_uncertainty
)

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def generate_toy_data(n_samples=500, noise_level=0.3, test_gaps=True):
    """
    Generate a synthetic 1D regression dataset with optional test gaps.
    
    Args:
        n_samples: Number of data points
        noise_level: Amount of noise to add to the data
        test_gaps: Whether to create gaps in the data to test uncertainty
        
    Returns:
        X, y: Features and targets for the dataset
    """
    # Generate inputs in the range [-4, 4]
    X = np.random.uniform(-4, 4, n_samples)
    
    if test_gaps:
        # Create data gaps to test uncertainty behavior
        # Remove samples in the ranges [-2, -1] and [1, 2]
        X = np.array([x for x in X if not (-2 <= x <= -1 or 1 <= x <= 2)])
        
    # Sort X for easier visualization later
    X = np.sort(X)
    
    # True function: y = x^3 - 2x^2 + 0.5x + noise
    y = X**3 - 2 * X**2 + 0.5 * X + np.random.normal(0, noise_level, len(X))
    
    # Reshape for sklearn compatibility
    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)
    
    return X, y

def main():
    print("Bayesian Neural Network 1D Regression Example")
    print("--------------------------------------------")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    X, y = generate_toy_data(n_samples=300, noise_level=0.5, test_gaps=True)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create a Bayesian Neural Network
    print("\nCreating Bayesian Neural Network...")
    model = BayesianRegressor(input_dim=1, hidden_dims=[32, 32], output_dim=1)
    
    # Create a trainer
    trainer = BayesianRegressionTrainer(
        model=model,
        learning_rate=0.001
    )
    
    # Train the model
    print("\nTraining the model...")
    train_losses, val_losses = trainer.train(
        X_train=X_train_tensor, 
        y_train=y_train_tensor,
        batch_size=32,
        epochs=1000,
        samples_nbr=5
    )
    
    # Plot the loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Generate predictions for plotting
    # Create a dense grid of points for visualization
    X_dense = np.linspace(-5, 5, 500).reshape(-1, 1)
    X_dense_tensor = torch.FloatTensor(X_dense)
    
    # Make predictions with uncertainty
    print("\nMaking predictions with uncertainty...")
    y_pred_mean, y_pred_std, _ = trainer.predict(X_dense_tensor, num_samples=100)
    
    # Plot the results
    print("\nPlotting the results...")
    plot_1d_regression_with_uncertainty(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        X_dense=X_dense, 
        y_pred_mean=y_pred_mean, 
        y_pred_std=y_pred_std
    )
    
    # Plot true function without noise
    X_true = np.linspace(-5, 5, 500)
    y_true = X_true**3 - 2 * X_true**2 + 0.5 * X_true
    
    plt.figure(figsize=(14, 6))
    plt.plot(X_true, y_true, 'k--', lw=2, label='True Function (no noise)')
    plt.plot(X_dense, y_pred_mean, 'r-', lw=2, label='BNN Prediction')
    plt.fill_between(
        X_dense.flatten(), 
        (y_pred_mean - 2 * y_pred_std).flatten(), 
        (y_pred_mean + 2 * y_pred_std).flatten(), 
        color='red', alpha=0.2, label='95% Confidence Interval'
    )
    plt.scatter(X_train, y_train, c='blue', s=20, alpha=0.6, label='Training Data')
    plt.scatter(X_test, y_test, c='green', s=20, alpha=0.6, label='Test Data')
    
    # Highlight the gap regions
    plt.axvspan(-2, -1, color='gray', alpha=0.2)
    plt.axvspan(1, 2, color='gray', alpha=0.2)
    plt.text(-1.5, max(y_train), 'Gap Region', ha='center', va='bottom')
    plt.text(1.5, max(y_train), 'Gap Region', ha='center', va='bottom')
    
    plt.title('Bayesian Neural Network Regression with Uncertainty')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Analyze uncertainty in different regions
    print("\nAnalyzing uncertainty in different regions...")
    analyze_uncertainty_by_region(X_dense, y_pred_std)


def analyze_uncertainty_by_region(X, y_std):
    """
    Analyze and plot the uncertainty in different regions of the input space.
    
    Args:
        X: Input feature array
        y_std: Prediction standard deviations
    """
    X_flat = X.flatten()
    y_std_flat = y_std.flatten()
    
    # Define regions
    gap_region_1 = (-2 <= X_flat) & (X_flat <= -1)
    gap_region_2 = (1 <= X_flat) & (X_flat <= 2)
    extrapolation_region_left = X_flat < -4
    extrapolation_region_right = X_flat > 4
    normal_region = ~(gap_region_1 | gap_region_2 | extrapolation_region_left | extrapolation_region_right)
    
    # Calculate average uncertainty in each region
    regions = {
        'Gap Region (-2 to -1)': gap_region_1,
        'Gap Region (1 to 2)': gap_region_2,
        'Extrapolation Left (< -4)': extrapolation_region_left,
        'Extrapolation Right (> 4)': extrapolation_region_right,
        'Training Region': normal_region
    }
    
    region_uncertainties = {}
    for name, mask in regions.items():
        if np.any(mask):
            region_uncertainties[name] = np.mean(y_std_flat[mask])
        else:
            region_uncertainties[name] = 0
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    names = list(region_uncertainties.keys())
    values = list(region_uncertainties.values())
    
    plt.bar(names, values, color=['orange', 'orange', 'red', 'red', 'blue'])
    plt.ylabel('Average Uncertainty (Standard Deviation)')
    plt.title('Uncertainty by Region')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    print("Average uncertainty by region:")
    for name, value in region_uncertainties.items():
        print(f"{name}: {value:.4f}")
    
    
if __name__ == "__main__":
    main() 