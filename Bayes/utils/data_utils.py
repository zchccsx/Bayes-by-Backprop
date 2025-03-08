import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def prepare_data(X, y, test_size=0.2, random_state=42, scale=True):
    """
    Prepare data for Bayesian regression.
    
    Args:
        X: Features
        y: Target variable
        test_size: Proportion of test data
        random_state: Random seed for reproducibility
        scale: Whether to standardize the data
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler_X, scaler_y)
    """
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    if scale:
        # Scale features
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)
        
        # Scale targets (ensure y is 2D for StandardScaler)
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)
            
        y_train = scaler_y.fit_transform(y_train)
        y_test = scaler_y.transform(y_test)
    
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y

def evaluate_uncertainty(y_true, y_pred_mean, y_pred_std, std_multiplier=2):
    """
    Evaluate prediction uncertainty using confidence intervals.
    
    Args:
        y_true: True target values
        y_pred_mean: Mean predictions
        y_pred_std: Standard deviation of predictions
        std_multiplier: Multiplier for standard deviation (e.g., 2 for 95% CI)
        
    Returns:
        dict: Dictionary with metrics
    """
    # Compute confidence intervals
    ci_upper = y_pred_mean + (std_multiplier * y_pred_std)
    ci_lower = y_pred_mean - (std_multiplier * y_pred_std)
    
    # Check if true values are within confidence intervals
    within_ci = (y_true >= ci_lower) & (y_true <= ci_upper)
    ci_accuracy = np.mean(within_ci)
    
    # Calculate regression metrics
    mse = mean_squared_error(y_true, y_pred_mean)
    r2 = r2_score(y_true, y_pred_mean)
    
    metrics = {
        'mse': mse,
        'r2': r2,
        'ci_accuracy': ci_accuracy
    }
    
    return metrics

def plot_predictions(y_true, y_pred_mean, y_pred_std, std_multiplier=2, max_samples=100):
    """
    Plot predictions with uncertainty.
    
    Args:
        y_true: True target values
        y_pred_mean: Mean predictions
        y_pred_std: Standard deviation of predictions
        std_multiplier: Multiplier for standard deviation
        max_samples: Maximum number of samples to plot
    """
    # Limit number of samples for clearer visualization
    n_samples = min(len(y_true), max_samples)
    indices = np.random.choice(len(y_true), n_samples, replace=False)
    
    # Sort by true values for clearer visualization
    sort_idx = np.argsort(y_true[indices].flatten())
    indices = indices[sort_idx]
    
    # Get data to plot
    x = np.arange(n_samples)
    y = y_true[indices].flatten()
    means = y_pred_mean[indices].flatten()
    stds = y_pred_std[indices].flatten()
    
    # Compute confidence intervals
    ci_upper = means + (std_multiplier * stds)
    ci_lower = means - (std_multiplier * stds)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot predictions and confidence intervals
    plt.scatter(x, y, color='blue', alpha=0.7, label='True Values')
    plt.scatter(x, means, color='red', alpha=0.7, label='Predictions')
    plt.fill_between(x, ci_lower, ci_upper, color='red', alpha=0.2, label=f'{std_multiplier}σ Confidence Interval')
    
    # Customize plot
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value')
    plt.title('Bayesian Neural Network Predictions with Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_1d_regression_with_uncertainty(X_train, y_train, X_test, y_test, 
                                        X_dense, y_pred_mean, y_pred_std):
    """
    Plot 1D regression with uncertainty for different confidence levels.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        X_dense: Dense grid for visualization
        y_pred_mean: Mean predictions
        y_pred_std: Standard deviation of predictions
    """
    plt.figure(figsize=(14, 8))
    
    # Plot data points
    plt.scatter(X_train, y_train, c='blue', s=20, alpha=0.6, label='Training Data')
    plt.scatter(X_test, y_test, c='green', s=20, alpha=0.6, label='Test Data')
    
    # Plot prediction and multiple confidence intervals
    plt.plot(X_dense, y_pred_mean, 'r-', lw=2, label='BNN Prediction')
    
    std_multipliers = [1, 2, 3]
    alphas = [0.3, 0.2, 0.1]
    
    for i, std_mult in enumerate(std_multipliers):
        plt.fill_between(
            X_dense.flatten(), 
            (y_pred_mean - std_mult * y_pred_std).flatten(), 
            (y_pred_mean + std_mult * y_pred_std).flatten(), 
            color='red', alpha=alphas[i], 
            label=f'{std_mult}σ ({int(100 * (1 - np.exp(-0.5 * std_mult**2)))}% CI)'
        )
    
    plt.title('Bayesian Neural Network Regression with Different Confidence Intervals')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.show() 