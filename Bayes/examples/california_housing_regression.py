import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import sys
sys.path.append("./")  # Replace with your actual path
import torch
import torch.nn as nn

from Bayes.models import BayesianRegressor
from Bayes.utils import (
    BayesianRegressionTrainer,
    prepare_data,
    evaluate_uncertainty,
    plot_predictions
)

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def main():
    print("Bayesian Neural Network Regression Example")
    print("-----------------------------------------")
    
    # Load dataset
    print("Loading California Housing dataset...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    feature_names = housing.feature_names
    
    print(f"Dataset shape: {X.shape}, Target shape: {y.shape}")
    print(f"Features: {feature_names}")
    X = X[:1000]
    y = y[:1000]
    # Prepare data
    print("\nPreparing data...")
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = prepare_data(
        X, y, test_size=0.2, random_state=42, scale=True
    )
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Create model
    print("\nCreating Bayesian Neural Network...")
    input_dim = X_train.shape[1]
    hidden_dims = [64, 32, 16]  # Network architecture
    model = BayesianRegressor(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=1)
    
    print(f"Model architecture: Input({input_dim}) -> Hidden{hidden_dims} -> Output(1)")
    
    # Create trainer
    trainer = BayesianRegressionTrainer(
        model=model,
        learning_rate=0.001
    )
    
    # Train the model
    print("\nTraining the model...")
    train_losses, val_losses = trainer.train(
        X_train=X_train, 
        y_train=y_train,
        X_val=X_test,  # Using test set as validation for simplicity
        y_val=y_test,
        batch_size=64,
        epochs=1000,
        samples_nbr=3  # Number of MC samples for ELBO
    )
    
    # Plot training history
    print("\nPlotting training history...")
    trainer.plot_loss()
    
    # Make predictions with uncertainty
    print("\nMaking predictions with uncertainty...")
    y_pred_mean, y_pred_std, _ = trainer.predict(X_test, num_samples=100)
    
    # If we scaled the target, transform predictions back to original scale
    if scaler_y is not None:
        y_test_orig = scaler_y.inverse_transform(y_test)
        y_pred_mean_orig = scaler_y.inverse_transform(y_pred_mean)
        # Scale the standard deviation accordingly
        y_pred_std_orig = y_pred_std * scaler_y.scale_
    else:
        y_test_orig = y_test
        y_pred_mean_orig = y_pred_mean
        y_pred_std_orig = y_pred_std
    
    # Evaluate predictions
    print("\nEvaluating predictions...")
    metrics = evaluate_uncertainty(
        y_true=y_test_orig,
        y_pred_mean=y_pred_mean_orig,
        y_pred_std=y_pred_std_orig,
        std_multiplier=2  # 2 standard deviations (~95% confidence interval)
    )
    
    print(f"Mean Squared Error: {metrics['mse']:.4f}")
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"Confidence Interval Accuracy (95%): {metrics['ci_accuracy']*100:.2f}%")
    
    # Plot predictions with uncertainty
    print("\nPlotting predictions with uncertainty...")
    plot_predictions(
        y_true=y_test_orig,
        y_pred_mean=y_pred_mean_orig,
        y_pred_std=y_pred_std_orig,
        std_multiplier=2,
        max_samples=100  # Limit for clearer visualization
    )
    
    # Feature importance through predictive uncertainty
    print("\nAnalyzing feature importance through predictive uncertainty...")
    feature_importance = analyze_feature_importance(
        model=model,
        X_test=X_test,
        feature_names=feature_names,
        n_samples=50
    )


def analyze_feature_importance(model, X_test, feature_names, n_samples=50):
    """
    Analyze feature importance by measuring change in predictive uncertainty
    when perturbing each feature.
    
    Args:
        model: Trained BayesianRegressor model
        X_test: Test features
        feature_names: Names of features
        n_samples: Number of samples for uncertainty estimation
    """
    device = next(model.parameters()).device
    X = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    # Baseline prediction uncertainty
    model.eval()
    baseline_preds = []
    for _ in range(n_samples):
        with torch.no_grad():
            pred = model(X)
            baseline_preds.append(pred)
    
    baseline_preds = torch.stack(baseline_preds)
    baseline_std = baseline_preds.std(dim=0).mean().item()
    
    # Perturb each feature and measure change in uncertainty
    importance_scores = []
    for i in range(X.shape[1]):
        # Create perturbed input by adding noise to feature i
        X_perturbed = X.clone()
        feature_std = X[:, i].std().item()
        X_perturbed[:, i] += torch.normal(0, feature_std, size=X[:, i].shape).to(device)
        
        # Get predictions with perturbed feature
        perturbed_preds = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = model(X_perturbed)
                perturbed_preds.append(pred)
        
        perturbed_preds = torch.stack(perturbed_preds)
        perturbed_std = perturbed_preds.std(dim=0).mean().item()
        
        # Change in uncertainty as importance score
        importance = perturbed_std - baseline_std
        importance_scores.append(importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    importance_scores = np.array(importance_scores)
    
    # Sort by importance
    sorted_idx = np.argsort(importance_scores)
    plt.barh(np.array(feature_names)[sorted_idx], importance_scores[sorted_idx])
    plt.xlabel('Increase in Predictive Uncertainty')
    plt.title('Feature Importance - Effect on Model Uncertainty')
    plt.tight_layout()
    plt.show()
    
    return dict(zip(feature_names, importance_scores))


if __name__ == "__main__":
    main() 