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

class ToyRegressionExample:
    """
    A class for demonstrating Bayesian Neural Network regression on toy data.
    
    This class handles:
    - Data generation
    - Model creation and training
    - Prediction and uncertainty estimation
    - Visualization of results
    """
    
    def __init__(self, 
                 n_samples=300, 
                 noise_level=0.5, 
                 test_gaps=True,
                 hidden_dims=[32, 32],
                 learning_rate=0.001,
                 batch_size=32,
                 epochs=1000,
                 train_samples=5,
                 predict_samples=100,
                 random_seed=42):
        """
        Initialize the toy regression example.
        
        Args:
            n_samples: Number of data points to generate
            noise_level: Amount of noise to add to data
            test_gaps: Whether to create gaps in data to test uncertainty
            hidden_dims: List of hidden layer dimensions for the BNN
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            epochs: Number of training epochs
            train_samples: Number of forward passes during training
            predict_samples: Number of forward passes during prediction
            random_seed: Random seed for reproducibility
        """
        # Store configuration
        self.n_samples = n_samples
        self.noise_level = noise_level
        self.test_gaps = test_gaps
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_samples = train_samples
        self.predict_samples = predict_samples
        
        # Set random seed
        self.set_random_seed(random_seed)
        
        # Initialize storage for data and results
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.trainer = None
        self.train_losses = None
        self.val_losses = None
        self.X_dense = None
        self.y_pred_mean = None
        self.y_pred_std = None
        self.uncertainties_by_region = None
        
    @staticmethod
    def set_random_seed(seed):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def generate_data(self):
        """Generate synthetic regression data."""
        print("Generating synthetic data...")
        
        # Generate inputs in the range [-4, 4]
        X = np.random.uniform(-4, 4, self.n_samples)
        
        if self.test_gaps:
            # Create data gaps to test uncertainty behavior
            # Remove samples in the ranges [-2, -1] and [1, 2]
            X = np.array([x for x in X if not (-2 <= x <= -1 or 1 <= x <= 2)])
            
        # Sort X for easier visualization later
        X = np.sort(X)
        
        # True function: y = x^3 - 2x^2 + 0.5x + noise
        y = X**3 - 2 * X**2 + 0.5 * X + np.random.normal(0, self.noise_level, len(X))
        
        # Reshape for sklearn compatibility
        X = X.reshape(-1, 1)
        y = y.reshape(-1, 1)
        
        self.X = X
        self.y = y
        return X, y
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and test sets."""
        if self.X is None or self.y is None:
            self.generate_data()
            
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # Convert to PyTorch tensors
        self.X_train_tensor = torch.FloatTensor(self.X_train)
        self.y_train_tensor = torch.FloatTensor(self.y_train)
        self.X_test_tensor = torch.FloatTensor(self.X_test)
        self.y_test_tensor = torch.FloatTensor(self.y_test)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def create_model(self):
        """Create and initialize the Bayesian Neural Network model."""
        print("\nCreating Bayesian Neural Network...")
        self.model = BayesianRegressor(input_dim=1, hidden_dims=self.hidden_dims, output_dim=1)
        self.trainer = BayesianRegressionTrainer(
            model=self.model,
            learning_rate=self.learning_rate
        )
        return self.model, self.trainer
    
    def train_model(self):
        """Train the Bayesian Neural Network model."""
        if self.trainer is None:
            self.create_model()
            
        print("\nTraining the model...")
        self.train_losses, self.val_losses = self.trainer.train(
            X_train=self.X_train_tensor, 
            y_train=self.y_train_tensor,
            batch_size=self.batch_size,
            epochs=self.epochs,
            samples_nbr=self.train_samples
        )
        return self.train_losses, self.val_losses
    
    def predict(self, X_dense_start=-5, X_dense_end=5, n_points=500):
        """Make predictions with uncertainty estimation."""
        if self.trainer is None:
            raise ValueError("Model must be trained before making predictions")
            
        print("\nMaking predictions with uncertainty...")
        self.X_dense = np.linspace(X_dense_start, X_dense_end, n_points).reshape(-1, 1)
        X_dense_tensor = torch.FloatTensor(self.X_dense)
        
        self.y_pred_mean, self.y_pred_std, _ = self.trainer.predict(
            X_dense_tensor, num_samples=self.predict_samples
        )
        return self.y_pred_mean, self.y_pred_std, self.X_dense
    
    def plot_training_loss(self):
        """Plot the training loss curve."""
        if self.train_losses is None:
            raise ValueError("Model must be trained before plotting loss")
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_predictions(self):
        """Plot predictions with uncertainty compared to true data."""
        if self.y_pred_mean is None or self.y_pred_std is None:
            raise ValueError("Predictions must be made before plotting")
            
        # Plot using the utility function
        plot_1d_regression_with_uncertainty(
            X_train=self.X_train, y_train=self.y_train,
            X_test=self.X_test, y_test=self.y_test,
            X_dense=self.X_dense, 
            y_pred_mean=self.y_pred_mean, 
            y_pred_std=self.y_pred_std
        )
        
        # Plot with true function without noise
        X_true = np.linspace(-5, 5, 500)
        y_true = X_true**3 - 2 * X_true**2 + 0.5 * X_true
        
        plt.figure(figsize=(14, 6))
        plt.plot(X_true, y_true, 'k--', lw=2, label='True Function (no noise)')
        plt.plot(self.X_dense, self.y_pred_mean, 'r-', lw=2, label='BNN Prediction')
        plt.fill_between(
            self.X_dense.flatten(), 
            (self.y_pred_mean - 2 * self.y_pred_std).flatten(), 
            (self.y_pred_mean + 2 * self.y_pred_std).flatten(), 
            color='red', alpha=0.2, label='95% Confidence Interval'
        )
        plt.scatter(self.X_train, self.y_train, c='blue', s=20, alpha=0.6, label='Training Data')
        plt.scatter(self.X_test, self.y_test, c='green', s=20, alpha=0.6, label='Test Data')
        
        # Highlight the gap regions
        plt.axvspan(-2, -1, color='gray', alpha=0.2)
        plt.axvspan(1, 2, color='gray', alpha=0.2)
        plt.text(-1.5, max(self.y_train), 'Gap Region', ha='center', va='bottom')
        plt.text(1.5, max(self.y_train), 'Gap Region', ha='center', va='bottom')
        
        plt.title('Bayesian Neural Network Regression with Uncertainty')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def analyze_uncertainty(self):
        """Analyze and plot uncertainty in different regions."""
        if self.y_pred_std is None:
            raise ValueError("Predictions must be made before analyzing uncertainty")
            
        print("\nAnalyzing uncertainty in different regions...")
        X_flat = self.X_dense.flatten()
        y_std_flat = self.y_pred_std.flatten()
        
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
        
        self.uncertainties_by_region = {}
        for name, mask in regions.items():
            if np.any(mask):
                self.uncertainties_by_region[name] = np.mean(y_std_flat[mask])
            else:
                self.uncertainties_by_region[name] = 0
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        names = list(self.uncertainties_by_region.keys())
        values = list(self.uncertainties_by_region.values())
        
        plt.bar(names, values, color=['orange', 'orange', 'red', 'red', 'blue'])
        plt.ylabel('Average Uncertainty (Standard Deviation)')
        plt.title('Uncertainty by Region')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        print("Average uncertainty by region:")
        for name, value in self.uncertainties_by_region.items():
            print(f"{name}: {value:.4f}")
            
        return self.uncertainties_by_region
    
    def run_full_example(self):
        """Run the complete example pipeline."""
        self.generate_data()
        self.split_data()
        self.create_model()
        self.train_model()
        self.plot_training_loss()
        self.predict()
        self.plot_predictions()
        self.analyze_uncertainty()
        

def main():
    """Run a demonstration of the toy regression example."""
    print("Bayesian Neural Network 1D Regression Example")
    print("--------------------------------------------")
    
    # Create and run the example with default parameters
    example = ToyRegressionExample(
        n_samples=300,
        noise_level=0.5,
        epochs=1000  # Reduced for quicker demonstration
    )
    example.run_full_example()
    
    # Access results for further analysis if needed
    # model = example.model
    # uncertainties = example.uncertainties_by_region
    
    
if __name__ == "__main__":
    main() 