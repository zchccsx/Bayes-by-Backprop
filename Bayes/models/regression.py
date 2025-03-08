import torch
import torch.nn as nn
import torch.nn.functional as F

from Bayes.modules import BayesianLinear
from Bayes.utils import variational_estimator

@variational_estimator
class BayesianRegressor(nn.Module):
    """
    Bayesian Neural Network for regression tasks.
    Uses BayesianLinear layers and implements the variational inference approach.
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=1):
        """
        Initialize a Bayesian Neural Network for regression.
        
        Args:
            input_dim (int): Number of input features
            hidden_dims (list): List of hidden layer dimensions
            output_dim (int): Number of output dimensions (usually 1 for regression)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build network architecture
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.append(BayesianLinear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(BayesianLinear(prev_dim, output_dim))
        
        # Create sequential model
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output predictions
        """
        return self.layers(x) 