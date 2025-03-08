import torch
import torch.nn as nn
import torch.nn.functional as F

from Bayes.losses.kl_divergence import kl_divergence_from_nn
from Bayes.modules.base_bayesian_module import BayesianModule

def variational_estimator(model_class):
    """
    Decorator that adds Bayesian neural network functionality to a nn.Module class.
    
    This decorator adds methods for:
    - Calculating the total KL divergence of the network
    - Computing the ELBO loss for training
    - Freezing/unfreezing the network for deterministic inference
    
    Args:
        model_class (type): A nn.Module subclass
        
    Returns:
        type: The decorated model class with additional methods
    """
    
    def _kl_divergence(self):
        """
        Calculate the total KL divergence of the network.
        
        Returns:
            torch.Tensor: The total KL divergence
        """
        return kl_divergence_from_nn(self)
    
    def _sample_elbo(self, inputs, labels, criterion, sample_nbr=3, complexity_cost_weight=1.0):
        """
        Sample the ELBO loss for a batch of data by sampling from the posterior.
        
        The ELBO (Evidence Lower Bound) is the standard loss function for variational inference.
        It consists of the expected log-likelihood term (data fit) and the KL divergence term (complexity cost).
        
        Args:
            inputs (torch.Tensor): Input data batch
            labels (torch.Tensor): Target labels batch
            criterion: Loss function for the data fit term
            sample_nbr (int): Number of Monte Carlo samples
            complexity_cost_weight (float): Weight for the KL divergence term
            
        Returns:
            torch.Tensor: ELBO loss
        """
        # Initialize losses
        loss = 0
        
        # Monte Carlo estimation of the expected log-likelihood
        for _ in range(sample_nbr):
            outputs = self(inputs)
            loss += criterion(outputs, labels)
            
        # Average over samples
        loss = loss / sample_nbr
        
        # Add weighted KL divergence term
        kl = self.kl_divergence()
        
        # Return ELBO loss
        return loss + complexity_cost_weight * kl
    
    def _freeze(self):
        """
        Freeze the network to use only the mean of the posterior distributions.
        This makes the network deterministic for evaluation.
        """
        for module in self.modules():
            if isinstance(module, BayesianModule):
                module.freeze_()
    
    def _unfreeze(self):
        """
        Unfreeze the network to sample from the posterior distributions.
        This enables stochastic behavior for training.
        """
        for module in self.modules():
            if isinstance(module, BayesianModule):
                module.unfreeze_()
    
    def _predict_with_uncertainty(self, inputs, num_samples=20):
        """
        Make predictions with uncertainty estimation.
        
        Args:
            inputs (torch.Tensor): Input data
            num_samples (int): Number of Monte Carlo samples
            
        Returns:
            tuple: (mean_prediction, std_prediction, all_predictions)
        """
        # Ensure model is in eval mode but not frozen
        self.eval()
        self.unfreeze()
        
        # Store original training state
        training_state = self.training
        
        # Set to evaluation mode but keep stochastic behavior
        self.train(False)
        
        # Initialize list for predictions
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                predictions.append(self(inputs))
                
        # Stack predictions (samples x batch_size x output_dim)
        predictions = torch.stack(predictions)
        
        # Calculate mean and standard deviation along sample dimension
        mean_prediction = predictions.mean(dim=0)
        std_prediction = predictions.std(dim=0)
        
        # Restore original training state
        self.train(training_state)
        
        return mean_prediction, std_prediction, predictions
    
    # Add methods to the model class
    setattr(model_class, 'kl_divergence', _kl_divergence)
    setattr(model_class, 'sample_elbo', _sample_elbo)
    setattr(model_class, 'freeze', _freeze)
    setattr(model_class, 'unfreeze', _unfreeze)
    setattr(model_class, 'predict_with_uncertainty', _predict_with_uncertainty)
    
    return model_class 