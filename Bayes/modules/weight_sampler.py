import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GaussianVariationalPosterior:
    """
    Implements the Gaussian variational posterior distribution for Bayesian neural networks.
    The weights are parameterized using a mean (mu) and a variance parameter (rho), 
    which is transformed through a softplus function to ensure positive variance.
    """
    
    def __init__(self, mu, rho):
        """
        Initialize the Gaussian posterior with mean and untransformed variance.
        
        Args:
            mu (torch.Tensor): Mean of the Gaussian distribution
            rho (torch.Tensor): Untransformed variance parameter (variance = log(1 + exp(rho)))
        """
        self.mu = mu
        self.rho = rho
        
    def sample(self):
        """
        Sample weights from the posterior distribution using the reparameterization trick.
        
        Returns:
            torch.Tensor: Sampled weights
        """
        # Transform rho to obtain the standard deviation
        sigma = torch.log1p(torch.exp(self.rho))
        
        # Sample from standard normal and scale by sigma
        epsilon = torch.randn_like(self.mu)
        return self.mu + sigma * epsilon
    
    def log_prob(self, sample):
        """
        Calculate the log probability of a sample under the posterior.
        
        Args:
            sample (torch.Tensor): The weight sample to evaluate
            
        Returns:
            torch.Tensor: Log probability
        """
        # Transform rho to obtain the variance
        sigma = torch.log1p(torch.exp(self.rho))
        
        # Log probability of Gaussian
        return -0.5 * (
            math.log(2 * math.pi) + 
            2 * torch.log(sigma) + 
            ((sample - self.mu) / sigma)**2
        )

class ScaleMixtureGaussianPrior:
    """
    Implements a scale mixture Gaussian prior distribution as proposed in the 
    'Weight Uncertainty in Neural Networks' paper. This prior is a mixture of two 
    Gaussian distributions with different standard deviations.
    """
    
    def __init__(self, sigma1=0.1, sigma2=0.4, pi=0.5):
        """
        Initialize the scale mixture prior with two Gaussian components.
        
        Args:
            sigma1 (float): Standard deviation of the first Gaussian
            sigma2 (float): Standard deviation of the second Gaussian
            pi (float): Mixture weight for the first Gaussian
        """
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.pi = pi
        
    def log_prob(self, sample):
        """
        Calculate the log probability of a sample under the prior.
        
        Args:
            sample (torch.Tensor): The weight sample to evaluate
            
        Returns:
            torch.Tensor: Log probability
        """
        # Log prob for first Gaussian
        gaussian1 = -0.5 * (
            math.log(2 * math.pi) + 
            2 * math.log(self.sigma1) + 
            (sample / self.sigma1)**2
        )
        
        # Log prob for second Gaussian
        gaussian2 = -0.5 * (
            math.log(2 * math.pi) + 
            2 * math.log(self.sigma2) + 
            (sample / self.sigma2)**2
        )
        
        # Mixture log prob (using log-sum-exp trick for numerical stability)
        log_prior = torch.logsumexp(
            torch.stack([
                torch.log(torch.tensor(self.pi)) + gaussian1,
                torch.log(torch.tensor(1 - self.pi)) + gaussian2
            ]),
            dim=0
        )
        
        return log_prior 