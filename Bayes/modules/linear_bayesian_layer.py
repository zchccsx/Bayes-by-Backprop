import torch
import torch.nn as nn
import torch.nn.functional as F

from Bayes.modules.base_bayesian_module import BayesianModule
from Bayes.modules.weight_sampler import GaussianVariationalPosterior, ScaleMixtureGaussianPrior

class BayesianLinear(BayesianModule):
    """
    Bayesian Linear layer implementing the approach described in 
    "Weight Uncertainty in Neural Networks" (Bayes by Backprop) paper.
    
    This layer performs linear transformations with weights drawn from
    a variational posterior distribution.
    """
    
    def __init__(self, 
                in_features, 
                out_features, 
                bias=True, 
                prior_sigma1=0.1, 
                prior_sigma2=0.4, 
                prior_pi=0.5, 
                posterior_mu_init=0, 
                posterior_rho_init=-6.0):
        """
        Initialize the Bayesian Linear layer.
        
        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            bias (bool): Whether to include bias parameters
            prior_sigma1 (float): Standard deviation for the first Gaussian in the prior
            prior_sigma2 (float): Standard deviation for the second Gaussian in the prior
            prior_pi (float): Mixture weight for the prior
            posterior_mu_init (float): Initial value for the posterior mean
            posterior_rho_init (float): Initial value for the posterior rho parameter
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # Initialize weight parameters (mu and rho for the posterior)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(posterior_mu_init, 0.1))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(posterior_rho_init, 0.1))
        
        # Initialize bias parameters if needed
        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(posterior_mu_init, 0.1))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(posterior_rho_init, 0.1))
        else:
            # Register as buffer to be included in state_dict but not as parameters
            self.register_buffer('bias_mu', None)
            self.register_buffer('bias_rho', None)
        
        # Initialize prior distribution
        self.weight_prior = ScaleMixtureGaussianPrior(prior_sigma1, prior_sigma2, prior_pi)
        if bias:
            self.bias_prior = ScaleMixtureGaussianPrior(prior_sigma1, prior_sigma2, prior_pi)
        
        # Initialize weight and bias samples
        self.weight = None
        self.bias = None
        
    def forward(self, x):
        """
        Forward pass through the Bayesian Linear layer.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Sample weights if not frozen or if first forward pass
        if self.training or not self.frozen or self.weight is None:
            self.weight = self._sample_weights()
            if self.use_bias:
                self.bias = self._sample_bias()
        
        # Perform linear transformation
        if self.use_bias:
            return F.linear(x, self.weight, self.bias)
        else:
            return F.linear(x, self.weight)
    
    def _sample_weights(self):
        """
        Sample weights from the posterior distribution.
        
        Returns:
            torch.Tensor: Sampled weights
        """
        if self.frozen:
            # Use posterior mean during evaluation
            return self.weight_mu
        else:
            # Sample from posterior
            weight_posterior = GaussianVariationalPosterior(self.weight_mu, self.weight_rho)
            return weight_posterior.sample()
    
    def _sample_bias(self):
        """
        Sample bias from the posterior distribution.
        
        Returns:
            torch.Tensor: Sampled bias
        """
        if self.frozen or not self.use_bias:
            # Use posterior mean during evaluation
            return self.bias_mu
        else:
            # Sample from posterior
            bias_posterior = GaussianVariationalPosterior(self.bias_mu, self.bias_rho)
            return bias_posterior.sample()
    
    def kl_divergence(self):
        """
        Calculate the KL divergence between the posterior and prior distributions.
        
        Returns:
            torch.Tensor: KL divergence value
        """
        # Weight KL divergence
        weight_kl = self._kl_divergence(self.weight_mu, self.weight_rho, self.weight_prior)
        
        # Bias KL divergence (if bias is used)
        if self.use_bias:
            bias_kl = self._kl_divergence(self.bias_mu, self.bias_rho, self.bias_prior)
            return weight_kl + bias_kl
        else:
            return weight_kl
    
    def _kl_divergence(self, mu, rho, prior):
        """
        Helper method to calculate KL divergence given mean and rho parameters.
        
        Args:
            mu (torch.Tensor): Mean of the posterior
            rho (torch.Tensor): Rho parameter of the posterior
            prior: Prior distribution object
            
        Returns:
            torch.Tensor: KL divergence
        """
        # Create posterior
        posterior = GaussianVariationalPosterior(mu, rho)
        
        # Sample from posterior
        sample = posterior.sample()
        
        # Calculate log probabilities
        log_posterior = posterior.log_prob(sample)
        log_prior = prior.log_prob(sample)
        
        # KL divergence
        return (log_posterior - log_prior).sum() 