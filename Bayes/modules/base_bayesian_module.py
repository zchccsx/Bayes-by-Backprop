import torch
import torch.nn as nn

class BayesianModule(nn.Module):
    """
    Base class for Bayesian neural network modules.
    Implements the basic functionality that will be shared across all Bayesian layers.
    """
    
    def __init__(self):
        super().__init__()
        self.frozen = False
    
    def freeze_(self):
        """
        Freezes the layer so that during inference only the mean of the posterior
        distribution is used for prediction. This makes the network deterministic
        for evaluation.
        """
        self.frozen = True
    
    def unfreeze_(self):
        """
        Unfreezes the layer so that during training/inference, the weights are sampled
        from the posterior distribution.
        """
        self.frozen = False
    
    def kl_divergence(self):
        """
        Calculate the KL divergence between the posterior and prior distributions.
        
        Returns:
            torch.Tensor: KL divergence value
        """
        # To be implemented by child classes
        return torch.tensor(0.) 