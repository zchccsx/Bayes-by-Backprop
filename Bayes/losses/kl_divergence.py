import torch

from Bayes.modules.base_bayesian_module import BayesianModule

def kl_divergence_from_nn(model):
    """
    Calculate the KL divergence for all Bayesian layers in a neural network.
    
    Args:
        model (torch.nn.Module): The Bayesian neural network
        
    Returns:
        torch.Tensor: The total KL divergence of the network
    """
    # Initialize KL divergence
    kl = 0
    
    # Iterate through all modules
    for module in model.modules():
        if isinstance(module, BayesianModule):
            kl += module.kl_divergence()
    
    return kl 