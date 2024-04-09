from typing import Union, Tuple, TypeVar, Optional

import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torch.distributions import Categorical, Normal

SelfDistribution = TypeVar("SelfDistribution", bound="Distribution")
SelfDiagGaussianDistribution = TypeVar("SelfDiagGaussianDistribution", bound="DiagGaussianDistribution")
SelfCategoricalDistribution = TypeVar("SelfCategoricalDistribution", bound="CategoricalDistribution")


class Distribution():
    """Base class for distributions."""

    def __init__(self):
        self.distribution = None
    
    def proba_distribution_params(self, *args, **kwargs) -> Optional[nn.Parameter]:
        """
            Create the parameters that represent the distribution if applicable.

            Subclasses must define this, but the arguments and return type vary between 
            concrete classes
        """
        raise NotImplementedError
    
    def proba_distribution(self: SelfDistribution, *args, **kwargs) -> SelfDistribution:
        """
            Set parameters of the distribution.

            :return: self
        """
        raise NotImplementedError
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
            Returns the log likelihood

            :param x: the taken action
            :return: The log likelihood of the distribution
        """
        raise NotImplementedError
    
    def entropy(self) -> Optional[torch.Tensor]:
        """
            Returns Shannon's entropy of the probability

            :return: the entropy, or None if no analytical form is known
        """
        raise NotImplementedError
    
    def sample(self) -> torch.Tensor:
        """
            Returns a sample from the probability distribution

            :return: the stochastic action
        """
        raise NotImplementedError
    
    def mode(self) -> torch.Tensor:
        """
            Returns the most likely action (deterministic output)
            from the probability distribution

            :return: the stochastic action
        """
        raise NotImplementedError
    
    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        """
            Return actions according to the probability distribution.

            :param deterministic:
            :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()
    

def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
        Continuous actions are usually considered to be independent,
        so we can sum components of the ``log_prob`` or the entropy.

        :param tensor: shape: (n_batch, n_actions) or (n_batch,)
        :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


class DiagGaussianDistribution(Distribution):
    """
        Guassian distribution with diagonal covariance matrix, for continuous actions.

        :param action_dim: Dimension of the action space.
    """

    def __init__(self, action_dim: int, device: torch.device):
        super().__init__()
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None
        self.device = device

    def proba_distribution_params(self, log_std_init: float = 0.0) -> nn.Parameter:
        """
            Create the parameter that represents the distribution:
            The parameter will be the standard deviation (log std 
            in fact to allow negative values)

            :param log_std_init: Initial value for the log standard deviation
            :return:
        """
        log_std = nn.Parameter(torch.ones(self.action_dim) * log_std_init, requires_grad=True)
        print(log_std.device)
        return log_std
    
    def proba_distribution(
        self: SelfDiagGaussianDistribution, mean_actions: torch.Tensor, log_std: torch.Tensor
    ) -> SelfDiagGaussianDistribution:
        """
            Create teh distribution given its parameters (mean, std)

            :param mean_actions:
            :param log_std:
            :return:
        """
        action_std = torch.ones_like(mean_actions, device=self.device) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self
    
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
            Get the log probabilities of actions according to the distribution.
            Note that you must first call the ``proba_distribution()`` method.

            :param actions:
            :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)
    
    def entropy(self) -> Optional[torch.Tensor]:
        return sum_independent_dims(self.distribution.entropy())
    
    def sample(self) -> torch.Tensor:
        # Reparameterization trick to pass gradients (rsample vs. sample)
        return self.distribution.rsample()
    
    def mode(self) -> torch.Tensor:
        return self.distribution.mean
    

class CategoricalDistribution(Distribution):
    """
        Categorical distribution for discrete actions.

        :param action_dim: Number of discrete actions
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def proba_distribution_params(self) -> None:
        """
            The Categorical Distribution does not contain any parameters
        """
        return None
    
    def proba_distribution(self: SelfCategoricalDistribution, action_logits: torch.Tensor) -> SelfCategoricalDistribution:
        self.distribution = Categorical(logits=action_logits)
        return self
    
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions)
    
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()
    
    def sample(self) -> torch.Tensor:
        return self.distribution.sample()
    
    def mode(self) -> torch.Tensor:
        return torch.argmax(self.distribution.probs, dim=1)