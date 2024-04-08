import os
import pathlib
import torch
import numpy as np
from typing import Any, Optional, Union
from torch import nn, device, Tensor
from torch.optim import Adam
from torch.distributions import Categorical, MultivariateNormal
from Utils import Standardizer


class ActorCriticAgent():
    """
        This is the PPO agent class that has both the actor and critic networks
        and is used in the PPO.py file
    """
    def __init__(
            self, 
            policy_class: nn.Module, 
            model_dir: str,
            obs_dim: int, 
            hid_dim: int, 
            act_dim: int,
            lr: float, 
            device: device,
            deterministic: bool = False
        ) -> None:
        """
            Initializes the Actor and Critic models.

            Parameters:
                policy_class - the policy class to use for our actor/critic networks.
                model_dir - directory where the models will be saved
                obs_dim - input dimensions as an int
                hidden_dim - hidden layer dimensions as an int
                act_dim - output dimensions as an int
                lr - learning rate of optimizers
                device - Device (cpu, cuda, ...) on which the code should be run.
                deterministic - Flag for whether to argmax or sample action outputs

            Return:
                None
        """
        # Set the device
        self.device = device

        # Save the model directory
        self.model_dir = model_dir

        # Extract dimension specs
        self.obs_dim = obs_dim
        self.hid_dim = hid_dim
        self.act_dim = act_dim

        # Initialize actor and critic networks
        self.actor = policy_class(self.obs_dim, self.hid_dim, self.act_dim)
        self.actor.to(self.device)
        self.critic = policy_class(self.obs_dim, self.hid_dim, 1)
        self.critic.to(self.device)

        # Set learning rate
        self.lr = lr

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        # Set Deterministic Flag
        self.deterministic = deterministic

        # Set Standardizer to None
        self.standardizer = None
    
    def get_action(
            self, 
            obs: Any, 
            device: Optional[Union[device, str]] = None
            ) -> tuple[Any, float]:
        """
            Queries an action from the actor network, should be called from _episode

            Parameters:
                obs - the observation at the current timestep
                device - Device (cpu, cuda, ...) on which the code should be run.
            
            Return:
                action - the action to take, as a numpy array or scalar
                log_prob - the log probability of the selected action in the distribution
        """
        # If no device is given the objects assigned device
        if device is None:
            device = self.device
        # Put observations into Tensor if not
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        # Standardize observation if needed
        if self.standardizer is not None:
            obs = self.standardizer.standardize(obs)
        # Query the actor network for a mean action
        probs = self.actor(obs)

        # Create a distribution with the mean action and std from the covariance matrix above.
        # For more information on how this distribution works, check out Andrew Ng's lecture on it:
        # https://www.youtube.com/watch?v=JjB58InuTqM
        #dist = MultivariateNormal(mean, self.cov_mat)
        dist = Categorical(logits=probs)

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # If we're testing, just return the deterministic action. Sampling should only be for training
        # as our "exploration" factor.
        if self.deterministic:
            #return mean, 1
            return torch.argmax(probs), 1
        
        # Return the sampled action and the log probability of that action in our distribution
        return action, log_prob, dist.entropy()
    
    def get_value(
            self,
            obs: Any,
            device: Optional[Union[device, str]] = None
            ) -> Any:
        """
            Queries a value from the critic network, should be called from _episode or train

            Parameters:
                obs - the observation at teh current timestep
                device - Device (cpu, cuda, ...) on which the code should be run.
            
            Return:
                value - the value of the observation,
        """
        # If no device is given the objects assigned device
        if device is None:
            device = self.device
        # Put observations into Tensor if not
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        # Standardize observation if needed
        if self.standardizer is not None:
            obs = self.standardizer.standardize(obs)
        # Query the critic network for a value
        value = self.critic(obs)
        return value


    def evaluate(
            self, 
            batch_obs: Tensor, 
            batch_acts: Tensor
            ) -> tuple[Tensor, Tensor, Tensor]:
        """
            Estimate the values of each observation, and the log probs of 
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.

            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
            
            Return:
                V - the predicted values of batch_obs
                log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
                entropy - 
        """
        # Standardize observations if needed
        if self.standardizer is not None:
            batch_obs = self.standardizer.standardize(batch_obs)
        # Query critic network for a value V for each batch_obs.
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        probs = self.actor(batch_obs)
        dist = Categorical(logits=probs)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch and
        # the log probabilities log_pobs and entropy of each action in the batch
        return V, log_probs, dist.entropy()

    def backward(self, actor_loss: Tensor, critic_loss: Tensor, max_grad_norm: float) -> Tensor:
        """
            Calculate gradients and perform backward propagation for actor and critic network

            Parameters:
                actor_loss: The loss of each action from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, 1)
                critic_loss: The loss of each value from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, 1)
                max_grad_norm: Gradient clipping threshold
            
            Return:
                actor_loss:
        """
        # Calculate gradients and perform backward propagation for actor network
        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        # Gradient Clipping with given threshold
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
        self.actor_optim.step()

        # Calculate gradients and perform backward propagation for critic network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
        self.critic_optim.step()

        return actor_loss.detach()
    
    def init_standardizer(self, state_min_max: list) -> None:
        """
            Initialize a standardizer for the agent by mapping observation/state 
            properties between their respective mins and maxes to values between 
            [-1, 1].
        """
        self.standardizer = Standardizer(state_min_max)
    
    def save(self):
        """
            Save actor and critic models to their given directory.
        """
        if not os.path.exists(self.model_dir):
            path = pathlib.Path(self.model_dir)
            path.mkdir(parents=True)
        actor_path = os.path.join(self.model_dir, "actor.pth")
        critic_path = os.path.join(self.model_dir, "critic.pth")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def safe_load(self):
        """
            Load actor and critic models if they are available in model directory
        """
        if not os.path.exists(self.model_dir):
            return
        print("Previous model found!")
        print("Loading saved model at:", self.model_dir)
        actor_path = os.path.join(self.model_dir, "actor.pth")
        critic_path = os.path.join(self.model_dir, "critic.pth")
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))

    def log_models(self, writer, num_iter):
        """
            Log model statistic histograms at given iteration number
        """
        self.actor.log_model(writer, "model/actor/", num_iter)
        self.critic.log_model(writer, "model/critic/", num_iter)

    def set_device(self):
        """
            Moves actor/critic models onto the device given when initialized.
            This allows the GPU to be utilized during backpropagation (not multiprocessing).
        """
        self.actor.to(self.device)
        self.critic.to(self.device)

    def set_lr(self, lr: float) -> None:
        """
            Set the learning rate of both the actor and critic optimizers.
        """
        self.lr = lr
        self.actor_optim.param_groups[0]["lr"] = lr
        self.critic_optim.param_groups[0]["lr"] = lr