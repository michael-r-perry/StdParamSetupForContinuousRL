import numpy as np
import torch


class RolloutBuffer:

    def __init__(self):
        self.obs = []
        self.acts = []
        self.log_probs = []
        self.rews = []
        self.lens = []
        self.vals = []
        self.dones = []

    def clear(self):
        """
            Clear all of the buffer properties to empty lists
        """
        self.obs = []
        self.acts = []
        self.log_probs = []
        self.rews = []
        self.lens = []
        self.vals = []
        self.dones = []

    def extend_obs(self, ep_obs):
        """
            Extend self.obs with the values from the list of
            episode observations (only if self.obs is list)
        """
        if not isinstance(self.obs, list):
            raise Exception("Tried to extend RolloutBuffer.obs when property was not of type list.")
        
        self.obs += ep_obs

    def extend_acts(self, ep_acts):
        """
            Extend self.acts with the values from the list of 
            episode actions (only if self.acts is list)
        """
        if not isinstance(self.acts, list):
            raise Exception("Tried to extend RolloutBuffer.acts when property was not of type list.")
        
        self.acts += ep_acts

    def extend_log_probs(self, ep_log_probs):
        """
            Extend self.log_probs with the values from the list of 
            episode log probabilities (only if self.log_probs is list)
        """
        if not isinstance(self.log_probs, list):
            raise Exception("Tried to extend RolloutBuffer.log_probs when property was not of type list.")
        
        self.log_probs += ep_log_probs

    def append_rews(self, ep_rews):
        """
            Append self.rews with the values from the list of 
            episode rewards (only if self.rews is list)
        """
        if not isinstance(self.rews, list):
            raise Exception("Tried to extend RolloutBuffer.rews when property was not of type list.")
        
        self.rews.append(ep_rews)

    def append_lens(self, ep_lens):
        """
            Append self.lens with the values from the list of
            episode lengths (only if self.lens is list)
        """
        if not isinstance(self.lens, list):
            raise Exception("Tried to extend RolloutBuffer.lens when property was not of type list.")
        
        self.lens.append(ep_lens)

    def append_vals(self, ep_vals):
        """
            Append self.vals with the values from the list of 
            episode Values (only if self.vals is list)
        """
        if not isinstance(self.vals, list):
            raise Exception("Tried to extend RolloutBuffer.vals when property was not of type list.")
        
        self.vals.append(ep_vals)

    def append_dones(self, ep_dones):
        """
            Append self.dones with the values from the list of 
            episode Dones (only if self.dones is list)
        """
        if not isinstance(self.dones, list):
            raise Exception("Tried to extend RolloutBuffer.dones when property was not of type list.")
        
        self.dones.append(ep_dones)

    def to_tensor(self, device):
        """
            Move obs, acts, and log_probs to PyTorch Tensors on given device.
        """
        # Parse obs to np.array if in list
        if isinstance(self.obs, list):
            self.obs = np.array(self.obs)
        
        self.obs = torch.tensor(self.obs, dtype=torch.float, device=device)
        self.acts = torch.tensor(self.acts, dtype=torch.float, device=device)
        self.log_probs = torch.tensor(self.log_probs, dtype=torch.float, device=device).flatten()