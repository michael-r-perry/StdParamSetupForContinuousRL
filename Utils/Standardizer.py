import numpy as np
import torch

class Standardizer:
    def __init__(self, min_maxes):
        self.means = np.array([np.mean(x) for x in min_maxes])
        self.inv_stds = np.array([2.0 / abs(x[1] - x[0]) for x in min_maxes])
        self.diag_inv_stds = np.diag(self.inv_stds)

    def standardize(self, values):
        # Parse input to tensor if not already and get device
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, dtype=torch.float)
        device = torch.device("cuda" if values.get_device() >= 0 else "cpu")
        # Set the means and diagonal inverse standard deviations to tensors on given device
        means = torch.tensor(self.means, dtype=torch.float, device=device)
        diag_inv_stds = torch.tensor(self.diag_inv_stds, dtype=torch.float, device=device)
        if values.ndim > 1:
            means = torch.tile(means, (values.shape[0],1))
        # Return the standardize values
        return (values - means) @ diag_inv_stds