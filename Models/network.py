import torch
from torch import nn, Tensor
import torch.nn.functional as F


class FeedForwardNN(nn.Module):
    '''
        A standard two-hidden-layer Feed Forward Neural Network.
    '''
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        '''
            Initialize the network and set up the layers.

            Parameters:
                in_dim - input dimensions as an int
                hidden_dim - hidden layer dimensions as an int
                out_dim - output dimensions as an int
            
            Return:
                None
        '''
        super(FeedForwardNN, self).__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, out_dim)

    def forward(self, obs: Tensor) -> Tensor:
        '''
            Runs a forward pass on the neural network.

            Parameters:
                obs - observation to pass as input (either Tensor or converts to Tensor)

            Return:
                output - the outpout of our forward pass
        '''
        if not isinstance(obs, Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)

        
        activation1 = F.tanh(self.fc1(obs))
        activation2 = F.tanh(self.fc2(activation1))
        activation3 = F.tanh(self.fc3(activation2))
        output = self.fc4(activation3)

        return output
    
    def log_model(self, writer, base_tag, num_iter):
        '''
            Log the model's distribution weights and biases through histograms for
            each layer on Tensorboard.

            Parameters:
                writer - the initialized Tensorboard SummaryWriter
                base_tag - the base tag for each of the logs
                num_iter - the iteration number that will be set for global step
        '''
        writer.add_histogram(base_tag + "layer1/weight", self.fc1.state_dict()['weight'], num_iter)
        writer.add_histogram(base_tag + "layer1/bias", self.fc1.state_dict()['bias'], num_iter)
        writer.add_histogram(base_tag + "layer2/weight", self.fc2.state_dict()['weight'], num_iter)
        writer.add_histogram(base_tag + "layer2/bias", self.fc2.state_dict()['bias'], num_iter)
        writer.add_histogram(base_tag + "layer3/weight", self.fc3.state_dict()['weight'], num_iter)
        writer.add_histogram(base_tag + "layer3/bias", self.fc3.state_dict()['bias'], num_iter)
        writer.add_histogram(base_tag + "layer4/weight", self.fc4.state_dict()['weight'], num_iter)
        writer.add_histogram(base_tag + "layer4/bias", self.fc4.state_dict()['bias'], num_iter)

