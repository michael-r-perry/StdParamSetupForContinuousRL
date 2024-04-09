import itertools as it
import gymnasium as gym
import multiprocessing as mp
import numpy as np
import os
import shutil
import traceback
import time
import torch
import yaml
from signal import signal, SIGINT
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from Models.agent import ActorCriticAgent
from Models.network import FeedForwardNN
from Utils import RolloutBuffer, explained_variance
from datetime import datetime


class PPO():
    """
        This is the PPO class that we will use in our PPOOutline.py files
    """
    def __init__(
            self, 
            env: gym.Env,
            model_dir: str,
            tensorboard_path: str,
            policy_class: nn.Module = FeedForwardNN,
            **hyperparameters: dict) -> None:
        """
            Initializes the PPO model, including hyperparameters.
            
            Parameters:
                policy_class - the policy class to use for our actor/critic networks.
                env - the environment to train on.
                model_dir - directory where the models will be saved.
                tensorboard_path - path where the models will be saved.
                hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

            Returns:
                None
        """
        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        # Set Environment Property
        self.env = env

        # Initialize PPO Agent
        self.agent = ActorCriticAgent(policy_class, model_dir, env.observation_space, env.action_space, self.hid_dim, self.lr, self.device)
        self.agent.to(self.device)
        self.agent.safe_load()
        #self.agent.init_standardizer(self.env.get_state_min_maxes())

        # Initialize RolloutBuffer
        self.buffer = RolloutBuffer()

        # Initialize Tensorboard Summary Writer
        self.tensorboard_path = tensorboard_path
        self.writer = SummaryWriter(tensorboard_path)
    
    def learn(self, total_timesteps):
        """
            Train the actor and critic networks. Here is where the main PPO algorithm resides.

            Parameters:
                total_timesteps - the total number of timesteps to train for

            Return:
                None
        """
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} tiemsteps per batch for a total of {total_timesteps} timesteps")

        self.num_timestep = 0 # Timesteps simulated so far
        self.num_iteration = 0 # Iterations ran so far

        # Load Run Config Data if Available
        self._load_config()

        try:
            while self.num_timestep < total_timesteps:
                # Collect batch of simulation info
                self.rollout()
                # Train on rollout data
                self.train()
                # Clear Cuda Memory
                torch.cuda.empty_cache()
        except KeyboardInterrupt:
            print("Process Interrupted...")
            raise KeyboardInterrupt
        except Exception as e:
            print("ERROR [learn()]:", e)
            traceback.print_exc()
            raise e
        finally:
            # Save Models
            self.agent.save()
            # Save Config Data
            self._save_config()
            # Clear Tensor CUDAs
            torch.cuda.empty_cache()

    def rollout(self):
        # Clear Buffer and Initialize Data Storage Variables
        self.buffer.clear()

        t = 0 # Keeps track of how many timesteps we've run so far this 
        
        # Keep looping over episodes until the batch has reach the minimum
        # number of timesteps
        while t < self.timesteps_per_batch:
            # Initialize Episode Information
            ep_obs = [] # observations/states collected per episode
            ep_acts = [] # actions collected per episode
            ep_log_probs = [] # log probabilities of actions collected per episode
            ep_entropies = [] # entropies of the action distribution at each step
            ep_rews = [] # rewards collected per episode
            ep_vals = [] # state values collected per episode
            ep_dones = [] # done flag collected episode
            ep_t = 0 # timestep of the episode

            # Reset the environment before next episode
            obs, _ = self.env.reset()
            # Keep looping over timesteps until episode is done
            done = False
            while not done:
                # Track done flag of the current state
                ep_dones.append(done)
                # Track observations for this step
                ep_obs.append(obs)

                # Calculate action and make a step in the env
                with torch.no_grad():
                    act, log_prob, entropy = self.agent.get_action(obs)
                    val = self.agent.get_value(obs)

                # Rescale and perform action
                clipped_act = act.cpu().numpy()

                if isinstance(self.env.action_space, gym.spaces.Box):
                    # Clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_act = np.clip(clipped_act, self.env.action_space.low, self.env.action_space.high)

                obs, rew, terminated, truncated, _ = self.env.step(clipped_act)
                done = terminated or truncated

                # Track recent reward, action, and action log probability
                ep_rews.append(rew)
                ep_vals.append(val.detach())
                ep_acts.append(act.detach())
                ep_log_probs.append(log_prob.detach())
                ep_entropies.append(entropy.detach())

                # Increase timestep
                ep_t += 1

            # Go through the results of the episode and add to batch data
            self.buffer.extend_obs(ep_obs)
            self.buffer.extend_acts(ep_acts)
            self.buffer.extend_log_probs(ep_log_probs)
            self.buffer.append_rews(ep_rews)
            self.buffer.append_lens(ep_t)
            self.buffer.append_vals(ep_vals)
            self.buffer.append_dones(ep_dones)
            t += ep_t

        # Move needed buffer data to tensors before finishing rollout batch
        self.buffer.to_tensor(self.device)

        # Move PyTorch models back to designated device
        self.agent.set_device()
            
        # Update num_timestep and num_ep_timestep
        self.num_timestep += t
        flat_batch_rews = [rew for ep in self.buffer.rews for rew in ep]
        self.writer.add_scalar('batch/mean_ep_len', np.mean(self.buffer.lens), self.num_iteration)
        self.writer.add_scalar('batch/mean_reward', np.mean(flat_batch_rews), self.num_timestep)

    def train(self):
        # Calculate advantage using GAE
        A_k = self.calculate_gae(self.buffer.rews, self.buffer.vals, self.buffer.dones)
        V = self.agent.get_value(self.buffer.obs).squeeze()
        batch_rtgs = A_k + V.detach()

        # Increment the number of iterations
        self.num_iteration += 1

        # One of the only tricks I use that isn't in the psuedocode. Normalizing advantages
        # isn't theoretically necessary, but in practice it decreases teh variance of
        # our advantages and makes convergence much more stable and faster. I added this because
        # solving some environments was too unstable without it
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        # This is the loop where we update our network for some n epochs
        step = self.buffer.obs.size(0)
        idxs = np.arange(step)
        minibatch_size = step // self.num_minibatches
        clip_fractions = []
        entropy_losses = []
        a_losses, c_losses = [], []
        loss = []
        
        continue_training = True
        for _ in range(self.n_updates_per_iteration):
            approx_kl_divs = []

            # Mini-batch Update
            np.random.shuffle(idxs) # Shuffling the index
            for start in range(0, step, minibatch_size):
                end = start + minibatch_size
                idx = idxs[start:end]
                # Extract data at the sampled indices
                mini_obs = self.buffer.obs[idx]
                mini_acts = self.buffer.acts[idx]
                mini_log_probs = self.buffer.log_probs[idx]
                mini_advantage = A_k[idx]
                mini_rtgs = batch_rtgs[idx]

                # Calculate the V_phi and pi_theta(a_t | s_t) and entropy
                V, curr_log_probs, entropy = self.agent.evaluate(mini_obs, mini_acts)

                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                # NOTE: we just subtract the logs, which is the same as
                # dividing the values and then canceling the log with e^log.
                # For why we use log probabilities instead of actual probabilities,
                # here's a great explanation:
                # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                # TL;DR makes gradient descent easier behind the scenes.
                log_ratios = curr_log_probs - mini_log_probs
                ratios = torch.exp(log_ratios)
                approx_kl = ((ratios - 1) - log_ratios).mean()
                approx_kl_divs.append(approx_kl)

                # Logging
                clip_fraction = torch.mean((torch.abs(ratios - 1) > self.clip).float()).item()
                clip_fractions.append(clip_fraction)

                # Calculate surrogate losses.
                surr1 = ratios * mini_advantage
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mini_advantage

                # Calculate actor and critic losses.
                # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                # the performance function, but Adam minimizes the loss. So minimizing the negative
                # performance function maximizes it.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                a_losses.append(actor_loss)
                critic_loss = nn.MSELoss()(V, mini_rtgs)
                c_losses.append(critic_loss)

                # Entropy Regularization
                entropy_loss = entropy.mean()
                # Discount entropy loss by given coefficient
                actor_loss = actor_loss - self.ent_coef * entropy_loss

                # Proceed with backpropagation
                actor_loss = self.agent.backward(actor_loss, critic_loss, self.max_grad_norm)

                entropy_losses.append(entropy_loss)
                loss.append(actor_loss)

                # Approximating KL Divergence
                if approx_kl > self.target_kl:
                    continue_training = False
                
            # Check to continue training
            if not continue_training:
                break

        # Get explained variance
        flatten_vals = np.array([val.cpu().item() for ep in self.buffer.vals for val in ep])
        explained_var = explained_variance(flatten_vals, batch_rtgs.cpu().numpy().flatten())

        print(f"Finished batch #{self.num_iteration}...")

        # Save our model if it's time
        if self.num_iteration % self.save_freq == 0:
            print("Saving model...")
            self.agent.log_models(self.writer, self.num_iteration)
            self.agent.save()

        # Logs
        self.writer.add_scalar('train/approx_kl', np.mean(torch.tensor(approx_kl_divs).numpy()), self.num_timestep)
        self.writer.add_scalar('train/clip_fraction', np.mean(torch.tensor(clip_fractions).numpy()), self.num_timestep)
        self.writer.add_scalar('train/clip_range', self.clip, self.num_timestep)
        self.writer.add_scalar('train/entropy_loss', np.mean(torch.tensor(entropy_losses).numpy()), self.num_timestep)
        self.writer.add_scalar('train/explained_variance', explained_var, self.num_timestep)
        self.writer.add_scalar('train/learning_rate', self.lr, self.num_timestep)
        self.writer.add_scalar('train/actor_loss', np.mean(torch.tensor(a_losses).numpy()), self.num_timestep)
        self.writer.add_scalar('train/critic_loss', np.mean(torch.tensor(c_losses).numpy()), self.num_timestep)
        self.writer.add_scalar('train/loss', loss[-1], self.num_timestep)
    
    def calculate_gae(self, rewards, values, dones):
        batch_advantages = [] # List to store computed advantages for each timestep

        # Iterate over each episode's rewards, values, and done flags:
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = [] # List to store advantages for the current episode
            last_advantage = 0 # Initialize the last computed advantage

            # Calculate episode advantage in reverse order (from last timestep to first)
            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    # Calculate the temporal differnce (TD) error for teh current timestep
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else:
                    # Special case at the boundary (last timestep)
                    delta = ep_rews[t] - ep_vals[t]

                # Calculate Generalized Advantage Estimation (GAE) for the current timestep
                advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage # Update the last advantage for the next timestep
                advantages.insert(0, advantage) # Insert advantage at teh beginning of the list
            
            # Extend the batch_advantages list with advantages computed for the current episode
            batch_advantages.extend(advantages)
        
        # Convert the batch_advantages list to a PyTorch tensor of type float
        return torch.tensor(batch_advantages, dtype=torch.float, device=self.device)
    
    def _init_hyperparameters(self, hyperparameters: dict) -> None:
        """
            Initialize default and custom values for hyperparameters

            Parameters:
                hyperparameters - the extra arguments included when creating the PPO model, should only include
                                  hyperparameters defined below with custom values
        """
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = 384                  # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 128            # Max number of timesteps per episode
        self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
        self.lr = 0.005                                 # Learning rate of actor optimizer
        self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        self.lam = 0.98                                 # Lambda Parameter for GAE 
        self.num_minibatches = 6                        # Number of mini-batches for Mini-batch Update
        self.ent_coef = 0                               # Entropy coefficient for Entropy Regularization
        self.target_kl = 0.02                           # KL Divergence threshold
        self.max_grad_norm = 0.5                        # Gradient Clipping threshold
        self.hid_dim = 64                               # Hidden Dimension size for actor/critic networks

        # Miscellaneous parameters
        self.render = False                             # If we should render during rollout
        self.save_freq = 5                              # How often we save in number of iterations
        self.deterministic = False                      # If we're testing, don't sample actions
        self.seed = None								# Sets the seed of our program, used for reproducibility of results
        self.proc_num = 1                               # Number of parallel processes running during rollout
        self.best_overall_reward = -0.95                # Set the initial best overall reward (helps not make bunch of best files in the beginning)

        # PyTorch parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))
        
        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert(type(self.seed) == int)

            # Set the seed 
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _load_config(self):
        """
            Overwrite class fields if data is stored in config.yml in
            the tensorboard directory.
        """
        # Get path to yaml file
        yml_path = os.path.join(self.tensorboard_path, "config.yml")
        # Return if file doesn't exist
        if not os.path.exists(yml_path):
            return
        # Logging
        print("Run configuration found!")
        print("Loading config data at:", yml_path)
        # Load in data from config
        with open(yml_path, 'r') as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
        # Change any default values to values saved in config
        for param, val in data.items():
            exec('self.' + param + ' = ' + str(val))
    
    def _save_config(self):
        """
            Save Run data to yaml file
        """
        # Get path to yaml file
        yml_path = os.path.join(self.tensorboard_path, "config.yml")
        # Prepare data to save
        data = {
            "num_timestep": self.num_timestep,
            "num_iteration": self.num_iteration,
            "best_overall_reward": self.best_overall_reward,
        }
        # Save data to file
        with open(yml_path, 'w') as f:
            yaml.dump(data,f,sort_keys=False)
        print("Configuration data has successfully written to file:", yml_path)
