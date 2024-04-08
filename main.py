import gymnasium as gym
import os
from Models import PPO

MODEL_DIRPATH = 'Models\\Trained\\ppo_model'
TENSORBOARD_PATH = 'logs\\ppo'

# Get absolute filepaths
cwd = os.getcwd()
model_path = os.path.join(cwd, MODEL_DIRPATH)
tensorboard_path = os.path.join(cwd, TENSORBOARD_PATH)

# Initialize Gymnasium Environment
env = gym.make("Hopper-v4")

# Set Hyperparameters
hyperparameters = {
    "lr": 0.0003,
    "gamma": 0.99,
    "hidden_size": 64,
    "timesteps_per_batch": 1000,
    "max_timesteps_per_episode": 1000,
    "n_updates_per_iteration": 6,
    "num_minibatches": 6,
}

model = PPO(
    env,
    model_path,
    tensorboard_path,
    **hyperparameters
)

model.learn(100_000)