import gymnasium as gym
import os
from Models import PPO

MODEL_DIRPATH = 'Models\\Trained\\ppo_model'
TENSORBOARD_PATH = 'logs\\ppo'

env = gym.make("Hopper-v4")

