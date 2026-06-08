import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_checker import check_env
from ot2_gym_wrapper import OT2Env

# Instantiate the environment
env = OT2Env()

# Check if the environment follows the Gymnasium API
check_env(env)

# Run a simple test loop
num_episodes = 5

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    step = 0

    while not done:
        # Take a random action from the action space
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Episode: {episode + 1}, Step: {step + 1}, Action: {action}, Reward: {reward}")

        step += 1
        if terminated or truncated:
            print(f"Episode finished after {step} steps. Info: {info}")
            break

env.close()
