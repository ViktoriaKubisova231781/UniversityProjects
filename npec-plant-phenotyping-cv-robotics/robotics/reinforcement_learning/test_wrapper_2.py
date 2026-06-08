import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_checker import check_env
from ot2_gym_wrapper import OT2Env

# Instantiate the environment with custom threshold and bonus reward
env = OT2Env(threshold=0.001, bonus_reward=100)

# Check if the environment follows the Gymnasium API
check_env(env)

# Run a simple test loop
num_episodes = 5
successes = 0

for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    step = 0

    while not done:
        # Take a random action from the action space
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Episode: {episode + 1}, Step: {step + 1}, Action: {action}, Reward: {reward:.4f}, Success: {info.get('success', False)}")

        step += 1
        if terminated or truncated:
            if info.get("success", False):
                successes += 1
            print(f"Episode finished after {step} steps. Info: {info}")
            break

print(f"\nSuccess rate: {successes}/{num_episodes} episodes")

env.close()
