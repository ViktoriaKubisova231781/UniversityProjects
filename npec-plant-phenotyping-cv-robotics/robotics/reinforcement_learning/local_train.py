import argparse
import numpy as np
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
import wandb
from ot2_gym_wrapper import OT2Env

# ----------------------------
# Parse CLI arguments
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=3e-4)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--timesteps', type=int, default=10000) 
parser.add_argument('--model_name', type=str, default="test_model")
args = parser.parse_args()

# ----------------------------
# Init Weights & Biases
# ----------------------------
wandb.init(project="ot2-rl", config=args.__dict__, name=args.model_name)

# ----------------------------
# Create Environment
# ----------------------------
env = OT2Env(render=False)

# ----------------------------
# PPO Model
# ----------------------------
model = PPO("MlpPolicy", env,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            batch_size=args.batch_size,
            verbose=1,
            tensorboard_log="./ppo_logs")

# ----------------------------
# Train Model
# ----------------------------
model.learn(total_timesteps=args.timesteps, callback=WandbCallback())

# ----------------------------
# Save Model
# ----------------------------
model.save(args.model_name)
print(f"Model saved as '{args.model_name}.zip'")
