import os
import wandb
import argparse
import numpy as np
from stable_baselines3 import PPO
from ot2_gym_wrapper_2 import OT2Env
from wandb.integration.sb3 import WandbCallback
# from clearml import Task  # Commented out for local use

# Disable CUDA (if no GPU support is desired)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["WANDB_API_KEY"] = "561c72b7cd9381070a202a9a5406cc53f3869261"

# Argument parser for flexibility and reproducibility
parser = argparse.ArgumentParser(description="PPO training on OT2 simulation")
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=12)
parser.add_argument("--gamma", type=float, default=0.985)
parser.add_argument("--gae_lambda", type=float, default=0.92)
parser.add_argument("--clip_range", type=float, default=0.25)
parser.add_argument("--policy", type=str, default="MlpPolicy")
parser.add_argument("--hidden_units", type=int, default=128)
parser.add_argument("--threshold", type=float, default=0.0009)
parser.add_argument("--reward_distance_scale", type=int, default=120)
parser.add_argument("--step_penalty", type=float, default=-0.75)
parser.add_argument("--bonus_reward", type=int, default=90)
parser.add_argument("--total_timesteps", type=int, default=250000)
parser.add_argument("--eval_freq", type=int, default=25000)

args = parser.parse_args()

# ClearML tracking (disabled for local)
# clearml_task = Task.init(...)
# clearml_task.execute_remotely(...)

# Environment setup
env = OT2Env(
    render=False,
    threshold=args.threshold,
    reward_distance_scale=args.reward_distance_scale,
    step_penalty=args.step_penalty,
    bonus_reward=args.bonus_reward
)

# Start wandb run
run = wandb.init(project="OT2_231781", config=vars(args), sync_tensorboard=True)

# PPO model setup
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    n_epochs=args.n_epochs,
    gamma=args.gamma,
    gae_lambda=args.gae_lambda,
    clip_range=args.clip_range,
    policy_kwargs=dict(net_arch=[args.hidden_units, args.hidden_units]),
    tensorboard_log=f"runs/{run.id}"
)

# Wandb callback (no saving models to avoid symlink error)
wandb_cb = WandbCallback(
    model_save_freq=0,  # disables automatic saving
    verbose=1
)

# Training loop with evaluation logging
for i in range(args.total_timesteps // args.eval_freq):
    model.learn(
        total_timesteps=args.eval_freq,
        callback=wandb_cb,
        reset_num_timesteps=False,
        tb_log_name=f"run_{run.id}"
    )

    model_path = f"models/{run.id}/{(i + 1) * args.eval_freq}.zip"
    model.save(model_path)

    # Manual evaluation
    rewards = []
    for _ in range(5):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)

    wandb.log({
        "eval/avg_reward": np.mean(rewards),
        "eval/std_reward": np.std(rewards)
    })
    print(f"Evaluation: Avg Reward = {np.mean(rewards):.2f}, Std = {np.std(rewards):.2f}")

# Final save and cleanup
model.save(f"models/{run.id}/final_model.zip")
wandb.finish()
env.close()
