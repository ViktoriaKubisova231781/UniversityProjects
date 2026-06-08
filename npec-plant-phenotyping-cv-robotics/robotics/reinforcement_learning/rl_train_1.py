import os
import wandb
import argparse
from stable_baselines3 import PPO
from ot2_gym_wrapper_2 import OT2Env 
from wandb.integration.sb3 import WandbCallback
import numpy as np

# Disable CUDA (if no GPU support is desired)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["WANDB_API_KEY"] = "561c72b7cd9381070a202a9a5406cc53f3869261"  

# Argument parser for flexibility and reproducibility
parser = argparse.ArgumentParser(description="PPO training on OT2 simulation")
parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
parser.add_argument("--bs", type=int, default=128, help="Batch size")
parser.add_argument("--steps", type=int, default=1024, help="Steps per update")
parser.add_argument("--epochs", type=int, default=8, help="Epochs per update")
parser.add_argument("--discount", type=float, default=0.97, help="Gamma discount factor")
parser.add_argument("--lam", type=float, default=0.9, help="GAE lambda")
parser.add_argument("--arch", type=int, default=64, help="Hidden layer size")
parser.add_argument("--total_steps", type=int, default=1500000, help="Total training timesteps")
parser.add_argument("--eval_interval", type=int, default=15000, help="Evaluation frequency")

args = parser.parse_args()

# ClearML setup (optional – uncomment when needed)
from clearml import Task
clearml_task = Task.init(project_name="Mentor Group S/Group 2", task_name="RL_231781")
clearml_task.set_base_docker("deanis/2023y2b-rl:latest")
clearml_task.connect(args)
clearml_task.execute_remotely(queue_name="default")

# Environment setup
env = OT2Env(render=False, threshold=args.threshold, bonus_reward=args.bonus_reward)

# Wandb session start
run = wandb.init(project="OT2_231781", config=vars(args), sync_tensorboard=True)

# PPO model definition
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=args.lr,
    batch_size=args.bs,
    n_steps=args.steps,
    n_epochs=args.epochs,
    gamma=args.discount,
    gae_lambda=args.lam,
    policy_kwargs=dict(net_arch=[args.arch, args.arch]),
    tensorboard_log=f"runs/{run.id}"
)

# Wandb callback
wandb_cb = WandbCallback(
    model_save_freq=1000,
    model_save_path=f"models/{run.id}",
    verbose=1
)

# Training loop with manual evaluation
eval_freq = args.eval_interval
for i in range(args.total_steps // eval_freq):
    model.learn(
        total_timesteps=eval_freq,
        callback=wandb_cb,
        reset_num_timesteps=False,
        tb_log_name=f"run_{run.id}"
    )
    model_path = f"models/{run.id}/{(i+1)*eval_freq}.zip"
    model.save(model_path)
    wandb.save(model_path)

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

    # Log evaluation stats
    wandb.log({
        "eval/avg_reward": np.mean(rewards),
        "eval/std_reward": np.std(rewards)
    })
    print(f"Evaluation: Avg Reward = {np.mean(rewards):.2f}, Std = {np.std(rewards):.2f}")

# Finish tracking
model.save(f"models/{run.id}/final_model.zip")
wandb.finish()
env.close()
