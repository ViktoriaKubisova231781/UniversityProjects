import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1)

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-0.1874, -0.1711, 0.1195, -0.1874, -0.1711, 0.1195]),
            high=np.array([0.253, 0.2202, 0.2902, 0.253, 0.2202, 0.2902]),
            shape=(6,),
            dtype=np.float32
        )

        # Keep track of the number of steps
        self.steps = 0
    
    def reset(self, seed=None):
        # being able to set a seed is required for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Reset the state of the environment to an initial state
        # Set a random goal position for the agent, consisting of x, y, and z coordinates within the working area
        self.goal_position = np.random.uniform(
            low=[-0.1874, -0.1711, 0.1195], high=[0.253, 0.2202, 0.2902]
        )

        # Call the environment reset function
        observation = self.sim.reset(num_agents=1)

        # Get the correct robot ID dynamically
        robot_key = list(observation.keys())[0]  # Selects the first available robot ID
        pipette_position = np.array(observation[robot_key]['pipette_position'], dtype=np.float32)

        observation = np.concatenate((pipette_position, self.goal_position)).astype(np.float32)

        self.steps = 0

        return observation, {}  #  Now returning a tuple (observation, info)
        
    def step(self, action):
        # Execute one time step within the environment
        scaled_action = np.append(action * 0.5, 0)

        # Call the environment step function
        observation = self.sim.run([scaled_action])

        # Get the correct robot ID dynamically
        robot_key = list(observation.keys())[0]  # Selects the first available robot ID
        pipette_position = np.array(observation[robot_key]['pipette_position'], dtype=np.float32)

        # Ensure pipette stays within the working envelope
        pipette_position = np.clip(pipette_position, self.observation_space.low[:3], self.observation_space.high[:3])
        observation = np.concatenate((pipette_position, self.goal_position)).astype(np.float32)

        # Calculate the reward
        reward = -np.linalg.norm(pipette_position - self.goal_position)

        # Ensure `terminated` is explicitly a boolean
        terminated = bool(np.linalg.norm(pipette_position - self.goal_position) < 0.001)

        # Ensure `truncated` is explicitly a boolean
        truncated = bool(self.steps >= self.max_steps)

        info = {"success": terminated}

        self.steps += 1

        return observation, reward, terminated, truncated, info  

    def render(self, mode='human'):
        pass
    
    def close(self):
        self.sim.close()
