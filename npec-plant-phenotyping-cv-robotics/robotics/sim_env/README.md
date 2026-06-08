# OT-2 Digital Twin Simulation Environment

## Overview

This project demonstrates how to control a simulated **Opentrons OT-2** robot using PyBullet. The simulation moves the pipette to predefined positions (representing the eight corners of its working envelope) by sending motor velocity commands. The pipette's final positions are recorded to determine its **working envelope**. This virtual setup allows testing of control algorithms safely before deployment on real hardware.

## Why Simulation?

Robotic manipulation tasks are complex and often involve expensive and fragile equipment. Testing control algorithms directly on physical robots can be costly and pose risks. **Simulation environments** provide a way to develop, refine, and validate robotic control strategies in a safe and controlled environment.  

For this project, we use **PyBullet**, an open-source physics engine widely used for robotic simulations and reinforcement learning. A **digital twin** of the Opentrons OT-2 has been set up in PyBullet, allowing us to interact with the robot virtually before real-world deployment.
---

## Environment Setup

### Dependencies

Before running the simulation, ensure you have the following dependencies installed:

- **Python 3.x** (Recommended: Python 3.8+)
- **PyBullet** (Physics engine for simulation)
- **Git** (For cloning the OT-2 Digital Twin repository)
- **OT-2 Digital Twin Files** (Simulation assets, including `sim_class.py`)

#### Installation Steps:

1. **Install Python Libraries**  
Run the following command in your terminal:
*pip install pybullet*

You can also use conda to install PyBullet. To do so, run the following command:
*conda install -c conda-forge pybullet*

This command will download and install the latest version of PyBullet along with its dependencies. 

2. **Clone the OT-2 Digital Twin Repository**
The OT-2 Digital Twin files are required to run the simulation. Clone the repository:
*git clone https://github.com/BredaUniversityADSAI/Y2B-2023-OT2_Twin.git*

Once the repository is cloned, copy the repository contents (especially sim_class.py) into your project directory.

3. **(Optional) Use a Virtual Environment**
It is recommended to use a virtual environment to manage dependencies:
*python -m venv env*
*source env/bin/activate  # On Windows: env\Scripts\activate*
*pip install pybullet*
---

## Running the Simulation

Once the dependencies are installed and the digital twin files are set up, you can run the simulation:

1. Navigate to the project directory where your simulation script (sim_env.py) is stored:
*cd path/to/your/project*

2. Run the simulation script:
*python sim_env.py*

This script moves the pipette to predefined positions and prints the recorded pipette coordinates.

3. Reset Simulation
The script will reset the pipette's position after completing all movements.
---

## Code Overview

The simulation code performs the following steps:

1. **Initialize the Simulation**
   - Creates an instance of the `Simulation` class with one robot.

2. **Move Pipette to Specific Positions**
   - The `move_to_position` function moves the pipette and checks when it stops moving.

3. **Record Final Pipette Positions**
   - The script records the pipette’s final position at each of the eight corners.

4. **Reset Simulation**
   - Once all movements are completed, the simulation resets.

Below is the main code structure:

```python
from sim_class import Simulation

# Initialize simulation with one robot.
sim = Simulation(num_agents=1)

def move_to_position(sim, velocity_x, velocity_y, velocity_z, max_steps=500):
    """Move the pipette using specified velocities until it stabilizes."""
    actions = [[velocity_x, velocity_y, velocity_z, 0]]  # 0 means no pipette drop
    prev_position = None
    state = None

    for _ in range(max_steps):
        state = sim.run(actions)
        current_position = state[f'robotId_{sim.robotIds[0]}']['pipette_position']
        if prev_position and all(abs(curr - prev) < 1e-4 for curr, prev in zip(current_position, prev_position)):
            break  # Stop when movement is negligible
        prev_position = current_position

    return state

# Define movement sequence
corner_velocities = [
    (-0.5, -0.5, 0.0),  # Bottom-left-front
    (0.5, 0.0, 0.0),    # Bottom-right-front
    (0.0, 0.5, 0.0),    # Bottom-right-back
    (-0.5, 0.0, 0.0),   # Bottom-left-back
    (0.0, 0.0, 0.5),    # Move vertically to top-left-back
    (0.0, -0.5, 0.0),   # Top-left-front
    (0.5, 0.0, 0.0),    # Top-right-front
    (0.0, 0.5, 0.0),    # Top-right-back
]

# Record working envelope
working_envelope = []
print("Starting simulation.")

for i, velocities in enumerate(corner_velocities):
    print(f"Moving to step {i+1}: Velocities = {velocities}")
    state = move_to_position(sim, *velocities)
    pipette_position = state[f'robotId_{sim.robotIds[0]}']['pipette_position']
    working_envelope.append(pipette_position)
    print(f"Step {i+1}: Pipette position: {pipette_position}")

# Reset simulation
sim.reset()
print("Simulation reset. Pipette returned to initial position.")

print("Working Envelope Coordinates:", working_envelope)
```

---

## Working Envelope

The following pipette tip positions were recorded at each step during the simulation:

| Corner | Pipette Position (x, y, z) |
|--------|----------------------------|
| 1      | `[-0.1873, -0.1711, 0.1195]` |
| 2      | `[0.253, -0.1705, 0.1195]`   |
| 3      | `[0.253, 0.2199, 0.1195]`    |
| 4      | `[-0.1874, 0.2195, 0.1195]`  |
| 5      | `[-0.187, 0.2195, 0.2902]`   |
| 6      | `[-0.187, -0.1708, 0.2895]`  |
| 7      | `[0.253, -0.1705, 0.2895]`   |
| 8      | `[0.253, 0.2202, 0.2895]`    |

These values represent the working envelope of the pipette tip in the simulation.
```

```
## PID Controller & Performance Benchmarking

### Objective

To precisely move the pipette tip of the OT-2 robotic arm to desired coordinates using a PID controller. This enables sub-millimeter accurate placement, as required by the client brief for integration with a computer vision-based root detection pipeline.

### Controller Overview

Two types of PID controllers were developed and tested:

- `PIDController`: A simple 1D controller for individual axis testing and baseline performance.
- `PIDController3D`: A fully integrated 3-axis controller for simultaneous control of X, Y, and Z axes, with individually tunable PID gains.

Each controller includes:
- Tunable parameters (`kp`, `ki`, `kd`)
- Integral and derivative tracking
- `reset()` functionality to reinitialize between steps

---

### Control Logic & Decision Rules

Each target coordinate is reached using the following logic:

1. **Error computation**: `target - current`
2. **Control signal generation** via PID
3. **Velocity clamping** based on proximity to target
4. **Real-time execution** in PyBullet simulation
5. **Success condition**:
   - Distance to target < `0.001 m`
   - Stable within threshold for `10 consecutive frames`
   - Pipette drop command triggered

All trajectory points are recorded per axis for performance analysis.

---

### Performance Benchmarking

Performance is measured and visualized using:

- Final **X, Y, Z delta** between target and actual
- **Euclidean distance** to goal
- **Step count** to reach convergence
- Drop locations in pixel space (for overlay on input image)

#### Output Artifacts:
- `droplet_log_13.csv`: Per-root performance log
- `benchmark_x_position.png`: Target vs actual X plot
- `benchmark_y_position.png`: Target vs actual Y plot
- `benchmark_distance.png`: Distance to target per root
- `trajectory_xyz_root_<n>.png`: Full axis tracking per root

---

### Accuracy Achieved

- Final **Euclidean distances**: All ≤ **0.00037 m**
- **100% accuracy** within the required **1 mm** threshold
- **Average steps to converge**: ~34.4
- Controller meets both **Task 12C (≤ 0.01 m)** and **Task 12D (≤ 0.001 m)** requirements

---

### Example Benchmark Output

| Root | Target_X | Actual_X | ΔX     | Distance (m) |
|------|----------|----------|--------|--------------|
| 0    | 0.18343  | 0.18308  | -0.00035 | 0.00035     |
| 1    | 0.17012  | 0.17028  | +0.00016 | 0.00034     |
| 2    | 0.16969  | 0.16969  | +0.00001 | 0.00036     |
| 3    | 0.16828  | 0.16830  | +0.00002 | 0.00037     |
| 4    | 0.17115  | 0.17112  | -0.00002 | 0.00035     |

---

### Summary

The controller is robust, accurate, and integrated with a real-time computer vision pipeline. It satisfies the client's demand for precision and formal control logic, while providing full performance traceability through CSV logs and benchmarking plots.


--- 

## Reinforcement Learning Environment (Gym-Compatible)

To enable reinforcement learning for controlling the OT-2 pipette, I developed **two custom Gymnasium-compatible wrappers** (`ot2_gym_wrapper.py`) for the simulation. These wrappers allow the environment to work seamlessly with RL libraries such as **Stable Baselines3**.

### Frameworks & Setup

The RL environment uses:

- `gymnasium`
- `stable-baselines3`
- `pybullet`
- `wandb`
- `numpy`

Install them with:

```bash
pip install gymnasium stable-baselines3 pybullet wandb numpy
```

---

### OT-2 Gym Wrappers

I created **two wrappers** with different reward structures and hyperparameter flexibility:

#### Wrapper 1 – Basic PID-style reward logic

- **Action Space:**  
  Continuous velocity control → `Box(low=-1, high=1, shape=(3,), dtype=np.float32)`
- **Observation Space:**  
  `[current_x, current_y, current_z, goal_x, goal_y, goal_z]`
- **Reward:**  
  `-Euclidean distance` to goal
- **Done condition:**  
  Success if distance < `0.001 m`

#### Wrapper 2 – Tunable RL wrapper

Includes support for:

- `reward_distance_scale`
- `bonus_reward` on success
- `step_penalty` per step
- Customizable `threshold` for goal accuracy

---

### Testing the Wrapper

The file `test_wrapper.py` runs 5 episodes with random actions to validate the environment:

```bash
python test_wrapper.py
```

Output sample:

```
Episode: 3, Step: 27, Action: [ 0.5 -0.2  0.3], Reward: -0.9823, Success: False
Episode finished after 1000 steps. Info: {'success': False}
```

The environment passes `check_env()` from Stable Baselines3.

---

### Gym API Compliance

Both wrappers implement:

- `reset(seed=None)`
- `step(action)`
- `render()`
- `close()`

And return standard Gym values:  
`(observation, reward, terminated, truncated, info)`

---

### Trained PPO Model

The environment was successfully used to train a reinforcement learning policy using **PPO (Proximal Policy Optimization)** with the following settings:

#### PPO Hyperparameters

| Parameter               | Value         |
|------------------------|---------------|
| Policy                 | `MlpPolicy`   |
| Hidden Units           | `[128, 128]`  |
| Learning Rate          | `3e-4`        |
| Batch Size             | `64`          |
| N Steps                | `2048`        |
| N Epochs               | `12`          |
| Gamma                  | `0.985`       |
| GAE Lambda             | `0.92`        |
| Clip Range             | `0.25`        |
| Total Timesteps        | `250,000`     |
| Evaluation Frequency   | `25,000`      |

#### Environment Parameters

| Parameter             | Value      |
|----------------------|------------|
| `threshold`          | `0.0009` m |
| `reward_distance_scale` | `120`    |
| `step_penalty`       | `-0.75`    |
| `bonus_reward`       | `90`       |

#### Logging

- **WandB** was used to log metrics, training curves, and evaluation rewards.
- Evaluation rewards were averaged across 5 rollouts after each training chunk.

---

### Why It Matters

This RL-compatible wrapper allows the OT-2 environment to be used in:

- RL-based precision control tasks
- Curriculum learning setups
- Hybrid learning (CV + RL integration)
- Performance comparison with PID

---

### Files

| File                  | Purpose                               |
|-----------------------|----------------------------------------|
| `ot2_gym_wrapper.py`  | The main Gym wrapper for OT-2         |
| `test_wrapper.py`     | Basic test script with random actions |
| `ppo_train.py`        | PPO training script with WandB        |
| `sim_class.py`        | Simulation logic                      |
| `README.md`           | Documentation and instructions        |
