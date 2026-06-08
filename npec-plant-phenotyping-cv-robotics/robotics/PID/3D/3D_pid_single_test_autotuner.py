import numpy as np
import csv
from itertools import product
from sim_class import Simulation
from pid_controller_3D import PIDController3D  # Make sure this class exists in the file

def run_pid_trial_3d(kp, ki, kd, target, dt=1.0, max_steps=1000, threshold=0.001, hold_steps=50):
    """
    Run a single PID control trial with given gains to a specified target.
    """
    # Initialize the simulation with one agent
    sim = Simulation(num_agents=1, render=False)
    obs = sim.reset(num_agents=1)
    agent_id = int(list(obs.keys())[-1][-1])
    current_pos = np.array(sim.get_pipette_position(agent_id))

    # Initialize the 3D PID controller with current gains
    pid = PIDController3D(kp, ki, kd)
    pid.reset()

    # Logging containers
    step_data = []
    hold_counter = 0
    success = False

    # Control loop
    for step in range(max_steps):
        error = target - current_pos
        control = pid.compute(error, dt)
        control = np.clip(control, -1, 1)

        # Apply control to simulation
        action = np.concatenate([control, [0]])  # 4th value = placeholder/gripper
        obs = sim.run([action])
        current_pos = np.array(sim.get_pipette_position(agent_id))
        dist = np.linalg.norm(current_pos - target)

        # Log position and distance
        step_data.append([step] + list(current_pos) + [dist])

        # Check if target is held within threshold for enough steps
        if dist < threshold:
            hold_counter += 1
            if hold_counter >= hold_steps:
                success = True
                break
        else:
            hold_counter = 0

    sim.close()
    return step_data, success, step


def autotune_pid_to_target(target, dt=1.0, log_filename="3D_pid_autotune_log.csv"):
    """
    Runs a grid search over PID gain combinations and logs the performance.
    """
    # Define gain ranges to test (same for x, y, z)
    kp_values = [10, 12, 15, 18]
    ki_values = [0.0, 0.05, 0.1]
    kd_values = [0.5, 0.7, 1.0]

    # Open CSV file to log the results
    with open(log_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kp", "ki", "kd", "Success", "StepsToHold"])

        # Iterate through all combinations of gains
        for kp, ki, kd in product(kp_values, ki_values, kd_values):
            print(f"Testing kp={kp}, ki={ki}, kd={kd}")

            # Apply same gain for all 3 axes
            gains = [ [kp]*3, [ki]*3, [kd]*3 ]
            step_data, success, steps = run_pid_trial_3d(*gains, target, dt)

            # Write the result of the test
            writer.writerow([kp, ki, kd, int(success), steps])
            print(f"Result: {'Success' if success else 'Failure'}, Steps to hold: {steps}")

    print(f"\nAutotune complete. Results saved to {log_filename}")


if __name__ == "__main__":
    # Define the target position for the test
    target = np.array([0.1, 0.1, 0.25])

    # Start the autotuning process
    print(f"Starting autotuning for target: {target}")
    autotune_pid_to_target(target)
