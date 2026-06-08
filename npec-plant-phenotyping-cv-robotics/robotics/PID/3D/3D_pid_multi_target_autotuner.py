# controller_test.py - Autotuner (Multi-target, Averaged Evaluation)
import numpy as np
import csv
import random
from itertools import product
from sim_class import Simulation
from pid_controller_3D import PIDController3D  # ✅ Updated to import the correct class

def run_pid_trial_3d(kp, ki, kd, target, dt=0.5, max_steps=1000, threshold=0.001, hold_steps=50):
    """
    Runs a single PID trial to a target point.
    Returns whether it succeeded and how many steps it took to hold.
    """
    # Start simulation
    sim = Simulation(num_agents=1, render=False)
    obs = sim.reset(num_agents=1)
    agent_id = int(list(obs.keys())[-1][-1])
    current_pos = np.array(sim.get_pipette_position(agent_id))

    # Initialize 3D PID controller
    pid = PIDController3D(kp, ki, kd)
    pid.reset()

    hold_counter = 0
    success = False

    # PID control loop
    for step in range(max_steps):
        error = target - current_pos
        control = pid.compute(error, dt)
        control = np.clip(control, -1, 1)

        # Apply control and update observation
        action = np.concatenate([control, [0]])
        obs = sim.run([action])
        current_pos = np.array(sim.get_pipette_position(agent_id))
        dist = np.linalg.norm(current_pos - target)

        # Success criteria
        if dist < threshold:
            hold_counter += 1
            if hold_counter >= hold_steps:
                success = True
                break
        else:
            hold_counter = 0

    sim.close()
    return success, step if success else max_steps


def autotune_pid_multi_target(trials_per_config=5, dt=0.5, log_filename="3D_pid_multi_autotune_log.csv"):
    """
    Tests multiple PID gain combinations across random targets.
    Logs average success rate and steps to hold.
    """
    # Define gain search space
    kp_values = [10, 12, 15, 18, 20]
    ki_values = [0.0, 0.02, 0.05]
    kd_values = [0.5, 0.7, 1.0, 1.2]

    # Define workspace bounds for target generation
    bounds = {
        "x": (-0.1874, 0.253),
        "y": (-0.1711, 0.2202),
        "z": (0.1195, 0.2902),
    }

    # Open CSV file to log results
    with open(log_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kp", "ki", "kd", "SuccessRate", "AvgStepsToHold"])

        # Loop over all combinations of gains
        for kp, ki, kd in product(kp_values, ki_values, kd_values):
            print(f"Testing kp={kp}, ki={ki}, kd={kd}")
            successes = 0
            total_steps = 0

            # Run multiple randomized trials per config
            for _ in range(trials_per_config):
                target = np.array([
                    random.uniform(*bounds["x"]),
                    random.uniform(*bounds["y"]),
                    random.uniform(*bounds["z"])
                ])

                # Use the same gain for all three axes
                success, steps = run_pid_trial_3d([kp]*3, [ki]*3, [kd]*3, target, dt)
                if success:
                    successes += 1
                total_steps += steps

            # Aggregate results for current gain combo
            success_rate = successes / trials_per_config
            avg_steps = total_steps / trials_per_config
            writer.writerow([kp, ki, kd, success_rate, avg_steps])
            print(f"Result: {success_rate:.2f} success rate, {avg_steps:.1f} avg steps")

    print(f"\nAutotune finished. Results saved to {log_filename}")


if __name__ == "__main__":
    # Run the multi-target autotuner
    autotune_pid_multi_target()
