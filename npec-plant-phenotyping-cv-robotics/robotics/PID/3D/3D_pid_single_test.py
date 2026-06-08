import numpy as np
import csv
import datetime
import matplotlib.pyplot as plt
from sim_class import Simulation
from pid_controller_3D import PIDController3D  

def run_pid_trial_3d(kp, ki, kd, target, dt=1.0, max_steps=1000, threshold=0.001, hold_steps=50):
    """
    Runs a PID control trial to a single target and logs the trajectory and metadata.

    Args:
        kp, ki, kd (list): PID gains [x, y, z]
        target (np.array): Target position [x, y, z]
        dt (float): Time step
        max_steps (int): Max number of simulation steps
        threshold (float): Error threshold for success
        hold_steps (int): Steps the target must be held

    Returns:
        list: Step log data [step, x, y, z, distance]
    """
    # Initialize the simulation with one agent
    sim = Simulation(num_agents=1, render=True)
    obs = sim.reset(num_agents=1)
    agent_id = int(list(obs.keys())[-1][-1])
    current_pos = np.array(sim.get_pipette_position(agent_id))

    # Initialize 3D PID controller
    pid = PIDController3D(kp, ki, kd)
    pid.reset()

    # Prepare logs for data tracking
    step_data = []
    error_log = []
    velocity_log = []
    hold_counter = 0

    # Control loop
    for step in range(max_steps):
        # Compute error and control
        error = target - current_pos
        control = pid.compute(error, dt)
        control = np.clip(control, -1, 1)

        # Send control command to simulation
        action = np.concatenate([control, [0]])  # 4th element is a placeholder (e.g., gripper)
        obs = sim.run([action])
        current_pos = np.array(sim.get_pipette_position(agent_id))
        dist = np.linalg.norm(current_pos - target)

        # Logging step info
        print(f"Step {step}: Pos={current_pos}, Dist={dist:.5f}")
        step_data.append([step] + list(current_pos) + [dist])
        error_log.append(error.tolist())
        velocity_log.append(control.tolist())

        # Check if target is held within threshold
        if dist < threshold:
            hold_counter += 1
            if hold_counter >= hold_steps:
                print(f"Target held for {hold_steps} steps at step {step}.")
                break
        else:
            hold_counter = 0

    else:
        print("Failed to reach and hold the target within threshold.")

    sim.close()

    # Save results to CSV
    filename = "pid_trial_3d_results_3.csv"
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)

        # Save metadata
        writer.writerow(["# PID Trial Metadata"])
        writer.writerow(["Target", *target])
        writer.writerow(["Kp", *kp])
        writer.writerow(["Ki", *ki])
        writer.writerow(["Kd", *kd])
        writer.writerow(["dt", dt])
        writer.writerow([])

        # Save per-step data
        writer.writerow(["Step", "X", "Y", "Z", "DistanceToTarget"])
        writer.writerows(step_data)

    print(f"\nResults saved to {filename}")

    # Plotting results
    step_data_np = np.array(step_data)
    steps = step_data_np[:, 0]
    positions = step_data_np[:, 1:4]
    distances = step_data_np[:, 4]
    errors = np.array(error_log)
    velocities = np.array(velocity_log)

    # Position plot
    plt.figure()
    plt.plot(steps, positions[:, 0], label="X")
    plt.plot(steps, positions[:, 1], label="Y")
    plt.plot(steps, positions[:, 2], label="Z")
    plt.title("Position Over Time")
    plt.xlabel("Step")
    plt.ylabel("Position")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("3d_pid_position_plot_3.png")
    plt.show()

    # Distance to target plot
    plt.figure()
    plt.plot(steps, distances, color='red', label="Distance to Target")
    plt.title("Distance to Target Over Time")
    plt.xlabel("Step")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("3d__pid_distance_plot_3.png")
    plt.show()

    # PID error plot
    plt.figure()
    plt.plot(steps, errors[:, 0], label="Error X")
    plt.plot(steps, errors[:, 1], label="Error Y")
    plt.plot(steps, errors[:, 2], label="Error Z")
    plt.title("PID Error Over Time")
    plt.xlabel("Step")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("3d_pid_error_plot_3.png")
    plt.show()

    # Velocity (control output) plot
    plt.figure()
    plt.plot(steps, velocities[:, 0], label="Vx")
    plt.plot(steps, velocities[:, 1], label="Vy")
    plt.plot(steps, velocities[:, 2], label="Vz")
    plt.title("Control Velocities Over Time")
    plt.xlabel("Step")
    plt.ylabel("Velocity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("3d_pid_velocity_plot_3.png")
    plt.show()

    return step_data


if __name__ == "__main__":
    # PID gains for [X, Y, Z]
    kp = [15, 15, 15]
    ki = [0, 0, 0]
    kd = [0.7, 0.7, 0.7]

    # Target position in 3D space
    target = np.array([0.1, 0.1, 0.25])

    print(f"\nRunning PID trial to target: {target}")
    run_pid_trial_3d(kp, ki, kd, target)
