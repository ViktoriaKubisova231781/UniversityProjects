import numpy as np
import csv
import time
import matplotlib.pyplot as plt
from pid_controller import PIDController
from sim_class import Simulation 

def run_trial_pid(target, dt=1.0, max_steps=1000, threshold=0.001, hold_steps=50):
    # Initialize simulation with rendering on
    sim = Simulation(num_agents=1, render=True)
    obs = sim.reset(num_agents=1)
    agent_id = int(list(obs.keys())[-1][-1])  # Extract agent ID
    current = np.array(sim.get_pipette_position(agent_id))  # Initial position

    # Create one PID per axis
    pid_x = PIDController(10, 0.2, 0.5)
    pid_y = PIDController(10, 0.2, 0.5)
    pid_z = PIDController(10, 0.2, 0.5)


    hold_counter = 0
    log_data = []  # To store step logs: [step, x, y, z, dist]
    error_log = []  # To store errors: [err_x, err_y, err_z]
    velocity_log = []  # To store velocities: [vx, vy, vz]

    for step in range(max_steps):
        error = target - current  # Error for each axis

        # Compute control velocities using PID
        vx = pid_x.compute(error[0], dt)
        vy = pid_y.compute(error[1], dt)
        vz = pid_z.compute(error[2], dt)

        control = np.clip([vx, vy, vz], -1, 1)  # Clamp control signals
        action = np.concatenate([control, [0]])  # Add dummy gripper control

        obs = sim.run([action])  # Apply action in simulation
        current = np.array(sim.get_pipette_position(agent_id))  # Get new position
        distance = np.linalg.norm(current - target)  # Compute distance to target

        # Log data
        print(f"Step {step}: Pos={current}, Dist={distance:.5f}")
        log_data.append([step, current[0], current[1], current[2], distance])
        error_log.append(error.tolist())
        velocity_log.append([vx, vy, vz])

        # Check if we're holding the target steadily
        if distance < threshold:
            hold_counter += 1
            if hold_counter >= hold_steps:
                print(f"Target held for {hold_steps} steps at step {step}.")
                break
        else:
            hold_counter = 0

    else:
        print("Failed to reach and hold the target.")

    sim.close()

    # Save log to CSV file
    with open("simple_pid_results_2.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Step", "X", "Y", "Z", "DistanceToTarget"])
        writer.writerows(log_data)

    # Plotting graphs
    log_array = np.array(log_data)
    error_array = np.array(error_log)
    velocity_array = np.array(velocity_log)

    # Position over time
    plt.figure()
    plt.plot(log_array[:, 0], log_array[:, 1], label="X")
    plt.plot(log_array[:, 0], log_array[:, 2], label="Y")
    plt.plot(log_array[:, 0], log_array[:, 3], label="Z")
    plt.title("Position Over Time")
    plt.xlabel("Step")
    plt.ylabel("Position")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("simple_pid_position_plot_2.png")
    plt.show()

    # Distance to target over time
    plt.figure()
    plt.plot(log_array[:, 0], log_array[:, 4], color='red', label="Distance to Target")
    plt.title("Distance to Target Over Time")
    plt.xlabel("Step")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("simple_pid_distance_plot_2.png")
    plt.show()

    # Error over time (X, Y, Z)
    plt.figure()
    plt.plot(log_array[:, 0], error_array[:, 0], label="Error X")
    plt.plot(log_array[:, 0], error_array[:, 1], label="Error Y")
    plt.plot(log_array[:, 0], error_array[:, 2], label="Error Z")
    plt.title("PID Error Over Time")
    plt.xlabel("Step")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("simple_pid_error_plot_2.png")
    plt.show()

    # Control (velocity) over time (X, Y, Z)
    plt.figure()
    plt.plot(log_array[:, 0], velocity_array[:, 0], label="Vx")
    plt.plot(log_array[:, 0], velocity_array[:, 1], label="Vy")
    plt.plot(log_array[:, 0], velocity_array[:, 2], label="Vz")
    plt.title("Control Velocities Over Time")
    plt.xlabel("Step")
    plt.ylabel("Velocity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("simple_pid_velocity_plot_2.png")
    plt.show()

    return True


if __name__ == "__main__":
    # Define target position manually
    target = np.array([0.1, 0.1, 0.25])  
    print(f"Running basic trial PID controller to target {target}\n")
    run_trial_pid(target)
