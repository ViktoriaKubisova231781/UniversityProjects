import numpy as np
import random
import csv
from pid_controller_3D import PIDController3D
from sim_class import Simulation

def run_pid_to_target(target, kp, ki, kd, dt=1.0, threshold=0.001, hold_time=50, max_steps=1000, render=False):
    sim = Simulation(num_agents=1, render=render)
    pid = PIDController3D(kp, ki, kd)

    obs = sim.reset(num_agents=1)
    agent_id = int(list(obs.keys())[-1][-1])
    current = np.array(sim.get_pipette_position(agent_id))

    pid.reset()
    steps_within_threshold = 0
    target_reached_step = None

    for step in range(max_steps):
        error = target - current
        control = pid.compute(error, dt)
        control = np.clip(control, -1, 1)

        action = np.concatenate([control, [0]])  # [dx, dy, dz, gripper]
        obs = sim.run([action])
        current = np.array(sim.get_pipette_position(agent_id))
        distance = np.linalg.norm(current - target)

        print(f"Step {step}: Pos={current}, Distance={distance:.6f}")

        if distance < threshold:
            if steps_within_threshold == 0:
                target_reached_step = step
            steps_within_threshold += 1
            if steps_within_threshold >= hold_time:
                sim.close()
                return True, target_reached_step
        else:
            steps_within_threshold = 0
            target_reached_step = None

    sim.close()
    return False, max_steps


if __name__ == "__main__":
    kp = [18, 18, 18]
    ki = [0.0, 0.0, 0.0]
    kd = [0, 0, 0]
    dt = 1.0
    hold_time = 100
    trials = 20
    log_file = "pid_3d_multi_target_results_4.csv"

    # Position bounds for random targets
    bounds = {
        "x": (-0.1874, 0.253),
        "y": (-0.1711, 0.2202),
        "z": (0.1195, 0.2902),  # kept for reference but not used now
    }
    fixed_z = 0.20
    successes = 0

    with open(log_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Trial", "Target_X", "Target_Y", "Target_Z",
            "Success", "StepsToHold",
            "kp_x", "kp_y", "kp_z", "ki_x", "ki_y", "ki_z", "kd_x", "kd_y", "kd_z"
        ])

        for i in range(trials):
            # Generate random target with fixed Z
            target = [
                random.uniform(*bounds["x"]),
                random.uniform(*bounds["y"]),
                fixed_z]

            print(f"\nTrial {i+1} — Target: {target}")

            success, steps = run_pid_to_target(target, kp, ki, kd, dt, hold_time=hold_time, render=True)

            writer.writerow([
                i + 1, *target, int(success), steps,
                *kp, *ki, *kd
            ])
            if success:
                successes += 1

    print(f"\n{successes}/{trials} trials succeeded.")
    print(f"Results saved to {log_file}")
