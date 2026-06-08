from sim_class import Simulation
import csv

# Initialize the simulation
sim = Simulation(num_agents=1)  # One robot

# Function to move to a position by setting motor velocities and checking when the pipette stops
def move_to_position(sim, velocity_x, velocity_y, velocity_z, max_steps=500):
    actions = [[velocity_x, velocity_y, velocity_z, 0]]  # 0 for no pipette drop
    prev_position = None
    state = None

    for step in range(max_steps):
        state = sim.run(actions)
        current_position = state[f'robotId_{sim.robotIds[0]}']['pipette_position']

        # Check if the pipette has stopped moving
        if prev_position and all(abs(curr - prev) < 1e-4 for curr, prev in zip(current_position, prev_position)):
            break  # Stop simulation if no significant movement

        prev_position = current_position

    return state

# Define sequence of velocities for each step.
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

# Record the working envelope
working_envelope = []
print("Starting simulation.")

# Move through the defined sequence
for i, velocities in enumerate(corner_velocities):
    velocity_x, velocity_y, velocity_z = velocities
    print(f"Moving to step {i + 1}: Velocities = {velocities}")

    # Move to the target
    state = move_to_position(sim, velocity_x, velocity_y, velocity_z)
    pipette_position = state[f'robotId_{sim.robotIds[0]}']['pipette_position']
    working_envelope.append(pipette_position)

# Reset the simulation at the end
sim.reset()
print("Simulation reset. Pipette returned to initial position.")

print("Working Envelope Coordinates:", working_envelope)

# Save coordinates to a CSV file
csv_file = 'working_envelope.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['X', 'Y', 'Z'])  # Header row
    for position in working_envelope:
        writer.writerow(position)

print(f"Working envelope saved to {csv_file}")
