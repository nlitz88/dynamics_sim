import numpy as np
import plotly.express as px

from box_dynamics import BoxDynamics

# Initialize the box dynamics model
box_dynamics = BoxDynamics()

# Define initial state and control input
x_k = np.array([0, 0, 0, 0])  # [p_x, p_y, v_x, v_y]

# First simulate the dynamics for 50 timesteps where we apply a force in the x
# direction, and then simulate for another 100 timesteps where we apply 0 force.

# Store the state at each timestep in a list.
states = [x_k]

for i in range(0, 50):
    u_k = np.array([1, 0])
    x_k = states[i]
    x_k_1 = box_dynamics.x_k_1(x_k, u_k)
    states.append(x_k_1)

for i in range(49, 150):
    u_k = np.array([0, 0])
    x_k = box_dynamics.x_k_1(x_k, u_k)

# # Create a plotly figure
# fig = px.line(x=states[:, 0], y=states[:, 1], title="Box Trajectory")
# fig.show()

# Create a plotly figure that adds a line for the box's position, velocity, and
# acceleration at each timestep. The x-axis should be the timestep and the
# y-axis should be the corresponding value of the position, velocity, or
# acceleration.

print(len(states))

states = np.array(states)
timesteps = np.arange(0, 150)
fig = px.line(x=timesteps, y=states[:, 0], title="Box Trajectory")
fig.add_scatter(x=timesteps, y=states[:, 2], mode="lines", name="Velocity")
fig.add_scatter(x=timesteps, y=states[:, 3], mode="lines", name="Acceleration")
fig.show()