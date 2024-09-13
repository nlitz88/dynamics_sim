import time
import numpy as np
import plotly.express as px

from box_dynamics import BoxDynamics
from plotting import plot_states

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

# Initialize the box dynamics model
TIMESTEP_LENGTH_S = 0.01
box_dynamics = BoxDynamics(surface_friction_coef=0.5,
                           dt=TIMESTEP_LENGTH_S)

# Define initial state and control input
x_k = np.array([0, 0, 0.0, 0.0])  # [p_x, p_y, v_x, v_y]

# First simulate the dynamics for 50 timesteps where we apply a force in the x
# direction, and then simulate for another 100 timesteps where we apply 0 force.

# Store the state at each timestep in a list.
states = [x_k]

for i in range(1, 100):
    u_k = np.array([10, 0])
    x_k = states[i-1]
    x_k_1 = box_dynamics.x_k_1(x_k, u_k)
    states.append(x_k_1)

for i in range(100, 250):
    u_k = np.array([0, 0])
    x_k = states[i-1]
    x_k = box_dynamics.x_k_1(x_k, u_k)
    states.append(x_k)

# # Create a plotly figure
# fig = px.line(x=states[:, 0], y=states[:, 1], title="Box Trajectory")
# fig.show()

# Create a plotly figure that adds a line for the box's position, velocity, and
# acceleration at each timestep. The x-axis should be the timestep and the
# y-axis should be the corresponding value of the position, velocity, or
# acceleration.

print(len(states))

states = np.array(states)
# timesteps = np.arange(0, 250)
# fig = px.line(x=timesteps, y=states[:, 0], title="px (meters)")
# fig.add_scatter(x=timesteps, y=states[:, 0], mode="lines", name="px (meters)")
# fig.add_scatter(x=timesteps, y=states[:, 1], mode="lines", name="py (meters)")
# fig.add_scatter(x=timesteps, y=states[:, 2], mode="lines", name="vx (m/s)")
# fig.add_scatter(x=timesteps, y=states[:, 3], mode="lines", name="vy (m/s)")
# fig.show()

fig = plot_states(states, ["px (meters)", "py (meters)", "vx (m/s)", "vy (m/s)"])
fig.show()


# Create a meshcat visualizer
vis = meshcat.Visualizer()
vis.open()

# Create a box geometry
vis["box"].set_object(g.Box([0.1, 0.1, 0.1]), g.MeshLambertMaterial(color=0x0000ff))

for i in range(0, 250, 1):
    vis["box"].set_transform(tf.translation_matrix([states[i, 0], states[i, 1], 0]))
    time.sleep(TIMESTEP_LENGTH_S)