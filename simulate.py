import time
import numpy as np

from models.box_dynamics import BoxDynamics
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

# Create graphs for each component of the state vector.
fig = plot_states(states, ["px (meters)", "py (meters)", "vx (m/s)", "vy (m/s)"])
fig.show()

# Create a meshcat visualizer
vis = meshcat.Visualizer()
vis.open()
vis.wait()
print(f"Open the visualizer at the following URL: {vis.url()}")

# Create a box geometry
vis["box"].set_object(g.Box([0.1, 0.1, 0.1]), g.MeshLambertMaterial(color=0x0000ff))

for i in range(0, 250, 1):
    vis["box"].set_transform(tf.translation_matrix([states[i, 0], states[i, 1], 0]))
    time.sleep(TIMESTEP_LENGTH_S)

# TODO: Come up with a better way of simulating forward given an arbitrary
# Dynamics object. Above code is a bit hacky.

# TODO: Think about how to make the simulation method able to accomodate any
# arbitrary controller function. To do this, just think about what we are
# requiring this to do.

# I.e., the primary goal of the simulator is to see how the system state evolves
# over time given a "ground truth" dynamics model and a controller. At each
# timestep, the simulator should pass the current state into the controller
# function and have it figure out what the control action should be. Then, we
# take the current state and the control action and simulate the dynamics model
# one timestep forward. We repeat this process for a specified number of
# timesteps or until a stopping condition is met.

# Maybe this could be a class that you could initialize with a particular
# dynamics model and controller function, and maybe the parent class has built
# in methods for meshcat visualization and plotting. It could have a function
# called simulate that takes in the number of timesteps to simulate, timestep
# length, and initial state (initial condition).

# Maybe if the user has special simulation needs, they can inherit from the
# parent class and implement their own simulate method. OR, similar to how the
# user wll inherit from the Dynamics or Controller class, they can also inherit
# from the Simulation class for each unique simulation scenario. 3D meshcat
# scenes and geometry could be associated with that particular simulation
# scenario as well.

# OR, maybe we just end up writing a new simulation script for each scenario
# we want, and then we can just use the plotting functions to visualize the data
# and so forth. That might just be the fastest path to getting something working
# for now without strangling us with too much complexity.