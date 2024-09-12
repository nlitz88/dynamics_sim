"""Simple 2D box being pushed by a force control input dynamics model example."""

import numpy as np

from dynamics import Dynamics

class BoxDynamics(Dynamics):
    """Implementing dynamics model for a box being pushed on a table with
    surface friction.
    """

    def __init__(self, dt=0.01,
                 box_mass=1.0,
                 surface_friction_coef=0.1,
                 gravity=9.81):
        """Initialize the box dynamics model with the box_mass and surface
        _surface_friction_coef parameters.

        Args:
            dt (float, optional): Simulation timestep length. Defaults to 0.01.
            box_mass (float, optional): Mass of the box. Defaults to 1.0.
            surface_friction_coef (float, optional): Friction coefficient.
            Defaults to 0.1.
            gravity (float, optional): Acceleration due to gravity. Defaults to
            9.81.
        """
        super().__init__(dt=dt)
        self._box_mass = box_mass
        self._surface_friction_coef = surface_friction_coef
        self._gravity = gravity

    def x_dot_k_1(self, x_k: np.ndarray, u_k: np.ndarray) -> np.ndarray:
        """Returns xdot at timestep k+1 given the current state x_k and control
        input u_k.

        Args:
            x_k (np.ndarray): The state of the system at timestep k. 
            u_k (np.ndarray): The control value to command at timestep k.

        Returns:
            np.ndarray: The instantaneous rate of change of the state (x_dot) at
            timestep k.
        """
        # Unpack state
        p_x, p_y, v_x, v_y = x_k
        # Unpack control input
        f_x, f_y = u_k

        # Compute acceleration (the core of the dynamics model / equations).
        a_x = f_x - self._surface_friction_coef*(self._box_mass*self._gravity + -1*f_y)
        a_y = 0

        return np.array([v_x, v_y, a_x, a_y])
