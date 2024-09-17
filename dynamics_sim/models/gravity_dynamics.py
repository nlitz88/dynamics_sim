"""Basic gravity-only dynamics model for a satellite in Low Earth Orbit
(LEO).
"""

import numpy as np
import quatmath as qm

from dynamics_sim.models.dynamics_model import DynamicsModel


class GravityDynamics(DynamicsModel):
    """Implementing dynamics model for a satellite in Low Earth Orbit (LEO).

    The state is defined as:
    x = [r_N, q_N_B, v_N, w_B] where:
        r_N: Position of the satellite in the Earth Centered Inertial (ECI)
             frame. Vector in R3. 
        q_N_B: Quaternion representing the orientation of the satellite body
               relative to the ECI frame. Quaternion in R4.
        v_N: Velocity of the satellite in the ECI frame. Vector in R3.
        w_B: Angular velocity of the satellite body frame relative to the ECI.
             Vector in R3.
    """
    def __init__(self, mu=3.986e14, R=6371e3):
        """Initialize the GravityDynamics class.

        Args:
            mu (float): Gravitational parameter of the Earth in m^3/s^2.
            R (float): Radius of the Earth in meters.
        """
        super().__init__()
        self.mu = mu
        self.R = R

    def x_dot_k_1(self, x_k: np.ndarray, u_k: np.ndarray) -> np.ndarray:
        """Returns xdot at timestep k+1 given the current state x_k and control
        input u_k.

        Args:
            x_k (np.ndarray): The state of the system at timestep k. 
            u_k (np.ndarray): The control value to command at timestep k. Not
                              used for now.

        Returns:
            np.ndarray: The instantaneous rate of change of the state (x_dot) at
            timestep k.
        """
        # Unpack state
        r_N, q_N_B, v_N, w_B = x_k
        # Unpack control input
        m = u_k

        # Compute acceleration (the core of the dynamics model / equations).
        # Compute the acceleration due to gravity
        a_gravity = -self.mu / (np.linalg.norm(r_N)**3) * r_N

        # Compute the total acceleration
        a_N = a_gravity

        # Compute the rate of change of the state
        r_dot = v_N
        q_dot = 0.5 * qm.Q(q_N_B) @ w_B
        v_dot = a_N
        w_dot = 0.0 # Not sure how to compute this next--how do we compute the inertia matrix? 
        # https://ocw.mit.edu/courses/16-07-dynamics-fall-2009/dd277ec654440f4c2b5b07d6c286c3fd_MIT16_07F09_Lec26.pdf
        # maybe this will help?

        return np.concatenate([r_dot, q_dot, v_dot, w_dot])