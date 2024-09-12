"""Module containing a dynamics base class"""


# NOTE: An integrator is included in the Dynamics to give subclasses the
# flexibility to implement x_k_1 (the discrete-time dynamics) as they please.
# This could be applicable for learned discrete-time dynamics models, for
# example--or if the user wishes for a different integrator to be used for the
# simulation.

import numpy as np

class Dynamics:

    def __init__(self, dt=0.01):
        """_summary_

        Args:
            dt (float, optional): Simulation timestep length. Defaults to 0.01.
        """
        self.dt = dt
    
    def x_dot_k_1(self, x_k: np.ndarray, u_k: np.ndarray) -> np.ndarray:
        """Returns xdot at timestep k+1 given the current state x_k and control
        input u_k. This is to be implemented by each specific dynamics model
        subclass.

        Args:
            x_k (np.ndarray): The state of the system at timestep k. 
            u_k (np.ndarray): The control value to command at timestep k.

        Returns:
            np.ndarray: The instantaneous rate of change of the state (x_dot) at
           timestep k.
        """
        raise NotImplementedError
    
    def x_k_1(self, x_k: np.ndarray, u_k: np.ndarray) -> np.ndarray:
        """Compute RK4 approximation of x_k+1 using the dynamics function
        x_dot_k_1, given the current state x_k, the control input u_k, and the
        timestep length dt.

        Args:
            x_k (np.ndarray): 
            u_k (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        k1 = self.x_dot_k_1(x_k, u_k)
        k2 = self.x_dot_k_1(x_k + 0.5 * self.dt * k1, u_k)
        k3 = self.x_dot_k_1(x_k + 0.5 * self.dt * k2, u_k)
        k4 = self.x_dot_k_1(x_k + self.dt * k3, u_k)
        return x_k + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)