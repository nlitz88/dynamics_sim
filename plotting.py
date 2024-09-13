"""Utility functions for plotting simulation results"""

from typing import List, Optional
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Write a function that creates a grid of plotly line plots, where each plot is
# one of the states in the states array. The x-axis should be the timestep and
# the y-axis should be the corresponding value of the state.
def plot_states(states: np.ndarray,
                state_names: List[str],
                num_columns: Optional[int] = 2) -> go.Figure:
    """Create a grid of plotly line plots, where each plot is one of the states
    in the states array. The x-axis should be the timestep and the y-axis should
    be the corresponding value of the state.

    Args:
        states (np.ndarray): A 2D numpy array where each row is a timestep and
        each column is a state value.
        state_names (List[str]): A list of strings where each string is the name
        of the corresponding state.
        num_columns (Optional[int], optional): Number of columns in the grid.
        Defaults to 2.

    Returns:
        go.Figure: A plotly figure object.
    """
    timesteps = np.arange(0, states.shape[0])
    num_states = states.shape[1]
    fig = make_subplots(rows=num_states, cols=num_columns, subplot_titles=state_names)

    for i in range(0, num_states, num_columns):
        for j in range(num_columns):
            fig.add_trace(go.Scatter(x=timesteps, y=states[:, i+j], mode="lines"), row=i+1, col=j+1)

    fig.update_layout(showlegend=True, title_text="State Trajectories")
    return fig