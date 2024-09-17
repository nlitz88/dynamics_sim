"""Functions for converting between qauternions and rotation matrix attitude
representations.

Based on Kevin Tracy's 16-745 (OCRL) recitation on attitude:
https://www.youtube.com/watch?v=pXbu2YYBSmY&t=1329s

Additionally, many of these dynamics and conventions are based on the "Planning
with Attitude paper" here:
https://rexlab.ri.cmu.edu/papers/planning_with_attitude.pdf

The quaternion is represented as a 4-element numpy array in the form [w, x, y,
z], where w is the scalar part and x, y, z are the vector part.
"""

# NOTE: This library maybe be pointeless if you have access to something like
# transforms3d.

import numpy as np

# "H" matrix constant. Used for converting a vector in R3 to a vector in R4 via
# matrix multiplication.
H = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

# "T" matrix constant. Used for inverting / conjugating a quaternion using
# matrix multiplication. Just negates/inverts the vector part of the quaternion.
T = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, -1.0]
])

def skew(v: np.ndarray) -> np.ndarray:
    """Compute the skew symmetric matrix of a vector in R3.

    Args:
        v (np.ndarray): A vector in R3.

    Returns:
        np.ndarray: The skew symmetric matrix of the given vector v in R3.
    """
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])

def hat(v: np.ndarray) -> np.ndarray:
    """Compute the "hat" map of a vector in R3 to a vector in R4.

    Args:
        v (np.ndarray): A vector in R3.

    Returns:
        np.ndarray: The "hat" map of the given vector v in R3 to a vector in R4.

    NOTE: This function uses np.dot under the hood--this may have performance
    implications if you are passing in an array of vectors to be mapped. Might
    want to consider using matmul instead.
    """
    return np.dot(H, v)

def unhat(v: np.ndarray) -> np.ndarray:
    """Compute the "unhat" map of a vector in R4 to a vector in R3.

    Args:
        v (np.ndarray): A "hatted" vector in R4.

    Returns:
        np.ndarray: The "unhat" map of the given vector v in R4 to a vector in R3.

    NOTE: This function uses np.dot under the hood--this may have performance
    implications if you are passing in an array of vectors to be mapped. Might
    want to consider using matmul instead.
    """
    return np.dot(H.T, v)

def invert(q: np.ndarray) -> np.ndarray:
    """Invert a quaternion.

    Args:
        q (np.ndarray): A quaternion in the form [w, x, y, z].

    Returns:
        np.ndarray: The inverted quaternion.
    """
    return np.dot(T, q)

def L(q: np.ndarray) -> np.ndarray:
    """Compute the "L" matrix from a given quaternion q.

    Args:
        q (np.ndarray): A quaternion in the form [w, x, y, z].

    Returns:
        np.ndarray: The "L" matrix corresponding to the given quaternion q.
    
    NOTE: This function directly forms skew symmetric matrices from the vector
    part of the quaternion. Could use the "skew" helper function instead, but
    this avoids the overhead of constructing intermediate arrays.
    """
    w, x, y, z = q
    return np.array([
        [w, -x, -y, -z],
        [x, w, -z, y],
        [y, z, w, -x],
        [z, -y, x, w]
    ])

def R(q: np.ndarray) -> np.ndarray:
    """Compute the "R" matrix from a given quaternion q.

    Args:
        q (np.ndarray): A quaternion in the form [w, x, y, z].

    Returns:
        np.ndarray: The "R" matrix corresponding to the given quaternion q.

    NOTE: This function directly forms skew symmetric matrices from the vector
    part of the quaternion. Could use the "skew" helper function instead, but
    this avoids the overhead of constructing intermediate arrays.
    """
    w, x, y, z = q
    return np.array([
        [w, -x, -y, -z],
        [x, w, z, -y],
        [y, -z, w, x],
        [z, y, -x, w]
    ])


# Write a function to "compose" two rotations == multiply two quaternions. Do
# this using the L and R functions defined above.
# def compose()

def compose(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Compose two rotations by multiplying two quaternions in the order q1 (*)
    q2.

    Args:
        q1 (np.ndarray): The first quaternion in the form [w, x, y, z].
        q2 (np.ndarray): The second quaternion in the form [w, x, y, z].

    Returns:
        np.ndarray: The quaternion that results from composing the two rotations
        q1 (*) q2.
    """
    return np.dot(L(q1), q2)


# Define function to compute "Q(q)" recover a rotation matrix from a given
# quaternion.
def Q(q: np.ndarray) -> np.ndarray:
    """Compute the equivalent rotation matrix from a quaternion q.

    Args:
        q (np.ndarray): A quaternion in the form [w, x, y, z].

    Returns:
        np.ndarray: The rotation matrix corresponding to the given quaternion q.
    """
    return H.T @ R(q).T @ L(q) @ H

def G(q: np.ndarray) -> np.ndarray:
    """Compute the attitude Jacobian matrix given a quaternion q
    
    Args:
        q (np.ndarray): A quaternion in the form [w, x, y, z].
    
    Returns:
        np.ndarray: The attitude Jacobian matrix corresponding to the given
        quaternion q.
    """
    return np.dot(L(q), H)