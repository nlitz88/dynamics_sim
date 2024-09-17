"""Unit tests for functions defined in the quatmath module."""

import numpy as np
import quatmath as qm

def test_invert():
    """Test the invert function."""
    # Test the invert function with a simple example.
    q = np.array([1, 0, 0, 0])
    q_inv = qm.invert(q)
    assert np.allclose(q_inv, q)

    # Test the invert function with a more complex example.
    q = np.array([1, 1, 1, 1])
    q_inv = qm.invert(q)
    assert np.allclose(q_inv, np.array([0.25, -0.25, -0.25, -0.25]))

def test_skew():
    """Test the skew function."""
    # Test the skew function with a simple example.
    v = np.array([1, 2, 3])
    v_skew = qm.skew(v)
    assert np.allclose(v_skew, np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]]))

    # Test the skew function with a more complex example.
    v = np.array([3, 2, 1])
    v_skew = qm.skew(v)
    assert np.allclose(v_skew, np.array([[0, -1, 2], [1, 0, -3], [-2, 3, 0]]))

def test_hat():
    """Test the hat function."""
    # Test the hat function with a simple example.
    v = np.array([1, 2, 3])
    v_hat = qm.hat(v)
    assert np.allclose(v_hat, np.array([0, 1, 2, 3]))

    # Test the hat function with a more complex example.
    v = np.array([3, 2, 1])
    v_hat = qm.hat(v)
    assert np.allclose(v_hat, np.array([0, 3, 2, 1]))

def test_unhat():
    """Test the unhat function."""
    # Test the unhat function with a simple example.
    v = np.array([0, 1, 2, 3])
    v_unhat = qm.unhat(v)
    assert np.allclose(v_unhat, np.array([1, 2, 3]))

    # Test the unhat function with a more complex example.
    v = np.array([0, 3, 2, 1])
    v_unhat = qm.unhat(v)
    assert np.allclose(v_unhat, np.array([3, 2, 1]))

def test_left_multiply():
    """Test the left_multiply function."""
    # Test the left_multiply function with a simple example.
    q1 = np.array([1, 0, 0, 0])
    q2 = np.array([1, 0, 0, 0])
    q_left_mult = qm.L(q1, q2)
    assert np.allclose(q_left_mult, q2)

    # Test the left_multiply function with a more complex example.
    q1 = np.array([1, 1, 1, 1])
    q2 = np.array([1, 1, 1, 1])
    q_left_mult = qm.L(q1, q2)
    assert np.allclose(q_left_mult, np.array([0, 2, 2, 2]))

def test_right_multiply():
    """Test the right_multiply function."""
    # Test the right_multiply function with a simple example.
    q1 = np.array([1, 0, 0, 0])
    q2 = np.array([1, 0, 0, 0])
    q_right_mult = qm.R(q1, q2)
    assert np.allclose(q_right_mult, q2)

    # Test the right_multiply function with a more complex example.
    q1 = np.array([1, 1, 1, 1])
    q2 = np.array([1, 1, 1, 1])
    q_right_mult = qm.R(q1, q2)
    assert np.allclose(q_right_mult, np.array([0, 2, 2, 2]))

def test_compose():
    """Test the compose function."""
    # Test the compose function with a simple example.
    q1 = np.array([1, 0, 0, 0])
    q2 = np.array([1, 0, 0, 0])
    q_composed = qm.compose(q1, q2)
    assert np.allclose(q_composed, np.array([1, 0, 0, 0]))

    # Test the compose function with a more complex example.
    q1 = np.array([1, 1, 1, 1])
    q2 = np.array([1, 1, 1, 1])
    q_composed = qm.compose(q1, q2)
    assert np.allclose(q_composed, np.array([0, 2, 2, 2]))

def test_q():
    """Test the Q(q) function that returns the equivalent rotation matrix from a
    given quaternion.
    """
    # Test the Q(q) function with a simple example.
    q = np.array([1, 0, 0, 0])
    R = qm.Q(q)
    assert np.allclose(R, np.eye(3))

    # Test the Q(q) function with a more complex example.
    q = np.array([1, 1, 1, 1])
    R = qm.Q(q)
    assert np.allclose(R, np.array([[0, 0, 2], [0, 0, 2], [-2, -2, 0]]))
