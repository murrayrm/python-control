# -*- coding: utf-8 -*-
"""
Tests for sysnorm module.

Created on Mon Jan  8 11:31:46 2024
Author: Henrik Sandberg
"""

import control as ct
import numpy as np
import pytest


def test_norm_1st_order_stable_system():
    """First-order stable continuous-time system"""
    s = ct.tf('s')
    
    G1 = 1/(s+1)
    assert np.allclose(ct.norm(G1, p='inf'), 1.0) # Comparison to norm computed in MATLAB
    assert np.allclose(ct.norm(G1, p=2), 0.707106781186547) # Comparison to norm computed in MATLAB
    
    Gd1 = ct.sample_system(G1, 0.1)
    assert np.allclose(ct.norm(Gd1, p='inf'), 1.0) # Comparison to norm computed in MATLAB
    assert np.allclose(ct.norm(Gd1, p=2), 0.223513699524858) # Comparison to norm computed in MATLAB


def test_norm_1st_order_unstable_system():
    """First-order unstable continuous-time system"""
    s = ct.tf('s')
    
    G2 = 1/(1-s)
    assert np.allclose(ct.norm(G2, p='inf'), 1.0) # Comparison to norm computed in MATLAB
    with pytest.warns(UserWarning, match="System is unstable!"):
        assert ct.norm(G2, p=2) == float('inf') # Comparison to norm computed in MATLAB
    
    Gd2 = ct.sample_system(G2, 0.1)
    assert np.allclose(ct.norm(Gd2, p='inf'), 1.0) # Comparison to norm computed in MATLAB
    with pytest.warns(UserWarning, match="System is unstable!"):
        assert ct.norm(Gd2, p=2) == float('inf') # Comparison to norm computed in MATLAB

def test_norm_2nd_order_system_imag_poles():
    """Second-order continuous-time system with poles on imaginary axis"""
    s = ct.tf('s')
   
    G3 = 1/(s**2+1)
    with pytest.warns(UserWarning, match="Poles close to, or on, the imaginary axis."):
        assert ct.norm(G3, p='inf') == float('inf') # Comparison to norm computed in MATLAB
    with pytest.warns(UserWarning, match="Poles close to, or on, the imaginary axis."):
        assert ct.norm(G3, p=2) == float('inf') # Comparison to norm computed in MATLAB
    
    Gd3 = ct.sample_system(G3, 0.1)
    with pytest.warns(UserWarning, match="Poles close to, or on, the complex unit circle."):
        assert ct.norm(Gd3, p='inf') == float('inf') # Comparison to norm computed in MATLAB
    with pytest.warns(UserWarning, match="Poles close to, or on, the complex unit circle."):
        assert ct.norm(Gd3, p=2) == float('inf') # Comparison to norm computed in MATLAB

def test_norm_3rd_order_mimo_system():
    """Third-order stable MIMO continuous-time system"""
    A = np.array([[-1.017041847539126,  -0.224182952826418,   0.042538079149249],
                  [-0.310374015319095,  -0.516461581407780,  -0.119195790221750],
                  [-1.452723568727942,   1.7995860837102088,  -1.491935830615152]])
    B = np.array([[0.312858596637428,  -0.164879019209038],
                  [-0.864879917324456,   0.627707287528727],
                  [-0.030051296196269,   1.093265669039484]])
    C = np.array([[1.109273297614398,   0.077359091130425,  -1.113500741486764],
                  [-0.863652821988714,  -1.214117043615409,  -0.006849328103348]])
    D = np.zeros((2,2))
    G4 = ct.ss(A,B,C,D) # Random system generated in MATLAB
    assert np.allclose(ct.norm(G4, p='inf'), 4.276759162964244) # Comparison to norm computed in MATLAB
    assert np.allclose(ct.norm(G4, p=2), 2.237461821810309) # Comparison to norm computed in MATLAB
    
    Gd4 = ct.sample_system(G4, 0.1)
    assert np.allclose(ct.norm(Gd4, p='inf'), 4.276759162964228) # Comparison to norm computed in MATLAB
    assert np.allclose(ct.norm(Gd4, p=2), 0.707434962289554) # Comparison to norm computed in MATLAB


#
# ChatGPT generated unit tests to cover exceptions and warnings
#
# Prompt: Generate a set of unit tests using the pytest framework that
# capture the various exceptions and warnings in the following code:
#

import pytest
import numpy as np
import control as ct
from ..sysnorm import norm, _h2norm_slycot
from .conftest import slycotonly

@pytest.fixture
def stable_system():
    """Fixture for a stable system."""
    return ct.tf([1], [1, 2, 1])  # Stable continuous-time system


@pytest.fixture
def unstable_system():
    """Fixture for an unstable system."""
    return ct.tf([1], [1, -2, 1])  # Unstable continuous-time system


@pytest.fixture
def direct_feedthrough_system():
    """Fixture for a system with direct feedthrough."""
    A = np.array([[-1]])
    B = np.array([[0.0]])
    C = np.array([[0.0]])
    D = np.array([[1.0]])  # Non-zero direct feedthrough
    return ct.ss(A, B, C, D)


@pytest.fixture
def discrete_stable_system():
    """Fixture for a stable discrete-time system."""
    A = np.array([[0.5]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    D = np.array([[0.0]])
    return ct.ss(A, B, C, D, True)  # Discrete-time stable system


@pytest.fixture
def discrete_unstable_system():
    """Fixture for an unstable discrete-time system."""
    A = np.array([[1.1]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    D = np.array([[0.0]])
    return ct.ss(A, B, C, D, True)  # Discrete-time unstable system


def test_invalid_system_type():
    """Test that an invalid system type raises a TypeError."""
    with pytest.raises(
            TypeError,
            match="must be a ``StateSpace`` or ``TransferFunction``"):
        norm(42)


def test_unstable_warning(unstable_system):
    """Test that a warning is raised for unstable systems."""
    with pytest.warns(UserWarning, match="System is unstable!"):
        norm(unstable_system)


def test_direct_feedthrough_warning(direct_feedthrough_system):
    """Test that a warning is raised for systems with direct feedthrough."""
    with pytest.warns(
            UserWarning, match="System has a direct feedthrough term!"):
        norm(direct_feedthrough_system)


def test_h2_norm_infinite_with_pole_on_stability_boundary():
    """Test H2 norm when a pole is on the stability boundary."""
    A = np.array([[0.0]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    D = np.array([[0.0]])
    system = ct.ss(A, B, C, D)
    with pytest.warns(
            UserWarning, match="Poles close to, or on, the imaginary axis"):
        assert norm(system, p=2) == float("inf")


def test_h2_norm_finite(stable_system):
    """Test H2 norm for a stable system."""
    assert np.isclose(norm(stable_system, p=2), 0.5, atol=1e-6)


def test_h2_norm_discrete_unstable():
    """Test H2 norm for a discrete-time unstable system."""
    A = np.array([[1.1]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    D = np.array([[0.0]])
    system = ct.ss(A, B, C, D, True)  # Discrete-time
    with pytest.warns(UserWarning, match="System is unstable!"):
        assert norm(system, p=2) == float("inf")


def test_unsupported_norm():
    """Test unsupported norm type."""
    system = ct.tf([1], [1, 2, 1])
    with pytest.raises(ct.ControlArgument, match="Norm computation for p=3 currently not supported"):
        norm(system, p=3)

@pytest.mark.skipif(ct.slycot_check(), reason="slycot installed")
def test_slycot_import_error():
    """Test behavior when slycot is not installed."""
    with pytest.raises(ct.ControlSlycot, match="Can't find slycot"):
        _h2norm_slycot(ct.tf([1], [1, 0, 1]))


@slycotonly
def test_slycot_arithmetic_error(monkeypatch):
    """Test SlycotArithmeticError handling."""
    with pytest.warns(
            UserWarning,
            match="System has pole\\(s\\) on the stability boundary!"):
        result = _h2norm_slycot(ct.tf([1], [1, 0, 1]))
        assert result == float("inf")


def test_poles_on_imag_axis_ct_warning():
    """Test poles on imaginary axis in continuous-time systems."""
    A = np.array([[0.0]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    D = np.array([[0.0]])
    system = ct.ss(A, B, C, D)
    with pytest.warns(UserWarning, match="Poles close to, or on, the imag"):
        assert norm(system, p="inf") == float("inf")


def test_poles_on_unit_circle_dt_warning():
    """Test poles on unit circle in discrete-time systems."""
    A = np.array([[1.0]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    D = np.array([[0.0]])
    system = ct.ss(A, B, C, D, True)  # Discrete-time
    with pytest.warns(UserWarning, match="Poles close to, or on, the complex"):
        assert norm(system, p="inf") == float("inf")


def test_discrete_pole_on_unit_circle_exception():
    """Test discrete systems with poles on unit circle raise exception."""
    A = np.array([[1.0]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    D = np.array([[0.0]])
    system = ct.ss(A, B, C, D, True)  # Discrete-time
    with pytest.warns(
            UserWarning,
            match="Poles close to, or on, the complex unit circle"):
        norm(system, p="inf")


def test_finite_infinity_norm_stable_system(stable_system):
    """Test finite L-infinity norm for stable continuous-time system."""
    result = norm(stable_system, p="inf", tol=1e-6)
    assert np.isclose(result, 1.0, atol=1e-6)  # Expected value


def test_finite_infinity_norm_stable_discrete_system(discrete_stable_system):
    """Test finite L-infinity norm for stable discrete-time system."""
    result = norm(discrete_stable_system, p="inf", tol=1e-6)
    assert result > 0  # Expect a positive finite value


def test_unsupported_method():
    """Test unsupported methods for L-infinity norm."""
    system = ct.tf([1], [1, 2, 1])
    with pytest.raises(
        ct.ControlArgument,
        match="Norm computation for p=stuff currently not supported"
    ):
        norm(system, p="stuff")


def test_custom_tolerance_infinity_norm(stable_system):
    """Test L-infinity norm computation with custom tolerance."""
    result = norm(stable_system, p="inf", tol=1e-3)
    assert np.isclose(result, 1.0, atol=1e-3)  # Verify with higher tolerance
        
