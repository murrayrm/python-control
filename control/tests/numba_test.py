# numba_test.py - tests for numba-enhanced optimization based control
# RMM, 29 Aug 2022
#
# This set of tests checks to see that it is possible to provide
# numba-compiled functions to the flatsys and optimal modules within
# python-contol.  It requires numba to be installed and is not currently
# part of the standard test suite.

import pytest
import numpy as np
import scipy as sp
import math
import control as ct
import control.optimal as opt
import control.flatsys as flat

try:
    import numba
    numba_jit = numba.jit
    numba_installed = True
except ImportError:
    def numba_jit(*args, **kwargs):
        return lambda fun: fun
    numba_installed = False

@pytest.mark.skipif(not numba_installed, reason="numba not installed")
@pytest.mark.parametrize("wrapper",[
    lambda fun: fun,
    numba_jit(),
    numba_jit(nopython=True),
])
def test_finite_horizon_simple(wrapper):
    # Define a linear system with constraints
    # Source: https://www.mpt3.org/UI/RegulationProblem

    # LTI prediction model
    sys = ct.ss([[1, 1], [0, 1]], [[1], [0.5]], np.eye(2), 0, 1)

    # State and input constraints
    constraints = [
        (sp.optimize.LinearConstraint, np.eye(3), [-5, -5, -1], [5, 5, 1]),
    ]

    # Quadratic state and input penalty
    Q = [[1, 0], [0, 1]]
    R = [[1]]
    cost = wrapper(opt.quadratic_cost(sys, Q, R))

    # Set up the optimal control problem
    time = np.arange(0, 5, 1)
    x0 = [4, 0]

    # Retrieve the full open-loop predictions
    res = opt.solve_ocp(
        sys, time, x0, cost, constraints, squeeze=True,
        terminal_cost=cost)     # include to match MPT3 formulation
    t, u_openloop = res.time, res.inputs
    np.testing.assert_almost_equal(
        u_openloop, [-1, -1, 0.1393, 0.3361, -5.204e-16], decimal=4)

    # Make sure the final cost is correct
    assert math.isclose(res.cost, 32.4898, rel_tol=1e-5)


@pytest.mark.skipif(not numba_installed, reason="numba not installed")
@pytest.mark.parametrize("wrapper",[
    lambda fun: fun,
    numba_jit(),
    numba_jit(nopython=True),
])
def test_flat_doubleint(wrapper):
    # Define a second order integrator
    sys = ct.ss([[0, 1], [0, 0]], [[0], [1]], [[1, 0]], 0)
    flatsys = flat.LinearFlatSystem(sys)

    x0 = [1, 0]; u0 = [0]
    xf = [-1, 0]; uf = [0]
    Tf = 10
    timepts = np.linspace(0, Tf, 10)
    basis = flat.BezierFamily(10, Tf)

    # Simple point to point trajectory generation
    traj = flat.point_to_point(
        flatsys, Tf, x0, u0, xf, uf, basis=basis)

    # Verify that the trajectory computation is correct
    x, u = traj.eval([0, Tf])
    np.testing.assert_array_almost_equal(x0, x[:, 0])
    np.testing.assert_array_almost_equal(u0, u[:, 0])
    np.testing.assert_array_almost_equal(xf, x[:, 1])
    np.testing.assert_array_almost_equal(uf, u[:, 1])

    # Add cost functions
    cost = wrapper(opt.quadratic_cost(flatsys, np.eye(2), 1))
    
    traj = flat.point_to_point(
        flatsys, timepts, x0, u0, xf, uf, cost=cost, basis=basis)
    x, u = traj.eval(timepts)
    np.testing.assert_array_almost_equal(x0, x[:, 0])
    np.testing.assert_array_almost_equal(u0, u[:, 0])
    np.testing.assert_array_almost_equal(xf, x[:, -1])
    np.testing.assert_array_almost_equal(uf, u[:, -1])

    # Solve as an optimal control problem with terminal cost
    term_cost = wrapper(opt.quadratic_cost(flatsys, 1e3, 1e3, x0=xf, u0=uf))
    traj = flat.solve_flat_ocp(
        flatsys, timepts, x0, u0, terminal_cost=term_cost, basis=basis)
    x, u = traj.eval(timepts)
    np.testing.assert_array_almost_equal(x0, x[:, 0])
    np.testing.assert_array_almost_equal(u0, u[:, 0])
    np.testing.assert_array_almost_equal(xf, x[:, -1], decimal=2)
    np.testing.assert_array_almost_equal(uf, u[:, -1], decimal=2)

    # Solve the same problem with a nonlinear constraint type
    lb, ub = [-2, np.min(x[1]) * 0.95], [2, 1]
    nl_constraints = [
        (sp.optimize.NonlinearConstraint, wrapper(lambda x, u: x), lb, ub)]
    traj = flat.solve_flat_ocp(
            flatsys, timepts, x0, u0, cost=cost, terminal_cost=term_cost,
            constraints=nl_constraints, basis=basis)
    x, u = traj.eval(timepts)
    np.testing.assert_array_almost_equal(x0, x[:, 0])
    np.testing.assert_array_almost_equal(u0, u[:, 0])
    
    # Make sure we got close on the terminal condition
    assert all(np.abs(x[:, -1] - xf) < 0.1)
