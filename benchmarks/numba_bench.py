# numba_bench.py - benchmarks using numba
# RMM, 25 Jun 2022
#
# This set of benchmarks mirrors some of the tests in flatsys_bench.py and
# optimal_bench.py but uses numba for cost and constraint functions to see
# how much of a difference this can make in computation times.
#

import numba
import numpy as np
import math
import control as ct
import control.flatsys as flat
import control.optimal as opt
import numba

#
# Regular version (copied from flatsys_bench.py)
#

# Vehicle steering dynamics
def vehicle_python_update(t, x, u, params):
    # Get the parameters for the model
    l = params.get('wheelbase', 3.)         # vehicle wheelbase
    phimax = params.get('maxsteer', 0.5)    # max steering angle (rad)

    # Saturate the steering input (use min/max instead of clip for speed)
    phi = max(-phimax, min(u[1], phimax))

    # Return the derivative of the state
    return np.array([
        math.cos(x[2]) * u[0],            # xdot = cos(theta) v
        math.sin(x[2]) * u[0],            # ydot = sin(theta) v
        (u[0] / l) * math.tan(phi)        # thdot = v/l tan(phi)
    ])

def vehicle_python_output(t, x, u, params):
    return x                            # return x, y, theta (full state)

# Flatness structure
def vehicle_python_forward(x, u, params={}):
    b = params.get('wheelbase', 3.)             # get parameter values
    zflag = [np.zeros(3), np.zeros(3)]          # list for flag arrays
    zflag[0][0] = x[0]                          # flat outputs
    zflag[1][0] = x[1]
    zflag[0][1] = u[0] * np.cos(x[2])           # first derivatives
    zflag[1][1] = u[0] * np.sin(x[2])
    thdot = (u[0]/b) * np.tan(u[1])             # dtheta/dt
    zflag[0][2] = -u[0] * thdot * np.sin(x[2])  # second derivatives
    zflag[1][2] =  u[0] * thdot * np.cos(x[2])
    return zflag

def vehicle_python_reverse(zflag, params={}):
    b = params.get('wheelbase', 3.)             # get parameter values
    x = np.zeros(3); u = np.zeros(2)            # vectors to store x, u
    x[0] = zflag[0][0]                          # x position
    x[1] = zflag[1][0]                          # y position
    x[2] = np.arctan2(zflag[1][1], zflag[0][1]) # angle
    u[0] = zflag[0][1] * np.cos(x[2]) + zflag[1][1] * np.sin(x[2])
    thdot_v = zflag[1][2] * np.cos(x[2]) - zflag[0][2] * np.sin(x[2])
    u[1] = np.arctan2(thdot_v, u[0]**2 / b)
    return x, u

vehicle_python = flat.FlatSystem(
    vehicle_python_forward, vehicle_python_reverse, vehicle_python_update,
    vehicle_python_output, inputs=('v', 'delta'), outputs=('x', 'y', 'theta'),
    states=('x', 'y', 'theta'))

#
# Numba jit versions
#

# Vehicle steering dynamics
@numba.jit(forceobj=True, cache=True)
def vehicle_jit_update(t, x, u, params):
    # Get the parameters for the model
    l = params.get('wheelbase', 3.)         # vehicle wheelbase
    phimax = params.get('maxsteer', 0.5)    # max steering angle (rad)

    # Saturate the steering input (use min/max instead of clip for speed)
    phi = max(-phimax, min(u[1], phimax))

    # Return the derivative of the state
    return np.array([
        math.cos(x[2]) * u[0],            # xdot = cos(theta) v
        math.sin(x[2]) * u[0],            # ydot = sin(theta) v
        (u[0] / l) * math.tan(phi)        # thdot = v/l tan(phi)
    ])

@numba.jit(forceobj=True, cache=True)
def vehicle_jit_output(t, x, u, params):
    return x                            # return x, y, theta (full state)

# Flatness structure
@numba.jit(forceobj=True, cache=True)
def vehicle_jit_forward(x, u, params={}):
    b = params.get('wheelbase', 3.)             # get parameter values
    zflag = [np.zeros(3), np.zeros(3)]          # list for flag arrays
    zflag[0][0] = x[0]                          # flat outputs
    zflag[1][0] = x[1]
    zflag[0][1] = u[0] * np.cos(x[2])           # first derivatives
    zflag[1][1] = u[0] * np.sin(x[2])
    thdot = (u[0]/b) * np.tan(u[1])             # dtheta/dt
    zflag[0][2] = -u[0] * thdot * np.sin(x[2])  # second derivatives
    zflag[1][2] =  u[0] * thdot * np.cos(x[2])
    return zflag

@numba.jit(forceobj=True, cache=True)
def vehicle_jit_reverse(zflag, params={}):
    b = params.get('wheelbase', 3.)             # get parameter values
    x = np.zeros(3); u = np.zeros(2)            # vectors to store x, u
    x[0] = zflag[0][0]                          # x position
    x[1] = zflag[1][0]                          # y position
    x[2] = np.arctan2(zflag[1][1], zflag[0][1]) # angle
    u[0] = zflag[0][1] * np.cos(x[2]) + zflag[1][1] * np.sin(x[2])
    thdot_v = zflag[1][2] * np.cos(x[2]) - zflag[0][2] * np.sin(x[2])
    u[1] = np.arctan2(thdot_v, u[0]**2 / b)
    return x, u

vehicle_jit = flat.FlatSystem(
    vehicle_jit_forward, vehicle_jit_reverse, vehicle_jit_update,
    vehicle_jit_output, inputs=('v', 'delta'), outputs=('x', 'y', 'theta'),
    states=('x', 'y', 'theta'))

#
# Numba njit versions
#

typed_params = numba.typed.Dict.empty(
    key_type=numba.types.unicode_type,
    value_type=numba.types.float64
)

# Vehicle steering dynamics
@numba.jit(nopython=True, cache=True)
def vehicle_njit_update(t, x, u, params):
    # Get the parameters for the model
    l = params.get('wheelbase', 3.)         # vehicle wheelbase
    phimax = params.get('maxsteer', 0.5)    # max steering angle (rad)

    # Saturate the steering input (use min/max instead of clip for speed)
    phi = max(-phimax, min(u[1], phimax))

    # Return the derivative of the state
    return np.array([
        math.cos(x[2]) * u[0],            # xdot = cos(theta) v
        math.sin(x[2]) * u[0],            # ydot = sin(theta) v
        (u[0] / l) * math.tan(phi)        # thdot = v/l tan(phi)
    ])

@numba.jit(nopython=True, cache=True)
def vehicle_njit_output(t, x, u, params):
    return x                            # return x, y, theta (full state)

# Flatness structure
@numba.jit(nopython=True, cache=True)
def vehicle_njit_forward(x, u, params):
    b = params.get('wheelbase', 3.)             # get parameter values
    zflag = [np.zeros(3), np.zeros(3)]          # list for flag arrays
    zflag[0][0] = x[0]                          # flat outputs
    zflag[1][0] = x[1]
    zflag[0][1] = u[0] * np.cos(x[2])           # first derivatives
    zflag[1][1] = u[0] * np.sin(x[2])
    thdot = (u[0]/b) * np.tan(u[1])             # dtheta/dt
    zflag[0][2] = -u[0] * thdot * np.sin(x[2])  # second derivatives
    zflag[1][2] =  u[0] * thdot * np.cos(x[2])
    return zflag

@numba.jit(nopython=True, cache=True)
def vehicle_njit_reverse(zflag, params):
    b = params.get('wheelbase', 3.)             # get parameter values
    x = np.zeros(3); u = np.zeros(2)            # vectors to store x, u
    x[0] = zflag[0][0]                          # x position
    x[1] = zflag[1][0]                          # y position
    x[2] = np.arctan2(zflag[1][1], zflag[0][1]) # angle
    u[0] = zflag[0][1] * np.cos(x[2]) + zflag[1][1] * np.sin(x[2])
    thdot_v = zflag[1][2] * np.cos(x[2]) - zflag[0][2] * np.sin(x[2])
    u[1] = np.arctan2(thdot_v, u[0]**2 / b)
    return x, u

vehicle_njit = flat.FlatSystem(
    vehicle_njit_forward, vehicle_njit_reverse, vehicle_njit_update,
    vehicle_njit_output, inputs=('v', 'delta'), outputs=('x', 'y', 'theta'),
    states=('x', 'y', 'theta'), params=typed_params)

#
# Basis functions
#

from numba import int32, float64

@numba.jit(nopython=True, cache=True)
def comb(n, k):
    result = 1
    for i in range(n, n-k, -1):
        result *= i
    for i in range(2, k+1):
        result /= i
    return result

@numba.jit(nopython=True, cache=True)
def factorial(k):
    result = 1
    for i in range(2, k+1):
        result *= i
    return result

@numba.jit(nopython=True, cache=True)
def bezier_deriv(i, k, t, N, T):
    if i < N and k < N:
        n = N - 1
        u = t / T
        return comb(n, i) * sum([
            (-1)**(j-i) *
            comb(n-i, j-i) * factorial(j)/factorial(j-k) * \
            np.power(u, j-k) / np.power(T, k)
            for j in range(max(i, k), n+1)
        ])
    else:
        return 0. * t

class njit_bezier(flat.BezierFamily):
    def eval_deriv(self, i, k, t, var=None):
        return bezier_deriv(i, k, t, self.N, self.T)

#
# Trajectory description
#

# Initial and final conditions
x0 = [0., -2., 0.]; u0 = [10., 0.]
xf = [100., 2., 0.]; uf = [10., 0.]
Tf = 10.

# Define the time points where the cost/constraints will be evaluated
# (use a large number to generate more computation)
timepts = np.linspace(0, Tf, 10, endpoint=True)

#
# Benchmark test parameters
#

basis_params = (['poly', 'bezier', 'bspline'], [8, 10, 12])
basis_param_names = ["basis", "size"]

def get_basis(name, size):
    if name == 'poly':
        basis = flat.PolyFamily(size, T=Tf)
    elif name == 'bezier':
        basis = flat.BezierFamily(size, T=Tf)
    elif name == 'bspline':
        basis = flat.BSplineFamily([0, Tf/2, Tf], size)
    return basis

numba_params = (['python', 'jit', 'njit'], )
numba_param_names = ['method']
numba_param_dict = {
    'python': vehicle_python, 'jit': vehicle_jit, 'njit': vehicle_njit}
numba_init_dict = {}

def get_wrapper(method):
    if method == 'python':
        return lambda fun: fun
    elif method == 'jit':
        return numba.jit(forceobj=True)
    elif method == 'njit':
        return numba.jit(nopython=True)
    else:
        raise ValueError(f"unknown method 'f{method}'")

#
# Benchmarks
#

def time_point_to_point_with_cost(basis_name, basis_size, method):
    basis = get_basis(basis_name, basis_size)
    vehicle = numba_param_dict[method]
    wrapper = get_wrapper(method)

    # Define cost and constraints
    traj_cost = wrapper(
        opt.quadratic_cost(vehicle, None, np.diag([0.1, 1.]), u0=uf))
    constraints = [
        opt.input_range_constraint(vehicle, [8., -0.1], [12., 0.1]) ]

    traj = flat.point_to_point(
        vehicle, timepts, x0, u0, xf, uf, cost=traj_cost,
        constraints=constraints, basis=basis
    )

    # Verify that the trajectory computation is correct
    x, u = traj.eval([0, Tf])
    np.testing.assert_array_almost_equal(x0, x[:, 0])
    np.testing.assert_array_almost_equal(u0, u[:, 0])
    np.testing.assert_array_almost_equal(xf, x[:, -1])
    np.testing.assert_array_almost_equal(uf, u[:, -1])

def setup_point_to_point_with_cost(basis_name, basis_size, method):
    # Run the function once to compile jit, njit function
    if numba_init_dict.get(f'{basis_name=}, {basis_size=}, {method=}', False):
        time_point_to_point_with_cost(basis_name, basis_size, method)
        numba_init_dict[f'{basis_name=}, {basis_size=}, {method=}'] = True

time_point_to_point_with_cost.params = tuple(
    list(basis_params) + list(numba_params))
time_point_to_point_with_cost.param_names = \
    basis_param_names + numba_param_names
time_point_to_point_with_cost.setup =  setup_point_to_point_with_cost


def time_solve_flat_ocp_terminal_cost(basis_name, basis_size, method):
    basis = get_basis(basis_name, basis_size)
    vehicle = numba_param_dict[method]
    wrapper = get_wrapper(method)

    # Define cost and constraints
    traj_cost = wrapper(opt.quadratic_cost(
        vehicle, None, np.diag([0.1, 1.]), u0=uf))
    term_cost = wrapper(opt.quadratic_cost(
        vehicle, np.diag([1.e3, 1.e3, 1.e3]), None, x0=xf))
    constraints = [
        opt.input_range_constraint(vehicle, [8., -0.1], [12., 0.1]) ]

    # Initial guess = straight line
    initial_guess = np.array(
        [x0[i] + (xf[i] - x0[i]) * timepts/Tf for i in (0, 1)])

    traj = flat.solve_flat_ocp(
        vehicle, timepts, x0, u0, basis=basis, initial_guess=initial_guess,
        trajectory_cost=traj_cost, constraints=constraints,
        terminal_cost=term_cost
    )

    # Verify that the trajectory computation is correct
    x, u = traj.eval([0, Tf])
    np.testing.assert_array_almost_equal(x0, x[:, 0])
    np.testing.assert_array_almost_equal(xf, x[:, -1], decimal=2)

def setup_solve_flat_ocp_terminal_cost(basis_name, basis_size, method):
    if numba_init_dict.get(f'{basis_name=}, {basis_size=}, {method=}', False):
        time_solve_flat_ocp_terminal_cost(basis_name, basis_size, method)
        numba_init_dict[f'{basis_name=}, {basis_size=}, {method=}'] = True

time_solve_flat_ocp_terminal_cost.params = tuple(
    list(basis_params) + list(numba_params))
time_solve_flat_ocp_terminal_cost.param_names = \
    basis_param_names + numba_param_names
time_solve_flat_ocp_terminal_cost.setup = setup_solve_flat_ocp_terminal_cost


def time_solve_flat_ocp_cost_wrappers(method):
    basis = get_basis('bezier', 10)
    vehicle = numba_param_dict[method]
    wrapper = get_wrapper(method)

    # Define cost and constraints
    traj_cost = wrapper(opt.quadratic_cost(
        vehicle, None, np.diag([0.1, 1.]), u0=uf))
    term_cost = wrapper(opt.quadratic_cost(
        vehicle, np.diag([1.e3, 1.e3, 1.e3]), None, x0=xf))
    constraints = [
        opt.input_range_constraint(vehicle, [8., -0.1], [12., 0.1]) ]

    # Initial guess = straight line
    initial_guess = np.array(
        [x0[i] + (xf[i] - x0[i]) * timepts/Tf for i in (0, 1)])

    traj = flat.solve_flat_ocp(
        vehicle, timepts, x0, u0, basis=basis, initial_guess=initial_guess,
        trajectory_cost=traj_cost, constraints=constraints,
        terminal_cost=term_cost
    )

    # Verify that the trajectory computation is correct
    x, u = traj.eval([0, Tf])
    np.testing.assert_array_almost_equal(x0, x[:, 0])
    np.testing.assert_array_almost_equal(xf, x[:, -1], decimal=2)

def setup_solve_flat_ocp_cost_wrappers(method):
    if numba_init_dict.get(f'{method=}', False):
        time_solve_flat_ocp_cost_wrappers(method)
        numba_init_dict[f'{method=}'] = True

time_solve_flat_ocp_cost_wrappers.params = (numba_params[0], )
time_solve_flat_ocp_cost_wrappers.param_names = ["method"]
time_solve_flat_ocp_cost_wrappers.setup = setup_solve_flat_ocp_cost_wrappers


def time_optimal_lq_wrappers(basis_method, system_method, cost_method):
    vehicle = numba_param_dict[system_method]
    wrapper = get_wrapper(cost_method)

    # Create a sufficiently controllable random system to control
    # TODO: user a nonlinear system that we can compile with numbda
    ntrys = 20
    while ntrys > 0:
        # Create a random system
        sys = ct.rss(2, 2, 2)

        # Compute the controllability Gramian
        Wc = ct.gram(sys, 'c')

        # Make sure the condition number is reasonable
        if np.linalg.cond(Wc) < 1e6:
            break

        ntrys -= 1
    assert ntrys > 0            # Something wrong if we needed > 20 tries

    # Define cost functions
    Q = np.eye(sys.nstates)
    R = np.eye(sys.ninputs) * 10

    # Figure out the LQR solution (for terminal cost)
    K, S, E = ct.lqr(sys, Q, R)

    # Define the cost functions
    traj_cost = wrapper(opt.quadratic_cost(sys, Q, R))
    term_cost = wrapper(opt.quadratic_cost(sys, S, None))
    constraints = opt.input_range_constraint(
        sys, -np.ones(sys.ninputs), np.ones(sys.ninputs))

    # Define the initial condition, time horizon, and time points
    x0 = np.ones(sys.nstates)
    Tf = 10
    timepts = np.linspace(0, Tf, 10)

    # Define the basis vechicles
    basis = njit_bezier(10, Tf) if basis_method == 'njit' \
        else flat.BezierFamily(10, Tf)

    res = opt.solve_ocp(
        sys, timepts, x0, traj_cost, constraints, terminal_cost=term_cost,
        basis=basis
    )
    # Only count this as a benchmark if we converged
    assert res.success

def setup_optimal_lq_wrappers(basis_method, system_method, cost_method):
    # Call the function once to force compilation
    if numba_init_dict.get(
            f'{basis_method=}, {system_method=} {cost_method=}', False):
        time_optimal_lq_wrappers(basis_method, system_method, cost_method)
        numba_init_dict[
            f'{basis_method=}, {system_method=} {cost_method=}'] = True


# Parameterize the test against different choices of integrator and minimizer
time_optimal_lq_wrappers.params = (
    ['njit', 'python'], numba_params[0], numba_params[0])
time_optimal_lq_wrappers.param_names = ["basis", "system", "cost"]
time_optimal_lq_wrappers.setup = setup_optimal_lq_wrappers
