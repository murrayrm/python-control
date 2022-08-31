# ntg_bench.py - benchmarks for flat systems using NTG
# RMM, 4 Sep 2022
#
# This benchmark tests the timing for the flat systems tested in
# flatsys_bench.py but using NTG (for comparison with python control).

import numpy as np
import math
try:
    import ntg
    ntg_installed = True
except ImportError:
    ntg_installed = False

#
# System setup: vehicle steering (bicycle model)
#

# Vehicle steering dynamics
def vehicle_update(t, x, u, params):
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

def vehicle_output(t, x, u, params):
    return x                            # return x, y, theta (full state)

# Flatness structure
def vehicle_forward(x, u, params={}):
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

def vehicle_reverse(zflag, params={}):
    b = params.get('wheelbase', 3.)             # get parameter values
    x = np.zeros(3); u = np.zeros(2)            # vectors to store x, u
    x[0] = zflag[0][0]                          # x position
    x[1] = zflag[1][0]                          # y position
    x[2] = np.arctan2(zflag[1][1], zflag[0][1]) # angle
    u[0] = zflag[0][1] * np.cos(x[2]) + zflag[1][1] * np.sin(x[2])
    thdot_v = zflag[1][2] * np.cos(x[2]) - zflag[0][2] * np.sin(x[2])
    u[1] = np.arctan2(thdot_v, u[0]**2 / b)
    return x, u

vehicle = ntg.FlatSystem(2, [3, 3]) if ntg_installed else None

# Initial and final conditions
x0 = [0., -2., 0.]; u0 = [10., 0.]
xf = [100., 2., 0.]; uf = [10., 0.]
Tf = 10

# Define the time points where the cost/constraints will be evaluated
timepts = np.linspace(0, Tf, 10, endpoint=True)

#
# Benchmark test parameters
#

basis_params = (['bspline'], [8, 10, 12])
basis_param_names = ["basis", "size"]

def get_basis(name, size):
    if name == 'bspline':
        basis = ntg.BSplineFamily([0, Tf/2, Tf], size)
    else:
        raise ValueError(f"basis type '{name}' not supported")
    return basis

#
# NTG definitions and conversions
#

z0 = vehicle_forward(x0, u0)
zf = vehicle_forward(x0, u0)

def quadratic_cost(sys, Q, R, x0=None, u0=None):
    if Q is not None or x0 is not None:
        raise NotImplemented("Q, x0 not supported")

    # Set the cost based on flat variables
    Qlist = [np.zeros((sys.flaglen[i], sys.flaglen[i])) for i in [0, 1]]
    Qlist[0][1], Qlist[1][1] = R[0][0], R[0][0]
    Qlist[0][2], Qlist[1][2] = R[1][1], R[1][1]
        
def input_range_constraints(sys, lb, ub):
    @numba.cfunc(ntg.numba_trajectory_constraint_signature)
    def _input_range_constraints(mode, nstate, j, f, df, zp):
        b = 3.                                          # wheelbase
        x = np.zeros(3); u = np.zeros(2)                # vectors to store x, u
        x[0] = zflag[0][0]                              # x position
        x[1] = zflag[1][0]                              # y position
        x[2] = math.arctan2(zflag[1][1], zflag[0][1])   # angle

        # Figure out the inputs
        thdot_v = zflag[1][2] * math.cos(x[2]) - zflag[0][2] * math.sin(x[2])
        u[0] = zflag[0][1] * math.cos(x[2]) + zflag[1][1] * math.sin(x[2])
        u[1] = math.arctan2(thdot_v, u[0]**2 / b)
    
        if mode == 0 or mode == 2:
            f[0] = u[0]
            f[1] = u[1]

        if mode == 1 or mode == 2:
            df[0][0] = 0
            df[0][1] = math.cos(x[2]) + \
                (zflag[0][1] * math.sin(x[2]) / math.sec(x[2])) / x[2]**2
            df[0][2] = 0
            df[0][3] = 0
            df[0][4] = math.sin(x[2]) + \
                (zflag[1][1] * math.cos(x[2])) / math.sec(x[2])
            df[0][5] = 0
            
            df[1][0] = 0
            df[1][1] = 0
            df[1][2] = 0
            df[1][3] = 0
            df[1][4] = 0
            df[1][5] = 0

def point_to_point(
        sys, timepts, z0, zf, cost=None, constraints=None, basis=None):
    init_constraint = ntg.flag_equality_constraint(sys, z0)
    term_constraint = ntg.flag_equality_constraint(sys, zf)
    return ntg.solve_flat_ocp(
        sys, timepts, basis,
        initial_constraints=init_constraint,
        final_constraints=term_constraint,
        trajectory_cost=cost, trajectory_constraints=constraints)
    

#
# Benchmarks
#

def time_point_to_point(basis_name, basis_size):
    assert ntg_installed
    basis = get_basis(basis_name, basis_size)

    # Find trajectory between initial and final conditions
    result = point_to_point(vehicle, timepts, z0, zf, basis=basis)
    traj = result.systraj

    # Verify that the trajectory computation is correct
    ztraj = traj.eval([0, Tf])
    np.testing.assert_array_almost_equal(z0, ztraj[:, :, 0])
    np.testing.assert_array_almost_equal(zf, ztraj[:, :, -1])

time_point_to_point.params = basis_params
time_point_to_point.param_names = basis_param_names


def time_point_to_point_with_cost(basis_name, basis_size):
    assert ntg_installed
    basis = get_basis(basis_name, basis_size)

    # Define cost and constraints
    traj_cost = quadratic_cost(
        vehicle, None, np.diag([0.1, 1]), u0=uf)
    constraints = None
    # constraints = [
    #     input_range_constraint(vehicle, [8, -0.1], [12, 0.1]) ]

    systraj, cost, inform = point_to_point(
        vehicle, timepts, z0, zf, 
        cost=traj_cost, constraints=constraints, basis=basis,
    )

    # Verify that the trajectory computation is correct
    ztraj = systraj.eval([0, Tf])
    np.testing.assert_array_almost_equal(z0, ztraj[:, :, 0])
    np.testing.assert_array_almost_equal(zf, ztraj[:, :, -1])

time_point_to_point_with_cost.params = basis_params
time_point_to_point_with_cost.param_names = basis_param_names


def time_solve_flat_ocp_terminal_cost(method, basis_name, basis_size):
    assert ntg_installed
    basis = get_basis(basis_name, basis_size)

    # Define cost and constraints
    traj_cost = quadratic_cost(
        vehicle, None, np.diag([0.1, 1]), u0=uf)
    term_cost = quadratic_cost(
        vehicle, np.diag([1e3, 1e3, 1e3]), None, x0=xf)
    # constraints = [
    #     input_range_constraint(vehicle, [8, -0.1], [12, 0.1]) ]
    constraints = None

    # Initial guess = straight line
    initial_guess = np.array(
        [x0[i] + (xf[i] - x0[i]) * timepts/Tf for i in (0, 1)])

    traj = ntg.solve_flat_ocp(
        vehicle, timepts, z0, basis=basis, initial_guess=initial_guess,
        trajectory_cost=traj_cost, constraints=constraints,
        terminal_cost=term_cost,
    )

    # Verify that the trajectory computation is correct
    x, u = traj.eval([0, Tf])
    np.testing.assert_array_almost_equal(x0, x[:, 0])
    np.testing.assert_array_almost_equal(xf, x[:, -1], decimal=2)

time_solve_flat_ocp_terminal_cost.params = tuple(
    [['ntg']] + list(basis_params))
time_solve_flat_ocp_terminal_cost.param_names = tuple(
    ['method'] + basis_param_names)
