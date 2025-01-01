# flatsys/__init__.py: flat systems package initialization file
#
# Author: Richard M. Murray
# Date: 1 Jul 2019

r"""Flat systems sub-package.

This sub-package contains a set of classes and functions to compute
trajectories for differentially flat systems.

A differentially flat system is defined by creating an object using the
`flatsys.FlatSystem` class, which has member functions for mapping the
system state and input into and out of flat coordinates.  The
`flatsys.point_to_point` function can be used to create a trajectory
between two endpoints, written in terms of a set of basis functions defined
using the `flatsys.BasisFamily` class.  The resulting trajectory is return
as a `flatsys.SystemTrajectory` object and can be evaluated using the
`flatsys.SystemTrajectory.eval` member function.  Alternatively, the
`flatsys.solve_flat_ocp` function can be used to solve an optimal control
problem with trajectory and final costs or constraints.

The docstring examples assume that the following import commands::

  >>> import numpy as np
  >>> import control as ct
  >>> import control.flatsys as fs

"""

# Basis function families
from .basis import BasisFamily
from .poly import PolyFamily
from .bezier import BezierFamily
from .bspline import BSplineFamily

# Classes
from .systraj import SystemTrajectory
from .flatsys import FlatSystem, flatsys
from .linflat import LinearFlatSystem

# Package functions
from .flatsys import point_to_point, solve_flat_ocp
