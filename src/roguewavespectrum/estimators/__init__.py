"""
Contents: Spectral estimators

Copyright (C) 2022
Sofar Ocean Technologies

Authors: Pieter Bart Smit
======================

Spectral estimators that can be used to create a 2D spectrum from buoy
observations

Classes:

- `None

Functions:

- `mem`, maximum entrophy method
- `mem2`, ...

How To Use This Module
======================
(See the individual functions for details.)
"""

from .estimate import estimate_directional_spectrum_from_moments, estimate_directional_distribution
from .mem2 import (
    mem2_newton_solver,
    mem2_directional_distribution,
    mem2_jacobian,
    initial_value,
    moment_constraints,
    get_direction_increment
)

