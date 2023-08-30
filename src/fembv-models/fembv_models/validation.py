"""Provides helper routines for validating user inputs."""

# License: MIT

from __future__ import absolute_import, division

import numbers

import numpy as np


INTEGER_TYPES = (numbers.Integral, np.integer)


def is_integer(x):
    """Check if x is an integer."""
    return isinstance(x, INTEGER_TYPES)


def is_scalar(x):
    """Check is x is a scalar value."""
    return np.ndim(x) == 0 or np.isscalar(x)


def check_unit_axis_sums(x, whom, axis=0):
    """Check sums along array axis are one."""
    axis_sums = x.sum(axis=axis)

    if not np.all(np.isclose(axis_sums, 1)):
        raise ValueError(
            'Array with incorrect axis sums passed to %s. '
            'Expected sums along axis %d to be 1.'
            % (whom, axis))


def check_array_shape(x, shape, whom):
    """Check array has the desired shape."""
    if x.shape != shape:
        raise ValueError(
            'Array with wrong shape passed to %s. '
            'Expected %s, but got %s' % (whom, shape, x.shape))
