"""Provides helper routines for constructing stochastic matrices."""

# License: MIT

from __future__ import absolute_import, division

import numpy as np
from sklearn.utils import check_random_state


def _uniform_stochastic_matrix(shape, random_state=None, axis=0):
    """Return random stochastic matrix."""
    rng = check_random_state(random_state)

    m = rng.uniform(size=shape)
    axis_sums = np.sum(m, axis=axis)

    if axis in (0, -2):
        return m / axis_sums[np.newaxis, :]

    if axis in (1, -1):
        return m / axis_sums[:, np.newaxis]

    raise ValueError(
        'axis %d is out of bounds for array of dimension %d' %
        (axis, m.ndim))


def left_stochastic_matrix(shape, random_state=None):
    """Return random matrix with unit column sums."""
    return _uniform_stochastic_matrix(shape, random_state=random_state, axis=0)


def right_stochastic_matrix(shape, random_state=None):
    """Return random matrix with unit row sums."""
    return _uniform_stochastic_matrix(shape, random_state=random_state, axis=1)
