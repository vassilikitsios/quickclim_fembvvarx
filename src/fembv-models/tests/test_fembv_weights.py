"""Provides test routines for FEM-BV weights solver."""

# License: MIT

from __future__ import absolute_import, division

import numpy as np
from sklearn.utils import check_random_state

from fembv_models import FEMBVWeights, right_stochastic_matrix


def test_weights_equality_constraints_with_no_max_tv_norm():
    """Test equality constraints match expected form."""
    random_seed = 0
    random_state = check_random_state(random_seed)

    n_samples = 4
    n_components = 3
    max_tv_norm = None

    initial_weights = right_stochastic_matrix((n_samples, n_components),
                                              random_state=random_state)

    weights_solver = FEMBVWeights(initial_weights, max_tv_norm=max_tv_norm)

    a_eq = np.array(weights_solver._a_eq.todense())
    b_eq = np.array(weights_solver._b_eq)

    expected_a_eq = np.array(
        [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    expected_b_eq = np.array([[1.0], [1.0], [1.0], [1.0]])

    assert np.allclose(a_eq, expected_a_eq)
    assert np.allclose(b_eq, expected_b_eq)


def test_weights_upper_bound_constraints_with_no_max_tv_norm():
    """Test inequality constraints match expected form."""
    random_seed = 0
    random_state = check_random_state(random_seed)

    n_samples = 3
    n_components = 3
    max_tv_norm = None

    initial_weights = right_stochastic_matrix((n_samples, n_components),
                                              random_state=random_state)

    weights_solver = FEMBVWeights(initial_weights, max_tv_norm=max_tv_norm)

    a_ub = np.array(weights_solver._a_ub.todense())
    b_ub = np.array(weights_solver._b_ub)

    expected_a_ub = -np.eye(n_components * n_samples)
    expected_b_ub = np.zeros((n_components * n_samples,))

    assert np.allclose(a_ub, expected_a_ub)
    assert np.allclose(b_ub, expected_b_ub)


def test_weights_equality_constraints_with_max_tv_norm():
    """Test equality constraints match expected form."""
    random_seed = 0
    random_state = check_random_state(random_seed)

    n_samples = 4
    n_components = 3
    max_tv_norm = 5

    initial_weights = right_stochastic_matrix((n_samples, n_components),
                                              random_state=random_state)

    weights_solver = FEMBVWeights(initial_weights, max_tv_norm=max_tv_norm)

    a_eq = np.array(weights_solver._a_eq.todense())
    b_eq = np.array(weights_solver._b_eq)

    n_parameters = n_components * (2 * n_samples - 1)
    expected_a_eq = np.zeros((n_samples, n_parameters), dtype='f8')
    expected_a_eq[0, 0:3] = 1.0
    expected_a_eq[1, 3:6] = 1.0
    expected_a_eq[2, 6:9] = 1.0
    expected_a_eq[3, 9:12] = 1.0

    expected_b_eq = np.array([[1.0], [1.0], [1.0], [1.0]])

    assert np.allclose(a_eq, expected_a_eq)
    assert np.allclose(b_eq, expected_b_eq)


def test_weights_upper_bound_constraints_with_max_tv_norm():
    """Test inequality constraints match expected form."""
    random_seed = 0
    random_state = check_random_state(random_seed)

    n_samples = 3
    n_components = 2
    max_tv_norm = 5

    initial_weights = right_stochastic_matrix((n_samples, n_components),
                                              random_state=random_state)

    weights_solver = FEMBVWeights(initial_weights, max_tv_norm=max_tv_norm)

    a_ub = np.array(weights_solver._a_ub.todense())
    b_ub = np.array(weights_solver._b_ub)

    n_parameters = n_components * (2 * n_samples - 1)
    n_constraints = (n_parameters + 2 * n_components * (n_samples - 1) +
                     n_components)

    assert a_ub.shape == (n_constraints, n_parameters)
    assert b_ub.shape == (n_constraints,)

    expected_a_ub = np.zeros((n_constraints, n_parameters))
    expected_a_ub[:n_parameters, :n_parameters] = -np.eye(n_parameters)

    expected_a_ub = np.array(
        [[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
         [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
         [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
         [0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
         [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
         [1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]])

    expected_b_ub = np.zeros((n_constraints,), dtype='f8')
    expected_b_ub[-n_components:] = max_tv_norm

    assert np.allclose(a_ub, expected_a_ub)
    assert np.allclose(b_ub, expected_b_ub)
