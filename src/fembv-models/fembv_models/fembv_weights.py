"""Provides class for calculating FEM-BV weights."""

# License: MIT

from __future__ import absolute_import, division

import numbers
import warnings

import cvxpy as cp
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils import check_array

from .validation import check_unit_axis_sums


INTEGER_TYPES = (numbers.Integral, np.integer)


def _check_init_weights(weights, whom):
    weights = check_array(weights)
    check_unit_axis_sums(weights, whom, axis=1)


class FEMBVWeights:
    """Solver for FEM-BV component weights."""

    def __init__(self, initial_weights, max_tv_norm=None,
                 weights_solver_kwargs=None):

        _check_init_weights(initial_weights, 'FEMBVWeights')

        self.n_samples, self.n_components = initial_weights.shape
        self.max_tv_norm = max_tv_norm
        self.weights = initial_weights.copy()

        if weights_solver_kwargs is None:
            self.weights_solver_kwargs = {}
        else:
            self.weights_solver_kwargs = weights_solver_kwargs

        if self._has_max_tv_norm_constraint():
            # Auxiliary variables \eta_t used to enforce TV-norm constraint.
            self.auxiliary_variables = np.zeros(
                (self.n_components * (self.n_samples - 1),))
        else:
            self.auxiliary_variables = None

        # Full parameter vector, including any auxiliary optimization variables,
        # structured such that _params = (vec(\Gamma^T), vec(\Eta^T)),
        # where \Gamma is the n_samples x n_components matrix of weights,
        # and \Eta is the (n_samples - 1) x n_components matrix of auxiliary
        # variables
        self._params = np.empty((self._get_number_of_weight_parameters(),))
        self._params[:self.n_samples * self.n_components] = self.weights.ravel()

        # Vector of objective function coefficients
        self._c = np.zeros((self._get_number_of_weight_parameters(),))

        # Matrices for equality and upper bound constraints
        self._a_eq = None
        self._b_eq = None
        self._a_ub = None
        self._b_ub = None
        self._initialize_constraints()

    def _has_max_tv_norm_constraint(self):
        return self.max_tv_norm is not None and self.max_tv_norm >= 0

    def _get_number_of_weight_parameters(self):
        n_weight_parameters = self.n_components * self.n_samples

        if self._has_max_tv_norm_constraint():
            n_weight_parameters += self.n_components * (self.n_samples - 1)

        return n_weight_parameters

    def _initialize_constraints(self):
        n_weight_parameters = self._get_number_of_weight_parameters()

        # Note, if the number of components is 1, then the solution for the
        # weights is trivial, hence need only build constraint matrices if
        # number of components is greater than 1.
        if self.n_components > 1:

            # First n_components x n_samples elements of parameter vector
            # are vec(\Gamma^T).
            eq_nnz_rows = np.repeat(np.arange(self.n_samples), self.n_components)
            eq_nnz_cols = np.arange(self.n_samples * self.n_components)

            # Constraints imposing unit sums of weights at each time.
            self._a_eq = csr_matrix(
                (np.ones((self.n_samples * self.n_components,),
                         dtype=self.weights.dtype),
                 (eq_nnz_rows, eq_nnz_cols)),
                shape=(self.n_samples, n_weight_parameters))
            self._b_eq = np.ones((self.n_samples,), dtype=self.weights.dtype)

            # Number of non-negativity constraints.
            n_non_neg_constraints = n_weight_parameters
            n_ub_constraints = n_non_neg_constraints

            # Non-negativity constraints on weight parameters - each weight
            # and any auxiliary variables must be non-negative, giving diagonal
            # constraint matrix.
            nnz_coeffs = np.full((n_weight_parameters,), -1.0)
            ub_nnz_rows = np.arange(n_weight_parameters)
            ub_nnz_cols = np.arange(n_weight_parameters)
            ub_values = np.zeros((n_weight_parameters,), dtype=self.weights.dtype)

            if self._has_max_tv_norm_constraint():
                # Number of auxiliary constraints.
                n_aux_constraints = 2 * self.n_components * (self.n_samples - 1)

                # The auxiliary variables provide the upper bounds
                # (\gamma_{t + 1})_i - (\gamma_t)_i \leq (\eta_t)_i
                # and -(\gamma_{t + 1})_i + (\gamma_t)_i \leq (\eta_t)_i,
                # each involving three non-zero elements.
                nnz_aux_rows = np.repeat(
                    np.arange(n_ub_constraints,
                              n_ub_constraints + n_aux_constraints), 3)

                # Since the parameters vector is (vec(\Gamma^T), vec(\Eta^T)),
                # the variables in each constraint are at columns
                # (t + 1) x n_components + i, t x n_components + i,
                # and n_components x n_samples + t x n_components + i.
                nnz_aux_cols = np.tile(np.array(
                    [[(t + 1) * self.n_components + i,
                      t * self.n_components + i,
                      self.n_components * self.n_samples + t * self.n_components + i]
                     for t in range(self.n_samples - 1)
                     for i in range(self.n_components)]).ravel(), 2)

                # Ordering of constraints is all positive branches of the
                # absolute value first, i.e., the constraints
                # (\gamma_{t + 1})_i - (\gamma_t)_i \leq (\eta_t)_i,
                # followed by the reverse sign constraint.
                nnz_aux_coeffs = np.concatenate(
                    [np.tile(np.array([1.0, -1.0, -1.0]),
                             self.n_components * (self.n_samples - 1)),
                     np.tile(np.array([-1.0, 1.0, -1.0]),
                             self.n_components * (self.n_samples - 1))])

                n_ub_constraints += n_aux_constraints
                ub_values = np.concatenate(
                    [ub_values,
                     np.zeros((2 * self.n_components * (self.n_samples - 1),),
                              dtype=self.weights.dtype)])

                # Number of total variation norm constraints - one for each
                # component.
                n_tv_constraints = self.n_components

                nnz_tv_rows = np.repeat(
                    np.arange(n_ub_constraints,
                              n_ub_constraints + n_tv_constraints),
                    self.n_samples - 1)

                # The TV-norm constraints take the form \sum_t \eta_i(t) \leq C,
                # so columns in each constraint appear separated by strides
                # of n_components length.
                nnz_tv_cols = np.concatenate(
                    [np.array(
                        [self.n_components * self.n_samples + t * self.n_components + i
                         for t in range(self.n_samples - 1)])
                     for i in range(self.n_components)])
                nnz_tv_coeffs = np.ones(
                    (self.n_components * (self.n_samples - 1),),
                    dtype=self.weights.dtype)

                n_ub_constraints += n_tv_constraints

                ub_values = np.concatenate(
                    [ub_values,
                     np.full((self.n_components,), self.max_tv_norm)])

                nnz_coeffs = np.concatenate(
                    [nnz_coeffs, nnz_aux_coeffs, nnz_tv_coeffs])
                ub_nnz_rows = np.concatenate(
                    [ub_nnz_rows, nnz_aux_rows, nnz_tv_rows])
                ub_nnz_cols = np.concatenate(
                    [ub_nnz_cols, nnz_aux_cols, nnz_tv_cols])

            self._a_ub = csr_matrix(
                (nnz_coeffs, (ub_nnz_rows, ub_nnz_cols)),
                shape=(n_ub_constraints, n_weight_parameters))
            self._b_ub = np.reshape(ub_values, (n_ub_constraints,))

    def update_weights(self, distance_matrix):
        """Update FEM-BV weights."""
        if self.n_components == 1:
            return

        if distance_matrix.shape != (self.n_samples, self.n_components):
            raise ValueError(
                "Distance matrix has incorrect shape in update_weights: "
                "expected (%d, %d) but got %r" %
                (self.n_samples, self.n_components, distance_matrix.shape))

        n_weight_parameters = self._get_number_of_weight_parameters()
        gamma_vec = cp.Variable(n_weight_parameters)

        self._c[:self.weights.size] = distance_matrix.ravel()

        constraints = [self._a_eq @ gamma_vec == self._b_eq,
                       self._a_ub @ gamma_vec <= self._b_ub]

        problem = cp.Problem(cp.Minimize(self._c.T @ gamma_vec), constraints)

        problem.solve(**self.weights_solver_kwargs)

        if problem.status in ['infeasible', 'unbounded']:
            warnings.warn('Updating weights failed', UserWarning)
        else:
            sol = problem.variables()[0].value

            self.weights = np.clip(np.reshape(
                sol[:self.n_components * self.n_samples],
                (self.n_samples, self.n_components)), 0.0, 1.0)

            if self._has_max_tv_norm_constraint():
                self.auxiliary_variables = np.reshape(
                    sol[self.n_components * self.n_samples:],
                    (self.n_samples - 1, self.n_components))
