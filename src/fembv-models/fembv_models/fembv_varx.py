"""Provides routines for performing FEM-BV-VARX fits."""

# License: MIT

from __future__ import absolute_import, division

import numbers
import time
import warnings

import numpy as np
import scipy.linalg as sl
import sklearn.cluster as cluster
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array, check_random_state

from .fembv_weights import FEMBVWeights
from .validation import check_array_shape


INTEGER_TYPES = (numbers.Integral, np.integer)


def _check_X(X, n_components=None, n_features=None, ensure_min_samples=1):
    """Check the input data X.

    See https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/mixture/_base.py .

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)

    n_components : integer

    Returns
    -------
    X : array, shape (n_samples, n_features)
    """
    X = check_array(X, dtype=[np.float64, np.float32],
                    ensure_min_samples=ensure_min_samples)
    if n_components is not None and X.shape[0] < n_components:
        raise ValueError('Expected n_samples >= n_components '
                         'but got n_components = %d, n_samples = %d'
                         % (n_components, X.shape[0]))
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError("Expected the input data X have %d features, "
                         "but got %d features"
                         % (n_features, X.shape[1]))
    return X


def _check_orders(orders, n_components):
    """Check the user-provided VAR orders.

    Parameters
    ----------
    orders : array-like, shape (n_components,)
        The orders of each VARX component.

    n_components : integer
        The number of components.

    Returns
    -------
    orders : array, shape (n_components,)
    """
    orders = check_array(orders, dtype=[np.int32], ensure_2d=False)
    check_array_shape(orders, (n_components,), 'orders')

    if any(np.less(orders, 0)):
        raise ValueError("The parameter 'orders' should be non-negative, "
                         "but got min value %d" % (np.min(orders)))

    return orders


def _check_weights(weights, n_samples, n_components):
    """Check the user-provided weights.

    Parameters
    ----------
    weights : array-like, shape (n_samples, n_components)
        The weights associated with each component for each observation.

    n_samples : integer
        The number of modelled observations.

    n_components : integer
        The number of components.

    Returns
    -------
    weights : array, shape (n_samples, n_components)
    """
    weights = check_array(weights, dtype=[np.float64, np.float32],
                          ensure_2d=False)
    check_array_shape(weights, (n_samples, n_components), 'weights')

    # check range
    if (any(np.less(weights, 0.)) or
            any(np.greater(weights, 1.))):
        raise ValueError("The parameter 'weights' should be in the range "
                         "[0, 1], but got max value %.5f, min value %.5f"
                         % (np.min(weights), np.max(weights)))

    # check normalization
    if not np.allclose(np.abs(1. - np.sum(weights, axis=1)), 0.):
        raise ValueError("The parameter 'weights' should be normalized, "
                         "but got sum(weights, axis=1) = %.5f" %
                         np.sum(weights, axis=1))
    return weights


def _check_means(means, n_components, n_features):
    """Check the user-provided means.

    Parameters
    ----------
    means : array-like, shape (n_components, n_features)
        The means of the mixture components.

    n_components : integer
        The number of components.

    n_features : integer
        The number of features.

    Returns
    -------
    means : array, shape (n_components, n_features)
    """
    means = check_array(means, dtype=[np.float64, np.float32])

    check_array_shape(means, (n_components, n_features), 'means')

    return means


def _check_endog_coefs(endog_coefs, n_components, n_features, max_order):
    """Check the user-provided endog_coefs.

    Parameters
    ----------
    endog_coefs : array-like, shape (n_components, max_order, n_features, n_features)
        The coefficients of the endogeneous variables.

    n_components : integer
        The number of components.

    n_features : integer
        The number of features.

    max_order : integer
        The maximum AR model order.

    Returns
    -------
    endog_coefs : array, shape (n_components, max_order, n_features, n_features)
    """
    endog_coefs = check_array(endog_coefs, dtype=[np.float64, np.float32],
                              ensure_2d=False, allow_nd=True)

    check_array_shape(endog_coefs, (n_components, max_order, n_features, n_features),
                      'endog_coefs')

    return endog_coefs


def _check_exog_coefs(exog_coefs, n_components, n_features, n_exog):
    """Check the user-provided exog_coefs.

    Parameters
    ----------
    exog_coefs : array-like, shape (n_components, n_features, n_exog)
        The coefficients of the exogeneous variables.

    n_components : integer
        The number of components.

    n_features : integer
        The number of features.

    n_exog : integer
        The number of exogeneous variables.

    Returns
    -------
    exog_coefs : array, shape (n_components, n_features, n_exog)
    """
    exog_coefs = check_array(exog_coefs, dtype=[np.float64, np.float32],
                             ensure_2d=False, allow_nd=True)

    check_array_shape(exog_coefs, (n_components, n_features, n_exog),
                      'exog_coefs')

    return exog_coefs


def _check_precision_positivity(precision):
    """Check a precision vector is positive-definite."""
    if np.any(np.less_equal(precision, 0.0)):
        raise ValueError('precision should be positive')


def _check_precision_matrix(precision):
    """Check a precision matrix is symmetric and positive-definite."""
    if not (np.allclose(precision, precision.T) and
            np.all(sl.eigvalsh(precision) > 0.)):
        raise ValueError('precision should be symmetric, '
                         'positive-definite')


def _check_precisions_full(precisions):
    """Check the precision matrices are symmetric and positive-definite."""
    for prec in precisions:
        _check_precision_matrix(prec)


def _check_precisions(precisions, n_components, n_features):
    """Check the user-provided precisions.

    Parameters
    ----------
    precisions : array-like, shape (n_components, n_features, n_features)

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    precisions : array
    """
    precisions = check_array(precisions, dtype=[np.float64, np.float32],
                             ensure_2d=False, allow_nd=True)

    check_array_shape(precisions, (n_components, n_features, n_features),
                      'precision')

    _check_precisions_full(precisions)

    return precisions


def _compute_precision_cholesky(covariances):
    """Compute the Cholesky decomposition of the precisions.

    Parameters
    ----------
    covariances : array-like, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.

    Returns
    -------
    precisions_cholesky : array-like
        The Cholesky decomposition of sample precisions of the current
        components.
    """
    estimate_precision_error_message = (
        "Fitting the FEM-BV-VARX model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar.")

    n_components, n_features, _ = covariances.shape
    precisions_chol = np.empty((n_components, n_features, n_features))
    for k, covariance in enumerate(covariances):
        try:
            cov_chol = sl.cholesky(covariance, lower=True)
        except sl.LinAlgError as chol_error:
            raise ValueError(estimate_precision_error_message) from chol_error

        precisions_chol[k] = sl.solve_triangular(cov_chol,
                                                 np.eye(n_features),
                                                 lower=True).T

    return precisions_chol


def _compute_covariances(precisions_chol):
    """Compute covariancess from Cholesky decomposition of the precision matrices.

    Parameters
    ----------
    precisions_chol : array-like, shape (n_components, n_features, n_features)
        The Cholesky decomposition of the sample precisions.

    Returns
    -------
    covariances : array-like
        The covariance matrices corresponding to the given precision
        matrices.
    """
    n_components, n_features, _ = precisions_chol.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k, prec_chol in enumerate(precisions_chol):
        cov_chol = sl.solve_triangular(prec_chol,
                                       np.eye(n_features), lower=True).T
        covariances[k] = np.dot(cov_chol, cov_chol.T)

    return covariances


class FEMBVVARX:
    """Fit FEM-BV-VARX model.

    Parameters
    ----------
    n_components : integer, default: 2
        The number of mixture components.

    orders : integer or array-like, shape (n_components,)
        If an integer, the AR order of each component. If an array,
        each element should be the AR order of the corresponding component.

    state_length : float, optional
        The average length of time (in units of the input data time step)
        spent consecutively in a single component. Internally converted to
        a bound on the TV norm for all components.

    presample_length : integer, optional
        If given, the number of observations at the start of the sample
        to hold out as presample values.

    tol : float, default: 1e-3
        The convergence threshold. Optimization iterations will stop when the
        change in the cost function is below this threshold.

    reg_covar : float, default: 1e-6
        Non-negative regularization added to the diagonal of the covariance.
        Ensures that all covariance matrices are positive definite.

    max_iter : integer, default: 100
        Maximum number of iterations in optimization.

    n_init : integer, default: 1
        The number of initializations to perform. The best results are kept.

    init_params : None | 'random'
        The method used to initialize the model. If None, defaults to 'random'.

    weights_init : array-like, shape (n_samples, n_components), optional
        The user-provided initial weights, defaults to None. If None, the
        weights are initialized using the init_params method.

    means_init : array-like, shape (n_components, n_features), optional
        The user-provided initial means, defaults to None. If None, the
        means are initialized using the init_params method.

    endog_coef_init : array-like, shape (n_components, max_order, n_features, n_features), optional
        The user-provided initial coefficients for the endogeneous variables.
        If None, the coefficients are initialized using the init_params method.

    exog_coef_init : array-like, shape (n_components, n_features, n_exog), optional
        The user-provided initial coefficients for the exogeneous variables,
        if present. If None and exogeneous variables are provided to the fit
        method, the coefficients are initialized using the init_params method.

    precisions_init : array-like, shape (n_components, n_features), optional
        The user-provided initial precisions (inverses of the covariance
        matrices). If None, the precisions are initialized using the init_params
        method.

    random_state : integer, RandomState or None
        If an integer, random_state is the seed used by the
        random number generator. If a RandomState instance,
        random_state is the random number generator. If None,
        the random number generator is the RandomState instance
        used by `np.random`.

    warm_start : bool, default: False
        If `warm_start` is True, the solution of the last fitting is used
        as initialization for the next call of fit().

    verbose : integer, default: 0
        Enable verbose output.

    verbose_interval : integer, default: 1
        The number of iterations between each verbose print message.

    Attributes
    ----------
    max_order_ : integer
        The maximum AR order of the components.

    weights_ : array-like, shape (n_samples, n_components)
        The weights of each of the components.

    means_ : array-like, shape (n_components, n_features)
        The means of each of the  components.

    endog_coef_ : array-like, shape (n_components, max_order_, n_features, n_features)
        The coefficient matrices of the lagged endogeneous variables
        for each component.

    exog_coef_ : None or array-like, shape (n_components, n_features, n_exog)
        If exogeneous predictors are included, the coefficient matrices
        of the exogeneous variables for each component.

    covariances_ : array-like, shape (n_components, n_features)
        The covariance of each component.

    precisions_ : array-like, shape (n_components, n_features)
        The inverse covariance of each  component.

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    n_iter_ : integer
        The number of iterations required to reach convergence.

    cost_ : float
        The optimal cost found by minimizing the loss function.

    log_likelihood_ : float
        The log-likelihood of the data under the mixture model.

    rmse_ : float
        The RMSE for the data weighted by the component affiliations.

    Examples
    --------
    import numpy as np
    X = np.random.rand((100, 5))
    from fembv_models import FEMBVVARX
    model = FEMBVVARX(n_components=2, random_state=0).fit(X)
    """

    def __init__(self, n_components=2, orders=0, state_length=None,
                 presample_length=None, tol=1e-4, reg_covar=1e-6, max_iter=100,
                 n_init=1, init_params=None, weights_init=None,
                 means_init=None, endog_coef_init=None, exog_coef_init=None,
                 precisions_init=None, random_state=None, warm_start=False,
                 verbose=0, verbose_interval=1, loss='least_squares',
                 weights_solver_kwargs=None,
                 require_monotonic_cost_decrease=True):

        self.n_components = n_components
        self.orders = orders
        self.state_length = state_length
        self.presample_length = presample_length
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.weights_init = weights_init
        self.means_init = means_init
        self.endog_coef_init = endog_coef_init
        self.exog_coef_init = exog_coef_init
        self.precisions_init = precisions_init
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self.loss = loss
        self.weights_solver_kwargs = weights_solver_kwargs
        self.require_monotonic_cost_decrease = require_monotonic_cost_decrease

    def fit(self, X, y=None, exog=None):
        """Estimate model parameters by block coordinate descent.

        The method fits the model ``n_init`` times and sets the parameters with
        which the model has the lowest cost. Within each
        trial, the method iterates between optimizing the weights and the
        model parameters for ``max_iter`` times until the change in the cost
        is less than ``tol``, otherwise, a ``ConvergenceWarning`` is raised.
        If ``warm_start`` is ``True``, then ``n_init`` is ignored and a single
        initialization is performed upon the first call. Upon consecutive
        calls, training starts where it left off.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        exog : array-like, shape (n_samples, n_exog), optional
            List of n_exog-dimensional exogeneous variables. Each row
            corresponds to a single data point.

        Returns
        -------
        self
        """
        self.fit_predict(X, exog=exog)
        return self

    def fit_predict(self, X, y=None, exog=None):
        """Estimate model parameters using X and predict the labels for X.

        The method fits the model ``n_init`` times and sets the parameters with
        which the model has the lowest cost. Within each
        trial, the method iterates between optimizing the weights and the
        model parameters for ``max_iter`` times until the change in the cost
        is less than ``tol``, otherwise, a ``ConvergenceWarning`` is raised.
        The affiliation for each datapoint is returned after fitting.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        exog : array-like, shape (n_samples, n_exog), optional
            List of n_exog-dimensional exogeneous variables. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        # need to check model orders are valid first so as to know how many
        # samples are required
        self._check_initial_parameters(X, exog=exog)

        if self.presample_length is None:
            self.presample_length = self.max_order_
        else:
            if self.presample_length < self.max_order_:
                raise ValueError(
                    'Presample length must be at least as long as the maximum order '
                    '(got presample_length=%d, max_order=%d)' %
                    (self.presample_length, self.max_order_))

            self.presample_length = max(self.presample_length, self.max_order_)

        min_samples = max(self.presample_length + 1, 2)
        X = _check_X(X, self.n_components, ensure_min_samples=min_samples)
        if exog is not None:
            exog = _check_X(exog, self.n_components,
                            ensure_min_samples=min_samples)

        self._check_parameters(X, exog=exog)

        self._stack_predictors(X, exog=exog)

        # calculate bound on TV norm corresponding to the required average
        # state length
        if self.state_length is None or self.state_length == 0:
            max_tv_norm = None
        else:
            n_samples = X.shape[0]
            max_tv_norm = max(0.0, (n_samples / (1.0 * self.state_length) - 1.0))

        self._max_tv_norm = max_tv_norm

        # if we enable warm_start, we will have a unique initialization
        do_init = not(self.warm_start and hasattr(self, 'converged_'))
        n_init = self.n_init if do_init else 1

        min_cost = np.infty
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        for init in range(n_init):
            self._init_start_time = time.perf_counter()
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize(X, random_state, exog=exog)

            self._fembv_weights = FEMBVWeights(
                self.weights_, max_tv_norm=self._max_tv_norm,
                weights_solver_kwargs=self.weights_solver_kwargs)

            cost = (np.infty if do_init else self.cost_)
            converged = False

            for n_iter in range(self.max_iter):

                self._iter_start_time = time.perf_counter()

                prev_cost = cost

                self._update_weights(X)
                self._update_parameters(X, self.weights_, exog=exog)

                cost = self._compute_cost(X, self.weights_)

                change = cost - prev_cost

                self._iter_end_time = time.perf_counter()
                self._print_verbose_msg_iter_end(n_iter, cost, change)

                cost_increased = abs(change) > self.tol and change > 0
                if cost_increased and self.require_monotonic_cost_decrease:
                    warnings.warn(
                        'Cost increased during iteration in initialization %d' %
                        (init + 1), UserWarning)
                    break

                if abs(change) < self.tol:
                    converged = True
                    break

            self._init_end_time = time.perf_counter()
            self._print_verbose_msg_init_end(cost, converged)

            if cost < min_cost:
                min_cost = cost
                best_weights = self.weights_
                best_params = self._get_parameters()
                best_n_iter = n_iter

            if not converged:
                warnings.warn('Initialization %d did not converge. '
                              'Try different init parameters, '
                              'or increase max_iter, tol '
                              'or check for degenerate data.'
                              % (init + 1), ConvergenceWarning)
            else:
                # at least one initialization converged
                self.converged_ = converged

        self.weights_ = best_weights
        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.cost_ = min_cost
        self.log_likelihood_ = self._log_likelihood(X, exog=exog)
        self.rmse_ = self._rmse(X, exog=exog)

        return self.weights_

    def predict(self, X, y=None, exog=None):
        """Predict the labels for X.

        The method calculates the affiliation sequence that minimizes
        the cost function for the given data and previously fitted parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        exog : array-like, shape (n_samples, n_exog), optional
            List of n_exog-dimensional exogeneous variables. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.

        cost : float
            Value of the cost function for the given label assignment.

        log_likelihood : float
            Bound on the value of the log-likelihood function for the given
            label assignment.
        """
        if not hasattr(self, 'means_'):
            raise ValueError(
                'Model must be fitted before label prediction can be performed')

        # need to check model orders are valid first so as to know how many
        # samples are required
        self._check_initial_parameters(X, exog=exog)

        if self.presample_length is None:
            self.presample_length = self.max_order_
        else:
            if self.presample_length < self.max_order_:
                raise ValueError(
                    'Presample length must be at least as long as the maximum order '
                    '(got presample_length=%d, max_order=%d)' %
                    (self.presample_length, self.max_order_))

            self.presample_length = max(self.presample_length, self.max_order_)

        min_samples = max(self.presample_length + 1, 2)
        X = _check_X(X, self.n_components, ensure_min_samples=min_samples)
        if exog is not None:
            exog = _check_X(exog, self.n_components,
                            ensure_min_samples=min_samples)

        n_samples = X.shape[0]

        self._stack_predictors(X, exog=exog)
        self._set_flattened_parameters(
            self.means_, endog_coef=self.endog_coef_, exog_coef=self.exog_coef_)

        # calculate bound on TV norm corresponding to the required average
        # state length
        old_max_tv_norm = self._max_tv_norm
        if self.state_length is None or self.state_length == 0:
            max_tv_norm = None
        else:
            max_tv_norm = max(0.0, (n_samples / (1.0 * self.state_length) - 1.0))

        self._max_tv_norm = max_tv_norm

        n_init = self.n_init

        min_cost = np.infty
        best_weights = None

        random_state = check_random_state(self.random_state)

        if self.init_params is None:
            self.init_params = 'random'

        for init in range(n_init):
            self._init_start_time = time.perf_counter()
            self._print_verbose_msg_init_beg(init)

            if self.init_params == 'kmeans':

                weights = np.zeros(
                    (n_samples - self.presample_length, self.n_components))
                labels = cluster.KMeans(n_clusters=self.n_components, n_init=1,
                                        random_state=random_state).fit(X).labels_
                weights[np.arange(n_samples - self.presample_length),
                        labels[self.presample_length:]] = 1

            elif self.init_params == 'random':

                weights = random_state.rand(
                    n_samples - self.presample_length, self.n_components)
                weights /= weights.sum(axis=1)[:, np.newaxis]

            else:
                raise ValueError("Unimplemented initialization method '%s'"
                                 % self.init_params)

            self.weights_ = weights.copy()

            self._fembv_weights = FEMBVWeights(
                self.weights_, max_tv_norm=self._max_tv_norm,
                weights_solver_kwargs=self.weights_solver_kwargs)

            cost = np.infty
            converged = False

            for n_iter in range(self.max_iter):

                self._iter_start_time = time.perf_counter()

                prev_cost = cost

                self._update_weights(X)

                cost = self._compute_cost(X, self.weights_)

                change = cost - prev_cost

                self._iter_end_time = time.perf_counter()
                self._print_verbose_msg_iter_end(n_iter, cost, change)

                cost_increased = abs(change) > self.tol and change > 0
                if cost_increased and self.require_monotonic_cost_decrease:
                    warnings.warn(
                        'Cost increased during iteration in initialization %d' %
                        (init + 1), UserWarning)
                    break

                if abs(change) < self.tol:
                    converged = True
                    break

            self._init_end_time = time.perf_counter()
            self._print_verbose_msg_init_end(cost, converged)

            if cost < min_cost:
                min_cost = cost
                best_weights = self.weights_

            if not converged:
                warnings.warn('Initialization %d did not converge. '
                              'Try different init parameters, '
                              'or increase max_iter, tol '
                              'or check for degenerate data.'
                              % (init + 1), ConvergenceWarning)

        log_likelihood = self._log_likelihood(X, exog=exog)
        rmse = self._rmse(X, exog=exog)

        # restore value of TV norm from original fit
        self._max_tv_norm = old_max_tv_norm

        return best_weights, min_cost, log_likelihood, rmse

    def _check_initial_parameters(self, X, exog=None):
        """Check values of the basic parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        exog : array-like, shape (n_samples, n_exog), optional
        """
        if not isinstance(self.n_components, INTEGER_TYPES) or self.n_components < 1:
            raise ValueError("Invalid value for 'n_components': %d "
                             "Estimation requires at least one component"
                             % self.n_components)

        if self.tol < 0.:
            raise ValueError("Invalid value for 'tol': %.5f "
                             "Tolerance used by the EM must be non-negative"
                             % self.tol)

        if not isinstance(self.n_init, INTEGER_TYPES) or self.n_init < 1:
            raise ValueError("Invalid value for 'n_init': %d "
                             "Estimation requires at least one run"
                             % self.n_init)

        if not isinstance(self.max_iter, INTEGER_TYPES) or self.max_iter < 1:
            raise ValueError("Invalid value for 'max_iter': %d "
                             "Estimation requires at least one iteration"
                             % self.max_iter)

        if self.reg_covar < 0.:
            raise ValueError("Invalid value for 'reg_covar': %.5f "
                             "regularization on covariance must be "
                             "non-negative"
                             % self.reg_covar)

        if self.state_length is not None and self.state_length < 0:
            raise ValueError("Invalid value for 'state_length': %.5f "
                             "average state length must be non-negative"
                             % self.state_length)

        if isinstance(self.orders, INTEGER_TYPES):
            self.orders = np.full((self.n_components,), self.orders)

        self.orders = _check_orders(self.orders, self.n_components)
        self.max_order_ = np.max(self.orders)

    def _check_parameters(self, X, exog=None):
        """Check that the model parameters are well-defined.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        exog : array-like, shape (n_samples, n_exog), optional
        """
        n_samples, n_features = X.shape

        if self.weights_init is not None:
            self.weights_init = _check_weights(
                self.weights_init, n_samples - self.presample_length,
                self.n_components)

        if self.means_init is not None:
            self.means_init = _check_means(self.means_init,
                                           self.n_components, n_features)

        if self.endog_coef_init is not None:
            self.endog_coef_init = _check_endog_coefs(
                self.endog_coef_init, self.n_components, n_features,
                self.max_order_)

        if exog is not None and self.exog_coef_init is not None:
            _, n_exog = exog.shape
            self.exog_coef_init = _check_exog_coefs(
                self.exog_coef_init, self.n_components, n_features, n_exog)

        if self.precisions_init is not None:
            self.precisions_init = _check_precisions(
                self.precisions_init, self.n_components, n_features)

    def _stack_predictors(self, X, exog=None):
        """Construct stacked matrices of lagged predictor variables."""
        self._Y_k = {k: None for k in range(self.n_components)}
        self._Theta_k = {k: None for k in range(self.n_components)}

        n_samples, n_features = X.shape

        if exog is None:
            n_exog = 0
        else:
            n_exog = exog.shape[1]

        for k in range(self.n_components):

            pk = self.orders[k]

            self._Y_k[k] = np.empty(
                (n_samples - self.presample_length, 1 + pk * n_features + n_exog),
                dtype=X.dtype)
            self._Theta_k[k] = np.zeros(
                (1 + pk * n_features + n_exog, n_features), dtype=X.dtype)

            # intercept term
            self._Y_k[k][:, 0] = 1.0

            # lagged endogeneous variables, if present
            col_index = 1
            if pk > 0:
                for i in range(1, pk + 1):
                    self._Y_k[k][:, col_index:col_index + n_features] = \
                        X[self.presample_length - i:-i]
                    col_index += n_features

            # exogeneous variables, if present
            if n_exog > 0:
                self._Y_k[k][:, col_index:] = exog[self.presample_length:]

    def _initialize(self, X, random_state, exog=None):
        """Initialize the weights and model parameters.

        Parameters
        ----------
        X : array-like, shape  (n_samples, n_features)

        random_state : RandomState
            A random number generator instance.

        exog : array-like, shape (n_samples, n_exog), optional
        """
        n_samples, _ = X.shape

        if self.init_params is None:
            self.init_params = 'random'

        # first initialize the model weights
        if self.init_params == 'kmeans':

            weights = np.zeros(
                (n_samples - self.presample_length, self.n_components))
            labels = cluster.KMeans(n_clusters=self.n_components, n_init=1,
                                    random_state=random_state).fit(X).labels_
            weights[np.arange(n_samples - self.presample_length),
                    labels[self.presample_length:]] = 1

        elif self.init_params == 'random':
            weights = random_state.rand(
                n_samples - self.presample_length, self.n_components)
            weights /= weights.sum(axis=1)[:, np.newaxis]

        else:
            raise ValueError("Unimplemented initialization method '%s'"
                             % self.init_params)

        self.weights_ = (weights if self.weights_init is None
                         else self.weights_init)

        self._initialize_parameters(X, self.weights_, exog=exog)

    def _initialize_parameters(self, X, weights, exog=None):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like, shape  (n_samples, n_features)

        weights : array-like, shape (n_samples - presample_length, n_components)

        exog : array-like, shape (n_samples, n_exog), optional
        """
        params, covariances = self._estimate_varx_parameters(
            X, weights, self.reg_covar)

        means, endog_coef, exog_coef = self._unpack_flattened_parameters(
            X, params, exog=exog)

        self.means_ = means if self.means_init is None else self.means_init
        self.endog_coef_ = (endog_coef if self.endog_coef_init is None
                            else self.endog_coef_init)
        self.exog_coef_ = (exog_coef if self.exog_coef_init is None
                           else self.exog_coef_init)

        self._set_flattened_parameters(
            self.means_, endog_coef=self.endog_coef_, exog_coef=self.exog_coef_)

        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(covariances)
        else:
            self.precisions_cholesky_ = np.array(
                [sl.cholesky(prec_init, lower=True)
                 for prec_init in self.precisions_init])
            self.covariances_ = _compute_covariances(
                self.precisions_cholesky_)

    def _component_negative_log_likelihoods(self, residuals, k):
        """Compute negative log-likelihood associated with the given component."""
        n_features = residuals.shape[1]

        res = np.dot(residuals, self.precisions_cholesky_[k])

        const = (0.5 * n_features * np.log(2 * np.pi) +
                 0.5 * np.linalg.slogdet(self.covariances_[k])[1])

        res2 = 0.5 * np.sum(res ** 2, axis=1)

        return const + res2

    def _log_likelihood(self, X, exog=None):
        """Calculate log-likelihood of the given data under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        exog : array-like, shape (n_samples, n_exog), optional

        Returns
        -------
        log_likelihood : float
            The log-likelihood of the data.
        """
        n_samples = X.shape[0]

        log_probabilities = np.empty(
            (n_samples - self.presample_length, self.n_components))

        for k in range(self.n_components):

            res = (X[self.presample_length:] -
                   np.dot(self._Y_k[k], self._Theta_k[k]))

            log_probabilities[:, k] = (
                np.log(self.weights_[:, k]) -
                self._component_negative_log_likelihoods(res, k))

        log_likelihood = 0
        for log_pk in log_probabilities:

            log_likelihood += logsumexp(log_pk)

        return log_likelihood

    def _rmse(self, X, exog=None):
        """Calculate value of weighted RMSE."""
        n_samples, n_features = X.shape

        res = np.empty(
            (self.n_components, n_samples - self.presample_length, n_features),
            dtype=X.dtype)

        for k in range(self.n_components):

            res[k] = self.weights_[:, k][:, np.newaxis] * (
                X[self.presample_length:] - np.dot(self._Y_k[k], self._Theta_k[k]))**2

        rmse = np.mean(np.mean(np.sum(res, axis=0), axis=0)**0.5)

        return rmse

    def _compute_distance_matrix(self, X):
        """Calculate matrix of loss function values."""
        n_samples, _ = X.shape
        distance_matrix = np.empty(
            (n_samples - self.presample_length, self.n_components),
            dtype=X.dtype)

        for k in range(self.n_components):

            res = X[self.presample_length:] - np.dot(self._Y_k[k], self._Theta_k[k])

            if self.loss == 'log_likelihood':

                distance_matrix[:, k] = self._component_negative_log_likelihoods(res, k)

            elif self.loss == 'least_squares':

                distance_matrix[:, k] = 0.5 * np.sum(res ** 2, axis=1) / n_samples

            else:
                raise ValueError("Invalid loss '%s'" % self.loss)

        return distance_matrix

    def _compute_cost(self, X, weights):
        """Calculate cost function."""
        distance_matrix = self._compute_distance_matrix(X)

        return np.sum(weights * distance_matrix)

    def _unpack_flattened_parameters(self, X, params, exog=None):
        """Extract parameters from flat matrix."""
        n_features = X.shape[1]

        means = np.empty((self.n_components, n_features))
        for k in range(self.n_components):
            means[k] = params[k][0]

        if self.max_order_ > 0:
            endog_coef = np.zeros(
                (self.n_components, self.max_order_, n_features, n_features))
            for k in range(self.n_components):
                pk = self.orders[k]
                if pk > 0:
                    n_endog_predictors = pk * n_features
                    endog_coef[k, :pk, :, :] = np.reshape(
                        params[k][1:n_endog_predictors + 1],
                        (pk, n_features, n_features)).swapaxes(1, 2)
        else:
            endog_coef = None

        if exog is not None:
            n_exog = exog.shape[1]
            exog_coef = np.zeros(
                (self.n_components, n_features, n_exog))
            for k in range(self.n_components):
                exog_coef[k] = params[k][-n_exog:].T
        else:
            exog_coef = None

        return means, endog_coef, exog_coef

    def _estimate_varx_parameters(self, X, weights, reg_covar):
        """Calculate optimal VARX parameters."""
        n_features = X.shape[1]

        params = {k: None for k in range(self.n_components)}
        covariances = np.empty((self.n_components, n_features, n_features))

        nk = np.sum(weights, axis=0)
        for k in range(self.n_components):

            Wk2 = weights[:, k][:, np.newaxis]

            lhs = np.dot(self._Y_k[k].T, Wk2 * self._Y_k[k])
            rhs = np.dot(self._Y_k[k].T, Wk2 * X[self.presample_length:])

            params[k] = sl.lstsq(lhs, rhs)[0]

            res = X[self.presample_length:] - np.dot(self._Y_k[k], params[k])

            if self.loss == 'log_likelihood':
                # compute maximum likelihood estimate of covariance matrix
                covariances[k] = np.dot(res.T, Wk2 * res) / nk[k]
                covariances[k].flat[::n_features + 1] += reg_covar
            elif self.loss == 'least_squares':
                # compute least squares estimate of covariance matrix
                if nk[k] <= 1.0:
                    warnings.warn(
                        'Effective degrees of freedom is less than 1 '
                        'for component %d (nk[%d]=%.2f)' %
                        (k, k, nk[k]), UserWarning)

                covariances[k] = np.dot(res.T, Wk2 * res) / np.abs(nk[k] - 1)
                covariances[k].flat[::n_features + 1] += reg_covar
            else:
                raise ValueError("Invalid loss '%d'" % self.loss)

        return params, covariances

    def _update_parameters(self, X, weights, exog=None):
        """Update model parameters with fixed component weights."""
        self._Theta_k, self.covariances_ = self._estimate_varx_parameters(
            X, weights, self.reg_covar)

        self.means_, self.endog_coef_, self.exog_coef_ = \
            self._unpack_flattened_parameters(X, self._Theta_k, exog=exog)

        # compute precision matrix via Cholesky decomposition
        self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_)

    def _update_weights(self, X):
        """Update model weights with fixed model parameters."""
        distance_matrix = self._compute_distance_matrix(X)

        self._fembv_weights.update_weights(distance_matrix)

        self.weights_ = self._fembv_weights.weights

    def _get_parameters(self):
        """Get parameters for model components."""
        return (self.means_, self.endog_coef_, self.exog_coef_,
                self.covariances_, self.precisions_cholesky_)

    def _set_flattened_parameters(self, means, endog_coef=None, exog_coef=None):
        """Construct flattened parameter matrix."""
        n_features = means.shape[-1]

        for k in range(self.n_components):

            self._Theta_k[k][0] = means[k]

            if endog_coef is not None:

                pk = self.orders[k]
                n_endog_predictors = self.orders[k] * n_features

                self._Theta_k[k][1:n_endog_predictors + 1, :] = np.reshape(
                    endog_coef[k][:pk, ...].swapaxes(1, 2),
                    (n_endog_predictors, n_features))

            if exog_coef is not None:

                n_exog = exog_coef.shape[-1]

                self._Theta_k[k][-n_exog:] = exog_coef[k, :, :].T

    def _set_parameters(self, params):
        """Set model parameters."""
        (self.means_, self.endog_coef_, self.exog_coef_,
         self.covariances_, self.precisions_cholesky_) = params

        # construct flattened parameter matrices
        self._set_flattened_parameters(
            self.means_, endog_coef=self.endog_coef_, exog_coef=self.exog_coef_)

        # construction precision matrices
        self.precisions_ = np.empty(self.precisions_cholesky_.shape)
        for k, prec_chol in enumerate(self.precisions_cholesky_):
            self.precisions_[k] = np.dot(prec_chol, prec_chol.T)

    def _print_verbose_msg_init_beg(self, n_init):
        """Print verbose message on initialization."""
        if self.verbose > 0:
            print('FEM-BV-VARX (n_components = {:d}): initialization {:d}'.format(
                self.n_components, n_init + 1))
            print('{:<12s} | {:<13s} | {:<13s} | {:<12s}'.format(
                'Iteration', 'Cost', 'Cost delta', 'Time'))
            print(60 * '-')

    def _print_verbose_msg_init_end(self, cost, converged):
        """Print verbose message on the end of initialization."""
        if self.verbose > 0:
            print(
                'Initialization converged: {} (total time: {:12.6e}, final cost: {:12.6e})'.format(
                    converged, self._init_end_time - self._init_start_time,
                    cost))

    def _print_verbose_msg_iter_end(self, n_iter, new_cost, cost_delta):
        """Print verbose message on iteration."""
        if n_iter % self.verbose_interval == 0:
            if self.verbose > 0:
                print('{:12d} | {: 12.6e} | {: 12.6e} | {: 12.6e}'.format(
                    n_iter + 1, new_cost, cost_delta,
                    self._iter_end_time - self._iter_start_time))
